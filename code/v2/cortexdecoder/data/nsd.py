import gc
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from scipy.io import loadmat
from tqdm.auto import tqdm

from ..core.common import require_file, torch_load_cpu
from ..core.config import CLIP_DIM, CLIP_NUM_QUERIES, TRIALS_PER_SESSION


class NSDSubjectData:
    """Flatmap sessions, image-id splits and CLIP targets for one NSD subject."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.subject = cfg.subject
        self.flatmap_dir = f"{cfg.data_dir}/flatmap_tokens_fp16"
        self.clip_targets_dir = f"{cfg.data_dir}/clip_targets_cache"

        mat = loadmat(require_file(f"{cfg.data_dir}/nsd_expdesign.mat"))
        masterordering = mat["masterordering"].reshape(-1).astype(np.int64) - 1
        subjectim = mat["subjectim"].astype(np.int64) - 1
        self.shared_ids = set(int(i) for i in mat["sharedix"].reshape(-1).astype(np.int64) - 1)
        self.imgbrick_ids = subjectim[int(self.subject[-2:]) - 1, masterordering]

        self.sessions = self._discover_sessions()
        print(f"{self.subject}: {len(self.sessions)} sessions {self.sessions}")

        # Image-id holdout: shared1000 is the final test set; VAL_FRAC of the remaining images is validation.
        self.final_heldout_ids = set(self.shared_ids)
        nonshared = np.array(sorted({
            int(img_id)
            for sess in self.sessions
            for img_id in self.ids_for_session(sess, TRIALS_PER_SESSION)
            if int(img_id) not in self.final_heldout_ids
        }), dtype=np.int64)
        perm = np.random.RandomState(cfg.seed).permutation(nonshared)
        self.val_heldout_ids = set(map(int, perm[:int(cfg.val_frac * len(perm))]))
        self.train_blocked_ids = self.final_heldout_ids | self.val_heldout_ids
        self._assert_clean_splits()

        sample = self.load_flatmap_tokens(self.sessions[0])
        if sample.ndim != 3:
            raise ValueError(f"expected flatmap tokens [trials, patches, dim], got {tuple(sample.shape)}")
        self.num_patches, self.token_dim = int(sample.shape[1]), int(sample.shape[2])
        print(f"flatmap direct tokens: patches={self.num_patches} dim={self.token_dim}")

        self.bad_sessions = set()
        self._session_cache = {}
        self._clip_targets = None

    # --- flatmap sessions ---------------------------------------------------

    def _flatmap_candidates(self, sess):
        return [
            f"{self.flatmap_dir}/{self.subject}_session{sess:02d}_fp16.pt",
            f"{self.flatmap_dir}/{self.subject}_flatmap_tokens_session{sess:02d}.pt",
        ]

    def _discover_sessions(self):
        if not os.path.isdir(self.flatmap_dir):
            raise FileNotFoundError(self.flatmap_dir)
        nums = []
        for name in os.listdir(self.flatmap_dir):
            if not name.startswith(self.subject):
                continue
            match = re.search(r"session(\d+)", name)
            if match:
                nums.append(int(match.group(1)))
        bad = set(self.cfg.known_bad_sessions)
        nums = sorted(set(s for s in nums if (self.subject, s) not in bad))
        if len(nums) < 2:
            raise RuntimeError(f"{self.subject} needs at least 2 usable sessions, found {nums}")
        return nums

    def load_flatmap_tokens(self, sess):
        for path in self._flatmap_candidates(sess):
            if os.path.isfile(path):
                break
        else:
            raise FileNotFoundError(f"Missing flatmap session {self.subject} {sess:02d}. Checked: {self._flatmap_candidates(sess)}")
        tokens = torch_load_cpu(path)
        if tokens.dtype != torch.float16:
            tokens = tokens.half()
        if not torch.isfinite(tokens).all():
            bad = int((~torch.isfinite(tokens)).sum().item())
            raise ValueError(f"{self.subject} session {sess:02d} has {bad} non-finite flatmap token values")
        return tokens.contiguous()

    def ids_for_session(self, sess, n_rows):
        start = (sess - 1) * TRIALS_PER_SESSION
        return self.imgbrick_ids[start:start + n_rows].astype(np.int64)

    # --- splits -------------------------------------------------------------

    def split_keep_mask(self, img_ids, split):
        ids = [int(i) for i in img_ids.tolist()]
        if split == "train":
            keep = [i not in self.train_blocked_ids for i in ids]
        elif split == "val":
            keep = [i in self.val_heldout_ids for i in ids]
        elif split in {"all", "eval"}:
            keep = [True] * len(ids)
        else:
            raise ValueError(f"unknown split={split!r}")
        return torch.as_tensor(keep, dtype=torch.bool)

    def _assert_clean_splits(self):
        seen = {}
        for split in ("train", "val"):
            seen[split] = set()
            for sess in self.sessions:
                ids = torch.from_numpy(self.ids_for_session(sess, TRIALS_PER_SESSION)).long()
                seen[split].update(int(i) for i in ids[self.split_keep_mask(ids, split)].tolist())
        assert seen["train"].isdisjoint(self.final_heldout_ids), "train/final image-id overlap"
        assert seen["train"].isdisjoint(self.val_heldout_ids), "train/val-heldout image-id overlap"
        assert seen["val"].isdisjoint(self.final_heldout_ids), "val/final image-id overlap"
        assert seen["train"].isdisjoint(seen["val"]), "train/val image-id overlap"
        print(
            f"{self.subject}: clean image-id split train_unique={len(seen['train'])} "
            f"val_unique={len(seen['val'])} final_heldout={len(self.final_heldout_ids)}"
        )

    def load_session(self, sess, split="train"):
        """Return (brain_tokens, img_ids) for one session filtered to `split`, or None to skip it."""
        key = (int(sess), split)
        # Val trials are ~5% of each session (a few GB total), so they are always cached; re-reading every
        # session from the volume each epoch just to pull them dominated the val pass.
        cache = self.cfg.cache_sessions_in_ram or split == "val"
        if cache and key in self._session_cache:
            return self._session_cache[key]
        if int(sess) in self.bad_sessions:
            return None
        try:
            tokens = self.load_flatmap_tokens(sess)
            ids = torch.from_numpy(self.ids_for_session(sess, tokens.shape[0])).long()
            keep = self.split_keep_mask(ids, split)
            tokens, ids = tokens[keep], ids[keep]
            if tokens.shape[0] == 0:
                return None
            if cache:
                self._session_cache[key] = (tokens, ids)  # boolean indexing above already copied the rows
            return tokens, ids
        except Exception as exc:
            print(f"[bad session] {self.subject} session {sess:02d}: {exc}; skipping")
            self.bad_sessions.add(int(sess))
            return None

    def expected_batches(self, split, batch_size):
        total = 0
        for sess in self.sessions:
            ids = torch.from_numpy(self.ids_for_session(sess, TRIALS_PER_SESSION)).long()
            n_rows = int(self.split_keep_mask(ids, split).sum().item())
            total += -(-n_rows // batch_size)
        return total

    # --- CLIP targets -------------------------------------------------------

    @property
    def clip_targets(self):
        if self._clip_targets is None:
            path = require_file(f"{self.clip_targets_dir}/{self.subject}_clip_targets.pt")
            self._clip_targets = torch_load_cpu(path)
            print(f"loaded CLIP targets {self.subject}: {len(self._clip_targets)} images")
        return self._clip_targets

    def clip_targets_for_ids(self, img_ids):
        targets = torch.stack([self.clip_targets[int(i)] for i in img_ids]).contiguous()
        if tuple(targets.shape[1:]) != (CLIP_NUM_QUERIES, CLIP_DIM):
            raise ValueError(f"expected CLIP targets [B, {CLIP_NUM_QUERIES}, {CLIP_DIM}], got {tuple(targets.shape)}")
        if not torch.isfinite(targets).all():
            bad = int((~torch.isfinite(targets)).sum().item())
            raise ValueError(f"CLIP target batch has {bad} non-finite values")
        return targets

    # --- shared-1000 test set -----------------------------------------------

    def collect_shared1000(self, out_dir):
        """Trial-averaged flatmap tokens for every shared-1000 image, cached in `out_dir`."""
        tokens_path = shared_tokens_path(out_dir, self.subject)
        ids_path = shared_ids_path(out_dir, self.subject)
        if os.path.exists(tokens_path) and os.path.exists(ids_path):
            with open(ids_path) as f:
                ids = json.load(f)
            if any(int(i) not in self.final_heldout_ids for i in ids):
                raise ValueError(f"{self.subject} cached shared1000 IDs include non-heldout image IDs")
            return torch_load_cpu(tokens_path), ids
        groups = defaultdict(list)
        for sess in tqdm(self.sessions, desc=f"collect shared-1000 flatmaps {self.subject}"):
            tokens = self.load_flatmap_tokens(sess)
            for row, img_id in enumerate(self.ids_for_session(sess, tokens.shape[0])):
                if int(img_id) in self.shared_ids:
                    groups[int(img_id)].append(tokens[row])
            del tokens
            gc.collect()
        ids = sorted(groups)
        avg = torch.stack([torch.stack(groups[i]).mean(0) for i in ids]).half().contiguous()
        torch.save(avg, tokens_path)
        with open(ids_path, "w") as f:
            json.dump([int(i) for i in ids], f)
        return avg, ids


def shared_tokens_path(out_dir, subject):
    return f"{out_dir}/{subject}_shared1000_flatmap_source_tokens.pt"


def shared_ids_path(out_dir, subject):
    return f"{out_dir}/{subject}_shared1000_img_ids.json"


def final_eval_indices(cfg, n_available):
    """Deterministic subset of shared-1000 rows used for DIP/diffusion/eval."""
    n_requested = min(cfg.final_eval_images, n_available)
    if n_requested >= n_available:
        return list(range(n_available))
    rng = np.random.RandomState(cfg.seed)
    return sorted(rng.choice(n_available, size=n_requested, replace=False).astype(int).tolist())


def rows_for_ids(source_ids, wanted_ids, label):
    lookup = {int(img_id): row for row, img_id in enumerate(source_ids)}
    missing = [int(i) for i in wanted_ids if int(i) not in lookup]
    if missing:
        raise ValueError(f"{label} is missing {len(missing)} requested image ids; first missing ids={missing[:8]}")
    return torch.as_tensor([lookup[int(i)] for i in wanted_ids], dtype=torch.long)
