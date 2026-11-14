import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ..core.common import require_file
from ..core.config import VGG_IMAGE_SIZE


class StimulusImages:
    """Lazy reader for NSD stimulus images (nsd_stimuli.hdf5), indexed by 73k image id."""

    def __init__(self, data_dir):
        self.path = require_file(f"{data_dir}/nsd_stimuli.hdf5")
        self._ds = None

    @property
    def dataset(self):
        if self._ds is None:
            h5 = h5py.File(self.path, "r")
            for key in ["imgBrick", "images", "stimuli"]:
                if key in h5:
                    self._ds = h5[key]
                    break
            else:
                raise KeyError(f"Could not find stimulus image dataset. HDF5 keys={list(h5.keys())}")
        return self._ds

    def for_ids(self, img_ids, size=VGG_IMAGE_SIZE):
        """Images in [0, 1], NCHW, bicubic-resized to `size`."""
        ids = np.asarray([int(i) for i in img_ids], dtype=np.int64)
        unique_ids, inverse = np.unique(ids, return_inverse=True)
        imgs = self.dataset[unique_ids][inverse]
        if imgs.ndim != 4:
            raise ValueError(f"expected NHWC or NCHW images, got {imgs.shape}")
        if imgs.shape[-1] == 3:
            img = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        elif imgs.shape[1] == 3:
            img = torch.from_numpy(imgs)
        else:
            raise ValueError(f"could not infer image channels from {imgs.shape}")
        img = img.float()
        if img.max() > 2:
            img = img / 255.0
        return F.interpolate(img, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
