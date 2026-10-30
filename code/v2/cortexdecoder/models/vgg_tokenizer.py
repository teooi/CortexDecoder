import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..core.config import VGG_IMAGE_SIZE, VGG_LAYER_NAMES, VGG_TOKEN_DIM


class BrainITVGGTokenizer(nn.Module):
    """VGG16-BN feature maps and tokenization described in BrainIT Appendix D.2."""

    def __init__(self):
        super().__init__()
        self.features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features.eval()
        self.capture = {5: 0, 12: 1, 22: 2, 32: 3, 42: 4}
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _repeat_to_512(tokens):
        if tokens.shape[-1] == VGG_TOKEN_DIM:
            return tokens
        reps = math.ceil(VGG_TOKEN_DIM / tokens.shape[-1])
        return tokens.repeat(1, 1, reps)[..., :VGG_TOKEN_DIM]

    def extract_features(self, images01):
        x = F.interpolate(images01, size=(VGG_IMAGE_SIZE, VGG_IMAGE_SIZE), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - self.mean) / self.std
        outs = [None] * len(VGG_LAYER_NAMES)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.capture:
                outs[self.capture[i]] = x
        return outs

    def _tokens_from_feature(self, feat, layer_idx):
        if layer_idx == 0:
            tokens = F.unfold(feat, kernel_size=2, stride=2).transpose(1, 2)
        elif layer_idx == 1:
            tokens = F.unfold(feat, kernel_size=2, stride=1).transpose(1, 2)
        else:
            tokens = feat.flatten(2).transpose(1, 2)
        return self._repeat_to_512(tokens).contiguous()

    def forward(self, images01):
        return [self._tokens_from_feature(feat, i) for i, feat in enumerate(self.extract_features(images01))]
