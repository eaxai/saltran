import torch.nn as nn
import torch
from self_attention_cv import ResNet50ViT

class SalTran(nn.Module):
    def __init__(self):
        super(SalTran, self).__init__()
        self.xformer = ResNet50ViT(img_dim=256, pretrained_resnet=True, classification=False)
        self.salmap = nn.Sequential(
            nn.Unflatten(2, (64, 64)),
            nn.LayerNorm(64),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.xformer(x)
        # print("\nXF", out.shape)
        out = out.permute(0, 2, 1)
        # print("\nPERM", out.shape)
        out = self.salmap(out)
        # print("\nSAL", out.shape)
        return out