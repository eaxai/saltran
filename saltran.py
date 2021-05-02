import torch.nn as nn
from self_attention_cv import ResNet50ViT

class SalTran(nn.Module):
    def __init__(self):
        super(SalTran, self).__init__()
        self.xformer = ResNet50ViT(img_dim=256, pretrained_resnet=True, classification=False)
        self.salmap = nn.Sequential(
            nn.Unflatten(2, (64, 64)),
            nn.Conv2d(512, 1, 1)
        )

    def forward(self, x):
        out = self.xformer(x)
        out = out.permute(0, 2, 1)
        out = self.salmap(out)
        return out