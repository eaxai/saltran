import numpy as np
import torch
import torch.nn as nn
import torchvision

'''
Input  -> FeatExtract -> VGG16 -> PosEncode -> MHSA -> FFN -----I
Output ->                      -> PosEncode -> MHSA --------> MHSA -> FFN -> Output (org. Linear -> Softmax, we might need a 1x1 conv or something like that)
'''

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        m = torchvision.models.vgg16(pretrained=True)
        for param in m.parameters():
            param.requires_grad = False
        del m.avgpool
        del m.classifier
        del m.layers[30]
        md = list(m.features)
        md.insert(24, torch.nn.ZeroPad2d((0,1,0,1)))
        m.features = torch.nn.Sequential(*md)
        m.features[23].stride=1
        m.features[25].padding = (2,2)
        m.features[25].dilation = (2,2)
        m.features[27].padding = (2,2)
        m.features[27].dilation = (2,2)
        m.features[29].padding = (2,2)
        m.features[29].dilation = (2,2)
        self.vgg = m.features


    def forward(self, data):
        out = self.vgg(data)
        return out



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class SalTran(nn.Module):
    def __init__(self):
        super(SalTran, self).__init__()
        self.vgg = FeatureExtractor() # VGG16-based feature extractor
        self.pe = PositionalEncoding(1200)
        enc_layer = nn.TransformerEncoderLayer(512, 8)
        self.enc = nn.TransformerEncoder(enc_layer, 6)
        self.dec =





    def forward(self, x, mask=None):
        src =  self.pe( )