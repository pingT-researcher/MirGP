import torch
import torch.nn as nn
import numpy as np
import math
class MixConv1d(nn.Module):
    def __init__(self, c1, c2, k=(1, 3, 5), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:
            i = torch.linspace(0, groups - 1E-6, c2).floor()
            c_ = [(i == g).sum() for g in range(groups)]
        else:
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k)
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()

        self.m = nn.ModuleList([
            nn.Conv1d(c1, int(c_[g]), k[g], stride=s, padding=k[g] // 2, bias=False)
            for g in range(groups)
        ])
        self.ln = nn.LayerNorm(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.match_channels = nn.Conv1d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        tmp = torch.cat([m(x) for m in self.m], dim=1)
        tmp = tmp.transpose(1, 2)
        tmp = self.ln(tmp)
        out = self.act(tmp.transpose(1, 2))
        x = self.match_channels(x)
        return x + out
    
class eca_layer(nn.Module):
    def __init__(self, channels, k_size=None,
                 gamma=2.0, b=1.0, max_k=9):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(channels) / gamma) + b))
            k_size = t if t % 2 == 1 else t + 1
            k_size = max(3, min(k_size, max_k if max_k % 2 == 1 else max_k - 1))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)                        # (B, C, 1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k_size,
                                  padding=k_size // 2, bias=False)     # 在通道维做卷积
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 4 and x.size(-1) == 1:
            x = x.squeeze(-1)  # -> (B, C, L)

        y = self.avg_pool(x)               # (B, C, 1)
        y = y.permute(0, 2, 1)             # (B, 1, C)
        y = self.conv(y)                   # (B, 1, C)
        y = self.sigmoid(y).permute(0, 2, 1)  # (B, C, 1)

        return x * y.expand_as(x)
    
    
class MirGPModel(nn.Module):
    def __init__(self, snp_dim, te_dim, hidden_dim=1000, conv_c2=10, gcn_dim1=10, gcn_dim2=10, input_dim1=10, input_dim2=10):
        super().__init__()

        self.snp_cnn = nn.Sequential(
            MixConv1d(1, conv_c2, k=(1, 3, 5, 7)),
            MixConv1d(conv_c2, conv_c2, k=(1, 3, 5, 7)),
            nn.MaxPool1d(kernel_size=2),
        )

        self.te_cnn = nn.Sequential(
            MixConv1d(1, conv_c2, k=(1, 3, 5, 7)),
            MixConv1d(conv_c2, conv_c2, k=(1, 3, 5, 7)),
            nn.MaxPool1d(kernel_size=2),
        )

        self.snp_fc = nn.Linear(snp_dim, hidden_dim)
        self.te_fc = nn.Linear(te_dim, hidden_dim)

        self.eca1 = eca_layer(3)

        self.add_cnn = nn.Sequential(
            MixConv1d(2, conv_c2, k=(1, 3, 5, 7)),
            nn.MaxPool1d(kernel_size=2),
        )
        
        self.eca2 = eca_layer(conv_c2)


        self.integrated_cnn = nn.Sequential(
            MixConv1d(conv_c2, 32, k=(1, 3, 5, 7)),
            nn.AdaptiveAvgPool1d((10)),
            nn.Flatten()
        )

        self.snp_l = nn.Linear(math.ceil(gcn_dim1 / 2), snp_dim // 2)
        self.te_l = nn.Linear(math.ceil(gcn_dim2 / 2), te_dim // 2)
        self.pre_out = nn.Sequential(nn.Linear(320, 128), nn.GELU(), nn.Dropout(p=0.2))
        self.out_fc = nn.Linear(128, 1)

    def forward(self, x_snp, x_te, re1=None, re2=None):
        x_snp = x_snp.unsqueeze(1)
        x_te = x_te.unsqueeze(1)
        cnn_snp = self.snp_cnn(x_snp)
        cnn_te = self.te_cnn(x_te)
        
        fc_snp = self.snp_fc(x_snp)
        fc_te = self.te_fc(x_te)

        merge = self.eca1(torch.cat((fc_snp, fc_te), dim=1))
        add = self.add_cnn(merge)

        integrated = self.eca2(torch.cat((cnn_snp, cnn_te, add), dim=2))
        out = self.integrated_cnn(integrated)
        out = self.pre_out(out)
        out = self.out_fc(out)

        return out.squeeze(-1)
