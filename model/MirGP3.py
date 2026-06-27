import torch
import torch.nn as nn
import numpy as np

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
        self.act = nn.LeakyReLU(0.1, inplace=False)

        self.match_channels = nn.Conv1d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        tmp = torch.cat([m(x) for m in self.m], dim=1)
        tmp = tmp.transpose(1, 2).contiguous()
        tmp = self.ln(tmp)
        out = self.act(tmp.transpose(1, 2).contiguous())
        x = self.match_channels(x)
        return x + out
    
class eca_layer(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).unsqueeze(1).contiguous()
        y = self.conv(y)
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1).contiguous()
        return x * y
    
    
class MirGPModel(nn.Module):
    def __init__(self, snp_dim, te_dim, se_dim, hidden_dim=1000, conv_c2=10,input_dim1=10, input_dim2=10):
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

        
        self.se_cnn = nn.Sequential(
            MixConv1d(1, conv_c2, k=(1, 3, 5, 7)),
            MixConv1d(conv_c2, conv_c2, k=(1, 3, 5, 7)),
            nn.MaxPool1d(kernel_size=2),
        )


        self.snp_fc = nn.Linear(snp_dim, hidden_dim)
        self.te_fc = nn.Linear(te_dim, hidden_dim)
        self.se_fc = nn.Linear(se_dim, hidden_dim)

        self.eca1 = eca_layer()

        self.add_cnn = nn.Sequential(
            MixConv1d(3, conv_c2, k=(1, 3, 5, 7)),
            nn.MaxPool1d(kernel_size=2),
        )
        
        self.eca2 = eca_layer()

        self.integrated_cnn = nn.Sequential(
            MixConv1d(conv_c2, 32, k=(1, 3, 5, 7)),
            nn.AdaptiveAvgPool1d((10)),
            nn.Flatten()
        )
        self.pre_out = nn.Sequential(nn.Linear(320, 128), nn.GELU(), nn.Dropout(p=0.2))
        self.out_fc = nn.Linear(128, 1)

    def forward(self, x_snp, x_te,x_se):
        x_snp = x_snp.unsqueeze(1)
        x_te = x_te.unsqueeze(1)
        x_se = x_se.unsqueeze(1)
        
        cnn_snp = self.snp_cnn(x_snp)
        cnn_te = self.te_cnn(x_te)
        cnn_se = self.se_cnn(x_se)
        
        fc_snp = self.snp_fc(x_snp)
        fc_te = self.te_fc(x_te)
        fc_se = self.se_fc(x_se)

        merge = torch.cat((fc_snp, fc_te, fc_se), dim=1).contiguous()
        merge = self.eca1(merge)

        add = self.add_cnn(merge)

        integrated = torch.cat((cnn_snp, cnn_te, cnn_se, add), dim=2).contiguous()
        integrated = self.eca2(integrated)
        out = self.integrated_cnn(integrated)
        out = self.pre_out(out)
        out = self.out_fc(out)

        return out.squeeze(-1)
