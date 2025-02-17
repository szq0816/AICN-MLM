import torch.nn as nn
from torch.nn.functional import normalize
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.view = view
        self.encoders = []
        self.decoders = []
        self.feature_contrastive_modules = []
        self.embed_projs = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
            self.feature_contrastive_modules.append(
                nn.Sequential(
                    nn.Linear(feature_dim, high_feature_dim),
                    #
                    # nn.Linear(feature_dim, feature_dim),
                    # nn.ReLU(),
                    # nn.Linear(feature_dim, high_feature_dim),
                )
            )

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_modules = nn.ModuleList(self.feature_contrastive_modules)

        self.prototype_modules = nn.Linear(feature_dim, class_num, bias=False)

        self.commonfeature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim * view, high_feature_dim),
        )
        self.mk = torch.nn.Linear(feature_dim * view, 64, bias=False)
        self.mv = torch.nn.Linear(64, feature_dim * view, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            h = self.encoders[v](x)
            z = normalize(self.feature_contrastive_modules[v](h), dim=1)
            xr = self.decoders[v](h)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return zs, qs, xrs, hs, []

    def forward_prototype(self, xs):
        ps = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            p = normalize(self.prototype_modules(z), dim=1)
            ps.append(p)
        return ps



