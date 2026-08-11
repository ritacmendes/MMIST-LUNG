from torch import nn
import torch

class FeatureTokenizer(nn.Module):
    def __init__(self, dim, d_model):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.feature_embed = nn.Embedding(dim, d_model)

    def forward(self, x):
        B, N, D = x.shape
        x = x.unsqueeze(-1)                     # (B,N,D,1)
        h = self.proj(x)                        # (B,N,D,d_model)

        feat_ids = torch.arange(D, device=x.device)
        h = h + self.feature_embed(feat_ids)

        return h.view(B, N*D, -1)

class ModalityEncoder(nn.Module):
    def __init__(self, modalities, d_model):
        super().__init__()

        self.encoders = nn.ModuleDict()
        self.proj = nn.ModuleDict()  # for images (conv)

        for name, mod in modalities.items():
            dim = mod.feat_dim
            mtype = mod.type
            single_token_flag =  mod.single_token

            if mtype == "tabular":
                if single_token_flag:
                    self.encoders[name] = nn.Sequential(
                    nn.Linear(dim, 1024),
                    nn.GELU(),
                    nn.Dropout(0.3),
                
                    nn.Linear(1024, 256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                
                    nn.Linear(256, d_model),
                    nn.LayerNorm(d_model),
                )
                else:
                    self.encoders[name] = FeatureTokenizer(dim, d_model)

            elif mtype == "image":
                self.proj[name] = nn.Linear(dim, d_model)

                self.encoders[name] = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.LayerNorm(d_model),
                )

            else:
                raise ValueError(f"Unknown type: {mtype}")

    def forward(self, x, modality):
        m = modality.lower()

        x = x.float()
        if m in self.proj:
            proj_x = self.proj[m](x)
        else:
            proj_x = x

        h = self.encoders[m](proj_x)

        return h
            
