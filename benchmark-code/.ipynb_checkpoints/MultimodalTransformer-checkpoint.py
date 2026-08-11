import torch
from torch import nn
from ModalityEncoder import ModalityEncoder
from utils import compute_attention_rollout

class CustomLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention layer
        attn_output, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        # Apply dropout and residual connection
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return (
            src,
            attn_weights,
        )  # Return both transformed features and attention weights

class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        mask_missing,
        all_modalities,
        modality_names, # only enabled modalities
        d_model=128,
        num_heads=4,
        num_layers=7,
        dropout=0.1,
    ):
        super().__init__()

        num_classes = 1 

        self.mask_missing = mask_missing
        
        self.modality_enc = ModalityEncoder(modalities=all_modalities, d_model=d_model)

        self.d_model = d_model

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.modalities = all_modalities # all modalities in config file
        
        self.modality_names = modality_names

        # === Shared Transformer Encoder ===
        self.layers = nn.ModuleList(
            [
                CustomLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, batch):
    
        B = len(batch["patient_ids"])
    
        tokens = []
        attention_mask = []
        token_names_batch = []
    
        # CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        tokens.append(cls_token)
        attention_mask.append(torch.zeros(B, 1, dtype=torch.bool, device=cls_token.device))
        token_names_batch.append([["CLS"] for _ in range(B)])
    
        # === DYNAMIC MODALITY LOOP ===
        for mod in self.modality_names:
            if mod in batch["out_batch_tab_feats"]:
                x = batch["out_batch_tab_feats"][mod].unsqueeze(1)
                mask = batch["out_batch_tab_mask"][mod]
                x = self.modality_enc(x, mod)
                _, tkn_len, _ = x.shape
               
                tokens.append(x)                
                attention_mask.append(mask)

                if self.modalities[mod].single_token:
                    token_names_batch.append(
                        [[mod.upper()] for _ in range(B)]
                    )
                else:
                    feat_names = self.modalities[mod].token_names
            
                    token_names_batch.append(
                        [feat_names for _ in range(B)]
                    )
        
        
            elif mod in batch["out_batch_img_feats"]:
                x = batch["out_batch_img_feats"][mod]   # (B, S, D)
                mask = batch["out_batch_img_mask"][mod] # (B, S)
                x = self.modality_enc(x, mod)                            

                if self.modalities[mod].single_token:
                    # valid scans = 1, padded scans = 0
                    valid = (~mask).float().unsqueeze(-1)    # (B, S, 1)
                    # masked mean over scans
                    x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)   
                    # x is now (B, D)
                    x = x.unsqueeze(1)      # (B, 1, D)
                    mask = mask.all(dim=1, keepdim=True)
                    token_names_batch.append([[f"{mod.upper()}_MEAN"] for _ in range(B)])
                else:
                    token_names_batch.append(batch["batch_img_filenames_dict"][mod])
                    
                tokens.append(x)
                attention_mask.append(mask)

            else:
                # modality completely missing
                tokens.append(torch.zeros_like(cls_token))

                if self.mask_missing:
                    attention_mask.append(torch.ones(B, 1, dtype=torch.bool, device=cls_token.device)) # TRUE
                else:
                    attention_mask.append(torch.zeros(B, 1, dtype=torch.bool, device=cls_token.device)) # FALSE, let model see zeroed tokens
        
                token_names_batch.append(
                    [[f"{mod.upper()}_PAD"] for _ in range(B)]
                )
    
        # concat tokens
        x = torch.cat(tokens, dim=1)
        
        attn_mask = torch.cat(attention_mask, dim=1)
        attn_mask = attn_mask.to(dtype=torch.bool, device=x.device)

        if not self.mask_missing:
            # let model see zeroed tokens => attend to all tokens (except padded img scans) (all mask FALSE)
            attn_mask = torch.zeros_like(attn_mask, dtype=torch.bool)
        
        #pooled = x.mean(dim=1)      # (batch_size, d_model)
        #pooled = x.max(dim=1).values
        #out = self.fc(pooled)
        
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_key_padding_mask=attn_mask)
            attn_weights_all.append(attn_weights)
        
        cls = x[:, 0, :]
        
        out = self.fc(cls)
    
        final_token_names_batch = [
            sum([mods[i] for mods in token_names_batch], [])
            for i in range(B)
        ]

        rollout = compute_attention_rollout(attn_weights_all, attn_mask)

        return {
            "outputs": out,
            "attn_weights_all": attn_weights_all,
            "final_token_names_batch": final_token_names_batch,
            "rollout": rollout,
        }
            