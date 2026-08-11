import torch
from torch import nn
from ModalityEncoder import ModalityEncoder

class LateFusionModule(nn.Module):
    def __init__(
        self,
        mask_missing,
        all_modalities,
        modality_names, # only enabled modalities
        fusion_head,
        instance_agg,
        d_model,
    ):
        super().__init__()

        num_classes = 1 

        self.fusion_head = fusion_head

        self.instance_agg = instance_agg

        self.mask_missing = mask_missing
        
        self.modality_enc = ModalityEncoder(modalities=all_modalities, d_model=d_model)

        self.d_model = d_model

        self.modalities = all_modalities # all modalities in config file
        
        self.modality_names = modality_names

        self.modality_heads = nn.ModuleDict({
            modality: nn.Linear(d_model, num_classes)
            for modality in self.modality_names
        })

    def pool_tokens(self, x, mask):
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    def forward(self, batch):
    
        B = len(batch["patient_ids"])
    
        tokens = []
        attention_mask = []
        token_names_batch = []
        modality_features = []
    
        predictions = []
    
        # === DYNAMIC MODALITY LOOP ===
        for mod in self.modality_names:
            if mod in batch["out_batch_tab_feats"]:
                x = batch["out_batch_tab_feats"][mod].unsqueeze(1)
                #print("x raw", x.shape)
                mask = batch["out_batch_tab_mask"][mod]
                x = self.modality_enc(x, mod)

                if self.instance_agg == 'mean':
                    x = self.pool_tokens(x, mask)
                    logit = self.modality_heads[mod](x)

                predictions.append(logit)
            
  
            elif mod in batch["out_batch_img_feats"]:
                x = batch["out_batch_img_feats"][mod]   # (B, S, D)
                B, S, D = x.shape
                mask = batch["out_batch_img_mask"][mod]  # (B, S)
            
                valid = ~mask
                
                encoded_valid = self.modality_enc(x[valid], mod)
            
                # Encode only valid tokens
                encoded = torch.zeros(
                    B, S, self.d_model,
                    device=x.device,
                    dtype=encoded_valid.dtype
                )
                encoded[valid] = encoded_valid
            
                # Zero-out padded tokens
                encoded = encoded.masked_fill(mask.unsqueeze(-1), 0.0)
            
                valid_counts = valid.sum(dim=1, keepdim=True).clamp(min=1)
            
                if self.instance_agg == "mean":
                    # Average features, then classify
                    modality_embedding = encoded.sum(dim=1) / valid_counts
                    logit = self.modality_heads[mod](modality_embedding)   # (B,1)
                
                predictions.append(logit)

    
        # Late fusion
        all_predictions = torch.stack(predictions, dim=0)   # (M, B, 1)

        if self.fusion_head == 'mean':
            final_logit = all_predictions.mean(dim=0)           # (B, 1)
        
        elif self.fusion_head == 'max':
            final_logit = all_predictions.max(dim=0).values # (B, 1)
            
        return {
            "outputs": final_logit,
            "attn_weights_all": None,
            "final_token_names_batch": None,
            "rollout": None,
        }
