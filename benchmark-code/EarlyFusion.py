import torch
from torch import nn
from ModalityEncoder import ModalityEncoder

class EarlyFusionModule(nn.Module):
    def __init__(
        self,
        mask_missing,
        all_modalities,
        modality_names, # only enabled modalities
        d_model=128,
        fusion_head='mean',
        instance_agg='mean',
    ):
        super().__init__()

        num_classes = 1 

        self.mask_missing = mask_missing
        
        self.modality_enc = ModalityEncoder(modalities=all_modalities, d_model=d_model)

        self.d_model = d_model

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.modalities = all_modalities # all modalities in config file
        
        self.modality_names = modality_names

        self.fc = nn.Linear(d_model, num_classes)

        self.fc_clin = nn.Linear(15, num_classes)

        self.fusion_head = fusion_head

        self.instance_agg = instance_agg

    def forward(self, batch):
    
        B = len(batch["patient_ids"])
    
        tokens = []
        attention_mask = []
        token_names_batch = []

        if self.fusion_head=='unimodal':
            clinical = torch.cat(
                [
                    batch["out_batch_tab_feats"]["demographics"],
                    batch["out_batch_tab_feats"]["diagnosis"],
                ],
                dim=1,
            )  
            
            out = self.fc_clin(clinical)
            return {
                "outputs": out,
                "attn_weights_all": None,
                "final_token_names_batch": None,
                "rollout": None,
            } 
    
        # === DYNAMIC MODALITY LOOP ===
        for mod in self.modality_names:
            if mod in batch["out_batch_tab_feats"]:
                x = batch["out_batch_tab_feats"][mod].unsqueeze(1)
                
                mask = batch["out_batch_tab_mask"][mod]
                x = self.modality_enc(x, mod)
                _, tkn_len, _ = x.shape

                if self.instance_agg == 'mean':
                    valid = (~mask).unsqueeze(-1).float()
                    x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)                

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

                if self.instance_agg == 'mean':
                    valid = (~mask).float().unsqueeze(-1)    # (B, S, 1)
                    # masked mean over scans
                    x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)   
                
                tokens.append(x)
                attention_mask.append(mask)
    

        stacked = torch.stack(tokens, dim=1)
        if self.fusion_head == 'mean':
            fused = stacked.mean(dim=1)
        elif self.fusion_head == 'max':
            fused, _ = stacked.max(dim=1)
        else:
            fused = tokens
        

        out = self.fc(fused)
  
        return {
            "outputs": out,
            "attn_weights_all": None,
            "final_token_names_batch": None,
            "rollout": None,
        }
    