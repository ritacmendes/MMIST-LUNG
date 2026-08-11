from PatientDataset import PatientDataset
from collate import collate_fn
import numpy as np
from torch.utils.data import Subset
import torch
from torch import nn, optim
from MultimodalTransformer import MultimodalTransformer
from LateFusion import LateFusionModule
from EarlyFusion import EarlyFusionModule
import yaml
from utils import get_class_weights, test_model, train_model_earlystop, print_mod_distribution, load_or_create_folds, aggregate_xgboost_results
from functools import partial
from datetime import datetime
import os 
import json
import warnings
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
# Load config file
with open('config.yaml', "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("On device:", device)

# -- Create Dataset
whole_dataset = PatientDataset(cfg)
print(f"Dataset created with {len(whole_dataset)} patients.")
print_mod_distribution(whole_dataset)

mod_list=''
for mod_info in cfg["modalities"]:
    mod_list += '-'+mod_info['name']

modality_names = [
            m["name"].lower()
            for m in cfg["modalities"]
            if not m.get("mask", False) # dont add modality if mask set to True in config file
        ]
  
# -- Set Train parameters -- 
params = {
    'd_model': 256,
    'num_heads': 4,
    'num_layers': 2,
    'dropout': 0.2723195214333366,
    'num_epochs': 100,
    'batch_size': 32,
    'lr': 0.00013,
    # for early stopping
    'min_train_epochs': 5,
    'patience':20     
}  


# the folds should have all patients in patient dataset. if not, create new folds based on the dataset
folds = load_or_create_folds(whole_dataset, cfg["folds_path"]) 

# Fusion strategy
late_fusion = False
early_fusion = False
fusion_strategy = cfg["fusion_strategy"]
classifier = cfg["classifier"]

if fusion_strategy == 'early':
    early_fusion=True
elif fusion_strategy == 'late':
    late_fusion=True

compute_all_metrics = False if late_fusion or early_fusion else True 
timestamp = datetime.now().strftime("%m%d_%H%M")
mask_imp_str = 'mask' if cfg["mask_missing"] else 'imp'

early_fusion_options = {
    'instance_agg':'mean',
    'fusion_head':['unimodal'],
}
late_fusion_options = {
    'instance_agg':'mean',
    'fusion_head':['mean'],
}
if late_fusion:
    options = late_fusion_options 
elif early_fusion:
    options = early_fusion_options
else: # attention
    options={
        'fusion_head':'attn',
    }

for fusion_head in options["fusion_head"]:
    exp_dir = f"experiments/{timestamp}-{fusion_head}{mod_list}"
    os.makedirs(exp_dir, exist_ok=True)

    all_results = {
        "folds": []
    }

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n---fold {fold}, train_idx={len(train_idx)}, val_idx={len(val_idx)}---")
        set_seed(42 + run * 1000 + fold)
    
        train_subset = torch.utils.data.Subset(whole_dataset, train_idx)
        print("Train:")
        print_mod_distribution(whole_dataset, train_idx)
    
        val_subset = torch.utils.data.Subset(whole_dataset, val_idx)
        print("\nVal:")
        print_mod_distribution(whole_dataset, val_idx)
    
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=params['batch_size'],
            shuffle=False,
            drop_last=False,
            collate_fn=partial(collate_fn, mask_missing_flag=cfg["mask_missing"], modalities=whole_dataset.modalities, device=device),
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=params['batch_size'],
            shuffle=False,
            drop_last=False,
            collate_fn=partial(collate_fn, mask_missing_flag=cfg["mask_missing"], modalities=whole_dataset.modalities, device=device),
        )

        # -- init model
        if late_fusion:
            model = LateFusionModule(
                    mask_missing=cfg["mask_missing"],
                    all_modalities=whole_dataset.modalities,
                    modality_names=modality_names,
                    fusion_head=fusion_head,
                    instance_agg=options["instance_agg"],
                    d_model=params['d_model']
                ).to(device)
        elif early_fusion:
            model = EarlyFusionModule(
                    mask_missing=cfg["mask_missing"],
                    all_modalities=whole_dataset.modalities,
                    modality_names=modality_names,
                    fusion_head=fusion_head,
                    instance_agg=options["instance_agg"],
                    d_model=params['d_model'],
                ).to(device)
        else: # attention
            model = MultimodalTransformer(
                        mask_missing=cfg["mask_missing"],
                        all_modalities=whole_dataset.modalities,
                        modality_names=modality_names,
                        d_model=params['d_model'],
                        num_heads=params['num_heads'],
                        num_layers=params['num_layers'],
                        dropout=params['dropout'],
                    ).to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        class_weights = get_class_weights(whole_dataset, train_idx)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights, device=device))
    
        print(f"\n --Training--")
        if classifier == "xgboost":
            if late_fusion:
                balacc, rocauc, best_th, best_score, models = train_xgboost_late(model, train_loader, val_loader,device)
        
                for mod, trained_model in models.items():
                    trained_model.save_model(f"{exp_dir}/{fold}-{mod}.json")
        
            elif early_fusion:
                balacc, rocauc, best_th, best_score, model, importances, tkn_names = train_xgboost_early(model, train_loader, val_loader,device)
        
                model.save_model(f"{exp_dir}/{fold}.json")
        
            # Store fold-level XGBoost metrics
            xgb_results["balacc"].append(balacc)
            xgb_results["rocauc"].append(rocauc)
            xgb_results["threshold"].append(best_th)
            xgb_results["best_score"].append(best_score)
            xgb_results["importance"].append(importances)
        
        
        elif classifier == "mlp":
            model, train_results = train_model_earlystop(model, train_loader, val_loader, optimizer, criterion, device, params) 
            torch.save(model.state_dict(),f"{exp_dir}/model_{fold}.pt")

            print("\n-- Testing --") 
            test_results = test_model(model, val_loader, device, decision_threshold=train_results["decision_threshold"],compute_all_metrics=compute_all_metrics)
        
            all_results["folds"].append(
                {
                    "run": run,
                    "fold": fold,
                    "train_results": train_results,
                    "test_results": test_results,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                }
            )
        
        # Aggregate XGBoost cross-validation results
        if classifier == "xgboost":
            results_dict[fusion_head] = aggregate_xgboost_results(xgb_results, fusion_head, tkn_names)

    # Save results
    if classifier == "xgboost":
        res_df = pd.DataFrame(results_dict).T
    
        fusion = "late" if late_fusion else "early"
        output_path = f"experiments/{fusion}-boost.csv"
    
        res_df.to_csv(output_path, index=False)
        print(f"Saved XGBoost results to {output_path}")
    
    else:
        torch.save(
            all_results,
            f"{exp_dir}/cv_results.pt",
        )
        print(f"Saved MLP results to {exp_dir}")
