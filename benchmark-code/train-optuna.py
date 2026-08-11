from PatientDataset import PatientDataset
from collate import collate_fn
import numpy as np
from torch.utils.data import Subset
import torch
from torch import nn, optim
from MultimodalTransformer import MultimodalTransformer
import yaml
from utils import get_class_weights, test_model, train_model_earlystop, print_mod_distribution
from functools import partial
from datetime import datetime
import os 
import json
import warnings
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import optuna

warnings.filterwarnings("ignore")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
# Load config file
with open('config.yaml', "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# the folds should have all patients in patient dataset. if not, create new folds based on the dataset
folds = load_or_create_folds(whole_dataset, cfg["folds_path"]) 

# Fusion strategy
late_fusion = False
early_fusion = False
fusion_strategy = cfg["fusion_strategy"]

if fusion_strategy == 'early':
    early_fusion=True
elif fusion_strategy == 'late':
    late_fusion=True


early_fusion_options = {
    'instance_agg':'mean', 
    'fusion_head':['unimodal'], # mean / max ?
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
    options = {
        'fusion_head':'attn',
    }

compute_all_metrics = False if late_fusion or early_fusion else True 

timestamp = datetime.now().strftime("%m%d_%H%M")
mask_imp_str = 'mask' if cfg["mask_missing"] else 'imp'
exp_dir = f"experiments/{timestamp}-{mask_imp_str}{mod_list}"
os.makedirs(exp_dir, exist_ok=True)


# -- Set Train parameters and train/test indices
def objective(trial):
    params = {
        'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'num_epochs': 100,
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'min_train_epochs': 5,
        'patience': 20
    }
    
    fold_baccs = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n---fold {fold}, train_idx={len(train_idx)}, val_idx={len(val_idx)}---")
        set_seed(42 + fold)
    
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
        model, train_results = train_model_earlystop(model, train_loader, val_loader, optimizer, criterion, device, params) 
        # save model 
        torch.save(model.state_dict(), f"{exp_dir}/model_{fold}.pt")
        
        print(f"\n --Testing--")
        test_results, _, _ = test_model(model, val_loader, device, decision_threshold=train_results["decision_threshold"], compute_all_metrics=False)
        
        fold_baccs.append(test_results["balacc"])

        trial.report(np.mean(fold_baccs), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_baccs)


study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

study.optimize(objective, n_trials=50)

print("Best trial:")
print(study.best_trial.value)
print(study.best_trial.params)

with open("optuna_best_trial.txt", "w") as f:
    f.write("Best trial:\n")
    f.write(f"Best value (bacc): {study.best_trial.value}\n")
    f.write(f"Best params:\n")
    for k, v in study.best_trial.params.items():
        f.write(f"  {k}: {v}\n")
