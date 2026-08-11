from torch.utils.data import Dataset
import yaml
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import re
import h5py

class BaseModality:
    def build(self):
        raise NotImplementedError

    def get(self, pid):
        raise NotImplementedError
# -------------------------
# Create tabular visit, dropping target columns, if any present
# -------------------------
class TabularModality(BaseModality):
    def __init__(self, name, csv_path, group_key, drop_cols=None, mask=False, single_token=True):
        self.df = pd.read_csv(csv_path).fillna(-1) # make sure this is a meaningless value in your data
        self.type = 'tabular'
        self.name = name
        self.group_key = group_key
        self.drop_cols = drop_cols or []
        self.mask = mask
        self.single_token=single_token
        
        valid_cols = [c for c in self.df.columns if c not in self.drop_cols]
    
        self.feat_dim = len(valid_cols) # num of features in modality
        self.token_names = valid_cols
        
        self.df = self.df.sort_values([self.group_key])
        self.mod_patient_ids = self.df[self.group_key].unique()

        self.invalid_inference_col = False


    def get(self, pid, inference_cols=None):
        times = torch.zeros(1, dtype=torch.float32) # default time = 0
        rows = self.df[self.df[self.group_key] == pid]
        if len(rows) == 0:
            return torch.tensor([])
    
        # ---- DROP META COLS ----
        rows = rows.drop(columns=self.drop_cols, errors="ignore")
    
        # ---- INFERENCE OVERRIDE ----
        if inference_cols is not None:
            for col, value in inference_cols.items():
                if col in rows.columns:
                    rows[col] = value
                    self.invalid_inference_col = False
                else:
                    self.invalid_inference_col = True
    
        arr = rows.to_numpy(dtype=np.float32)
        values = torch.from_numpy(arr).squeeze(0) # cuidado com este squeeze, isto para o temporal ja nao se pode fazer, porque aí é [T,D] porque um paciente pode ter varios eventos da mesma mod
    
        return values

class ImageModality(BaseModality):
    def __init__(self, name, folder, mask=False, single_token=True):
        self.name = name
        self.type = 'image'
        self.folder = Path(folder)
        self.filetype = None
        self.patient_map = defaultdict(list)
        self.mask = mask
        self.feat_dim = None
        self.max_files_per_pid = 0 # aka max_pad, img modality will be padded to max_pad in collate_fn for easier batching
        self.single_token=single_token

    def build(self):
        for f in self.folder.rglob(f"*.*"):
            if self.filetype is None:
                self.filetype = f.suffix.lstrip(".") # first file defines the expected type
            pid, code = self.extract_pid_from_filename(f.name)
            self.patient_map[pid].append((f, code))

            # find maximum number of series for a patient in the entire dataset
            current_len = len(self.patient_map[pid])
            if current_len > self.max_files_per_pid:
                self.max_files_per_pid = current_len
            
            # load first feat to extract feat_dim 
            if self.feat_dim is None:
                feat_tensor = self.load_feat(f, self.filetype)
                self.feat_dim = feat_tensor.squeeze().shape[0]

        self.mod_patient_ids = list(self.patient_map.keys())
    
    def get(self, pid, inference_cols):
        if inference_cols is not None:
            for feat, val in inference_cols.items():
                if feat == self.name and val == -1:
                    return {}
        feats = {}

        for path, code in self.patient_map.get(pid, []):
            feat_tensor = self.load_feat(path, self.filetype)
            feats[f"{pid}-{code}"] = feat_tensor.squeeze() # [DIM]
        return feats

    def load_feat(self, path, filetype):
        if filetype == 'npz':
            arr = np.load(path)["arr_0"] 
            feat_tensor = torch.from_numpy(arr)
        elif filetype == 'pt':
            feat_tensor = torch.load(f"{path}")
            if isinstance(feat_tensor, dict):
                feat_tensor = feat_tensor['features']
                if feat_tensor.ndim == 5 and feat_tensor.shape[-3:] == (1, 1, 1):
                    feat_tensor = feat_tensor.flatten(1)
        elif filetype == 'h5':
            with h5py.File(path, 'r') as f:
                feat_tensor = torch.tensor(np.array(f['features']), dtype=torch.float)
        else:
            raise ValueError(f"Invalid file extension for modality: {filetype}")

        return feat_tensor
    
    def extract_pid_from_filename(self, fname):
        name = Path(fname).stem
    
        pid = None
        code = None
    
        if name.startswith("TCGA"):
            # Case 1: bracketed metadata format
            if "[" in name and "]" in name:
                # TCGA-17-Z050_['...']_['...']
                parts = name.split("_")
                pid = parts[0]  # TCGA-17-Z050
    
                # optional: extract something meaningful from last bracket block
                if len(parts) > 2:
                    match = re.search(r"\['(.*?)'\]", parts[-1])
                    code = match.group(1) if match else parts[-1]
    
            else:
                # Case 2: standard dash-separated TCGA UUID format
                parts = name.split("-")
    
                # robust PID extraction: first 3 chunks usually define patient ID
                pid = "-".join(parts[:3]) if len(parts) >= 3 else name
    
                # last chunk is often sample/site code, but only if it exists
                code = parts[-1] if len(parts) > 1 else None
        
        else: # CPTAC
            if len(name.split("-")) == 3: # WSI
                parts = name.split("-")
                pid = "-".join(parts[:2])
                code = parts[-1]
            else: # CT / PET
                parts = name.split("_")
                pid = "-".join(parts[:1])
                code = parts[-1]         
        
        return pid, code

class PatientDataset(Dataset):
    def __init__(self, cfg, inference_cols=None):
        self.modalities = {}
        self.patient_samples = {}
        self.labels = {}
        
        patient_id_col = cfg['patient_id_col']
        target_label = cfg['target_label']
        main_mod_name = cfg['main_modality']

        # -------------------------
        # build modalities dynamically
        # -------------------------
        modalities = cfg['modalities']
        for mcfg in modalities:
            name = mcfg["name"]
            mask =  mcfg.get("mask", False)
            single_token =  mcfg.get("single_token", True) # default = TRUE (1 modality = 1 token)
            if mcfg["type"] == "tabular":
                raw_drop = mcfg.get("drop_cols", [])
                drop = [raw_drop] if isinstance(raw_drop, str) else list(raw_drop)
                drop += [patient_id_col, target_label]
                
                mod = TabularModality(
                    name=name,
                    csv_path=mcfg["csv_path"],
                    group_key=patient_id_col,
                    drop_cols=drop,
                    mask=mask,
                    single_token=single_token,
                )

            elif mcfg["type"] == "image":
                mod = ImageModality(
                    name=name,
                    folder=mcfg["folder"],
                    filetype=mcfg.get("filetype", "npz"),
                    mask=mask,
                    single_token=single_token,
                )
                mod.build()

            else:
                raise ValueError(f"Unknown modality type: {mcfg['type']}")

            self.modalities[name] = mod

        # -- build patient_samples
        main_mod = self.modalities[main_mod_name]
        
        self.patient_ids = main_mod.df[patient_id_col].unique().tolist()

        if main_mod.mask: # if main modality is masked
            all_patient_ids = set()
        
            for mod_name, mod in self.modalities.items():
                if mod_name != main_mod_name:
                    all_patient_ids.update(mod.mod_patient_ids)
            # intersect main mod ids with union of all available modalities
            self.patient_ids = list(set(self.patient_ids) & all_patient_ids)

            print(f"Filtered to {len(self.patient_ids)} patients due to main modality '{main_mod_name}' masked.")
        
        for pid in self.patient_ids:  
            visits = {}
            mask = main_mod.df[patient_id_col] == pid # filter pid rows
            label = main_mod.df.loc[mask, target_label].iloc[0]

            for mod_name, mod in self.modalities.items():
                if not mod.mask:
                    visits[mod_name] = mod.get(pid, inference_cols)
    
            
            self.patient_samples[pid] = {
                "patient_id": pid,
                "visits": visits,
                "label": label,
            }
            
        for mod_name, mod in self.modalities.items():
            if mod.type == 'tabular' and inference_cols is not None:
                if mod.invalid_inference_col:
                    print(f"Inference_col not in {mod_name}")
                else: 
                    print(f"Found Inference_col in {mod_name}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        return self.patient_samples[pid]

    def modality_distribution(self, indices=None):
        if indices is None:
            indices = range(len(self.patient_samples))
    
        stats = {m: 0 for m in self.modalities.keys()}
    
        for i in indices:
            pid = self.patient_ids[i]
            sample = self.patient_samples[pid]
            visits = sample["visits"]
    
            for mod_name, data in visits.items():
                if data is not None and len(data) > 0:
                    stats[mod_name] += 1
    
        total = len(indices)
    
        return {
            m: {
                "count": stats[m],
                "present_pct": 100 * stats[m] / total,
                "missing_pct": 100 * (1 - stats[m] / total),
            }
            for m in stats
        }