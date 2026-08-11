import torch
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)
import numpy as np
import copy
import torch.nn.functional as F

def load_or_create_folds(dataset, folds_path, n_splits=5):
    if not os.path.exists(folds_path):
        labels = []
    
        for i in range(len(dataset)):
            sample = dataset[i]
            labels.append(sample['label'])
    
        labels = np.array(labels)
    
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
        folds = []
    
        for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
            folds.append((train_idx, val_idx))
    
        os.makedirs(os.path.dirname(folds_path), exist_ok=True)
    
        np.save(folds_path,np.array(folds, dtype=object),allow_pickle=True)
    
    return np.load(folds_path, allow_pickle=True)
    
def print_mod_distribution(dataset, indx=None):
    dist = dataset.modality_distribution(indx)

    print(f"Total patients: {len(indx) if indx is not None else len(dataset.patient_samples)}\n")

    for mod, d in dist.items():
        print(
            f"{mod:<15} "
            f"{d['count']:>5} | "
            f"{d['present_pct']:6.1f}% present | "
            f"{d['missing_pct']:6.1f}% missing"
        )

def get_class_weights(whole_dataset, idx):
    labels = []

    for i in idx:
        labels.append(int(whole_dataset[i]["label"]))

    labels = torch.tensor(labels)

    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()

    # pos_weight for BCEWithLogitsLoss
    pos_weight = num_neg / num_pos

    return pos_weight
    
def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    else:
        return x

def train_model_earlystop(model, train_loader, val_loader, optimizer, criterion, device, params):
    pbar = tqdm(range(params['num_epochs']))
    avg_train_loss_curve, train_bacc_curve = [], []
    epochs_no_improve, best_val_balacc = 0, 0
    
    for epoch in pbar:
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        for batch in train_loader:
            batch = move_to_device(batch, device)

            out = model(batch)
            outputs = out["outputs"]
            
            batch_labels = (
                torch.tensor(batch["patient_labels"], dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )

            optimizer.zero_grad()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())

        train_bacc = balanced_accuracy_score(train_targets, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_curve.append(avg_train_loss)
        train_bacc_curve.append(train_bacc)

        # Early Stopping
        results, all_probs_np, all_labels_np = test_model(model, val_loader, device, compute_all_metrics=False)
        val_balacc = results["balacc"]
        if val_balacc > best_val_balacc and epoch > params["min_train_epochs"]: # train minimum epochs
            best_val_balacc = val_balacc
            epochs_no_improve = 0
            best_model = copy.deepcopy(model)
            best_probs = all_probs_np
            best_labels = all_labels_np
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == params["patience"]:
            print("Early stopping at epoch",epoch)
            break 
        
        pbar.set_description(
            f"Epoch {epoch + 1}-"
            f"Avg Train Loss = {avg_train_loss:.3f}, Train Bal Acc = {train_bacc*100:.2f}%, Val Bal Acc = {val_balacc*100:.2f}%"
        )

    # final results       
    fpr, tpr, thresholds = roc_curve(best_labels, best_probs)
    best_idx = np.argmax(tpr - fpr)  # Youden index
    best_threshold = thresholds[best_idx]
    
    results = {
        "params":params,
        "train_len": len(train_targets),
        "final_train_bacc": train_bacc,
        "avg_train_loss_curve": avg_train_loss_curve,
        "train_bacc_curve": train_bacc_curve,
        "decision_threshold":best_threshold
    }
    
    return best_model, results

def test_model(model, test_loader, device, decision_threshold=0.5, compute_all_metrics=False):
    all_preds, all_labels, all_probs = [], [], []
    patient_sequences = {}
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):       
            batch = move_to_device(batch, device)
            out = model(batch)
            outputs = out["outputs"]
            batch_labels = (
                torch.tensor(batch["patient_labels"], dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )
            
            probs = torch.sigmoid(outputs)
            predicted = (probs > decision_threshold).long().squeeze(1)

            all_preds.append(predicted.detach().cpu())
            all_labels.append(batch_labels.detach().cpu())
            all_probs.append(probs.detach().cpu())

            if compute_all_metrics:
                attn_weights_all=out["attn_weights_all"]
                token_names=out["final_token_names_batch"]
                rollout=out["rollout"]
                attn_all = torch.stack(attn_weights_all) 

                for i, pid in enumerate(batch['patient_ids']):
                    patient_attn = attn_all[:, i]
                    cls_attn = patient_attn[:, :, 0, :]
                    patient_sequences[pid] = {
                        'cls_attn': cls_attn.cpu().numpy(),
                        'cls_rollout': rollout[i, 0, :].cpu().numpy(),
                        'cls_attn_mean_heads':cls_attn.mean(dim=1).cpu().numpy(),
                        'cls_attn_mean_layers':cls_attn.mean(dim=0).cpu().numpy(),
                        'cls_attn_global':cls_attn.mean(dim=(0,1)).cpu().numpy(),
                        'token_names': token_names[i],
                        'pred': np.array(predicted[i].item()),  
                        'prob': np.array(probs[i].item()),
                        'label': np.array(batch_labels[i].item())  
                    }

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    all_preds_np = all_preds.numpy()
    all_labels_np = all_labels.numpy()
    all_probs_np = all_probs.numpy()

    balacc = balanced_accuracy_score(all_labels_np, all_preds_np)
    conf_matrix = confusion_matrix(all_labels_np, all_preds_np)
    roc_auc = roc_auc_score(all_labels_np, all_probs_np)
    f1 = f1_score(all_labels_np, all_preds_np)

    results = {
        'balacc': np.array(balacc),
        'decision_threshold': decision_threshold,
        'roc_auc': np.array(roc_auc),
        'conf_matrix': np.array(conf_matrix),
        'patient_sequences': patient_sequences,
        'f1_score': np.array(f1),
    }

    return results, all_probs_np, all_labels_np

def compute_attention_rollout(attn_weights_all, attention_mask, discard_ratio=0.0, head_fusion="mean"):
    """
    attn_weights_all:
        list of tensors [(B,H,T,T), ...]

    Returns:
        rollout: (B,T,T)
    """

    device = attn_weights_all[0].device
    B, H, T, _ = attn_weights_all[0].shape

    result = torch.eye(T, device=device).unsqueeze(0).repeat(B, 1, 1)

    for attn in attn_weights_all:
        # ---- fuse heads ----
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=1)

        elif head_fusion == "max":
            attn_fused = attn.max(dim=1)[0]

        elif head_fusion == "min":
            attn_fused = attn.min(dim=1)[0]

        else:
            raise ValueError(head_fusion)

        # ---- optionally remove tiny attentions ----
        if discard_ratio > 0:
            flat = attn_fused.view(B, -1)
            num_discard = int(flat.shape[-1] * discard_ratio)

            if num_discard > 0:
                threshold, _ = flat.topk(num_discard, dim=-1, largest=False)
                threshold = threshold[:, -1]
                mask = attn_fused <= threshold.view(B, 1, 1)
                attn_fused = attn_fused.masked_fill(mask, 0)

        identity = torch.eye(T, device=device)
        attn_fused = attn_fused + identity.unsqueeze(0)
        attn_fused = attn_fused.masked_fill(attention_mask.unsqueeze(1),0)

        # ---- normalize rows ----
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        # ---- recursive rollout ----
        result = torch.bmm(attn_fused, result)

    return result

def aggregate_xgboost_results(fold_results, fusion_head, tkn_names):
    balaccs = fold_results["balacc"]
    rocaucs = fold_results["rocauc"]
    best_ths = fold_results["threshold"]
    best_scores = fold_results["best_score"]
    importance_folds = fold_results["importance"]
    
    avg_balacc = np.mean(balaccs)
    std_balacc = np.std(balaccs)
    avg_rocauc = np.mean(rocaucs)
    std_rocauc = np.std(rocaucs)
    avg_ths = np.mean(best_ths)
    std_ths = np.std(best_ths)
    best_balaccs = np.mean(best_scores)
    std_best_balaccs = np.std(best_scores)
    importance_folds = np.stack(importance_folds)   # (K, D)
    mean_importance = importance_folds.mean(axis=0)  # (D,)
    std_importance = importance_folds.std(axis=0)  # (D,)
    
    return {
        'fusion_head': fusion_head,
        'balacc': f"{avg_balacc*100:.2f} ± {std_balacc*100:.2f}",
        'rocauc': f"{avg_rocauc*100:.2f} ± {std_rocauc*100:.2f}",
        'avg_balacc':avg_balacc,
        'std_balacc':std_balacc,
        'avg_rocauc':avg_rocauc,
        'std_rocauc':std_rocauc,
        'avg_ths':avg_ths,
        'std_ths':std_ths,
        'best_bacc':best_balaccs,
        'std_best_bacc':std_best_balaccs,
        'mean_importance':mean_importance,
        'std_importance':std_importance,
        'tkn_names':tkn_names,
    }