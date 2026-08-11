from collections import defaultdict
import torch

def pad_img_modality(patient_imaging, filenames, mod, device):
    """
    pads sequence to max_pad so that every sequence in batch has the same length
    """
    modality_name = mod.name
    max_pad = mod.max_files_per_pid
    modality_dim = mod.feat_dim
    
    flat_filenames = [f for sublist in filenames for f in sublist]
    if len(patient_imaging) > 0:
        flat_patient_imaging = [t for sublist in patient_imaging for t in sublist]
        flat_patient_imaging = [t.to(device) for t in flat_patient_imaging]
        feats = torch.stack(flat_patient_imaging)  # (n_scans, feat_dim)
        num_imgs = feats.size(0)
        if num_imgs > max_pad:  # truncate
            feats = feats[:max_pad]
            num_imgs = max_pad

        mask = torch.zeros(max_pad, dtype=torch.bool, device=device)
        
        # else, calculate pad_len
        pad_len = max_pad - num_imgs
        if pad_len > 0:
            pad = torch.zeros((pad_len, feats.size(1)), dtype=feats.dtype, device=device)
            feats = torch.cat([feats, pad], dim=0)
            mask[num_imgs:] = True # mask padded entries
        
        flat_filenames.extend([f"{modality_name}_PAD"]*pad_len)

    else: # if modality is missing entirely
        feats = torch.zeros((max_pad, modality_dim), dtype=torch.float32, device=device)  # no scans

        # either way, create a mask of ones (TRUE) so the model ignores missing scans
        mask = torch.ones(max_pad, dtype=torch.bool, device=device) 
        flat_filenames.extend([f"{modality_name}_PAD"]*max_pad)

    return feats, mask, flat_filenames

def collate_fn(batch, mask_missing_flag, modalities, device):        
    batch_img_feats_dict, batch_img_mask_dict, batch_img_filenames_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    
    patient_ids = []
    patient_labels = []
    token_names = []

    tab_feats_dict, tab_mask_dict = defaultdict(list), defaultdict(list)

    for patient in batch:
        img_feats_dict, img_filenames_dict = defaultdict(list), defaultdict(list)    

        visits = patient["visits"]

        if not any(len(feats) > 0 for feats in visits.values()):
            continue  # skip this patient entirely

        patient_ids.append(patient["patient_id"])
        patient_labels.append(patient["label"])

        for modality, feats in visits.items():  
            mod = modalities[modality]

            if mod.type == "tabular":
                if len(feats)==0: # entirely missing modality
                    tab_feats_dict[modality].append(torch.zeros(mod.feat_dim)) # zeros placeholder
                    if mask_missing_flag:
                        tab_mask_dict[modality].append(torch.ones(1, dtype=torch.bool)) # 1 TRUE token (Mask)
                    else:
                        tab_mask_dict[modality].append(torch.zeros(1, dtype=torch.bool)) # 1 FALSE token (Attend)
                else:
                    tab_feats_dict[modality].append(feats)
                    if mod.single_token:
                        tab_mask_dict[modality].append(torch.zeros(1, dtype=torch.bool)) # 1 FALSE token (modality exists)
                    else:
                        # if we consider each feature to be a token, check which features are missing
                        missing_feats_mask = (feats == -1) # TRUE where feature is missing (ignores that feature)
                        if mask_missing_flag:
                            tab_mask_dict[modality].append(missing_feats_mask)
                        else:
                            # attend to everything
                            tab_mask_dict[modality].append(torch.zeros_like(missing_feats_mask,dtype=torch.bool))

        
            elif mod.type == "image":
                if len(feats) == 0:
                    # create empty placeholder for entirely missing modality
                    img_feats_dict[modality]=[]
                    filenames=[]
                    if not mask_missing_flag:
                        img_feats_dict[modality]=[torch.zeros(1,mod.feat_dim, device=device)]
                        filenames=[modality]
                else:
                    tensors = torch.stack(list(feats.values()))
                    img_feats_dict[modality].append(tensors)
                    filenames = [
                        f"{modality}-{fname.split('-')[-1]}"
                        for fname in feats.keys()
                    ]

                img_filenames_dict[modality].append(filenames)
                
        # pad imaging modalities to MAX_PAD
        for img_mod in img_feats_dict.keys():
            mod = modalities[img_mod]
            feats, mask, filenames = pad_img_modality(img_feats_dict[img_mod], img_filenames_dict[img_mod], mod, device)

            batch_img_feats_dict[img_mod].append(feats)
            batch_img_mask_dict[img_mod].append(mask)
            batch_img_filenames_dict[img_mod].append(filenames)

    out_batch_img_feats, out_batch_img_mask, out_batch_tab_feats, out_batch_tab_mask = {}, {}, {}, {}
    for modality, list_of_samples in batch_img_feats_dict.items():
        out_batch_img_feats[modality] = torch.stack(list_of_samples)
        out_batch_img_mask[modality] = torch.stack(batch_img_mask_dict[modality]).bool()

    for modality, list_of_samples in tab_feats_dict.items():
        out_batch_tab_feats[modality] = torch.stack(list_of_samples)
        out_batch_tab_mask[modality] = torch.stack(tab_mask_dict[modality]).bool()
    
    return {
        "patient_ids": patient_ids,
        "patient_labels": patient_labels,
        "batch_img_filenames_dict": batch_img_filenames_dict,
        "out_batch_tab_feats":out_batch_tab_feats, 
        "out_batch_tab_mask":out_batch_tab_mask,
        "out_batch_img_feats":out_batch_img_feats,
        "out_batch_img_mask":out_batch_img_mask
    }