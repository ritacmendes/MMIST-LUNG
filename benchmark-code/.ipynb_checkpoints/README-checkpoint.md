# 👉 How it works

This code trains a _**Transformer-based architecture**_ designed to fuse _**multi-modal data**_ while handling _**missing data**_.

The model operates on a **fixed-length input sequence**. To enforce this:
- Missing tokens are **padded** up to the maximum sequence length  
- A **masking mechanism** ensures that padded tokens are ignored during attention  
  - `True` → mask / ignore token  
  - `False` → attend to token

## Key Capabilities

With this framework, you can:

- **Experiment with different modality combinations**  
- **Control how missing modalities are handled**  
  - Fully masked vs. meaningless-value-filled inputs  
- **Compare tokenization strategies**  
  - One token per modality  
  - One token per individual feature  

See the configuration section below to set up your `config.yaml` and run experiments.

---

# 👉 How to configure
## config.yaml

Defines how data is loaded and structured.

- **patient_id_col**: column linking all datasets  
- **target_label**: prediction target  
- **main_modality**: name of modality that contains all patient IDs and target label
- **single_token**:
    - `True` → Each modality corresponds to a single token
    - `False` → Each individual tabular feature corresponds to a token
- **mask_missing**: 
  - `True` → missing tokens are fully masked 
  - `False` → model attends to zeroed tokens

> 💡 The `mask_missing` and `single_token` settings can be used together.
> - If `single_token = False` and `mask_missing = True`  
>     → Missing **individual tabular features** are masked and ignored
> - If `single_token = True`  
>    → Masking applies at the **modality level**
      
### 🔸 Modalities

Each entry in `modalities` includes:

- **name**: modality name  
- **type**: `image` or `tabular`  

#### ▫️ Imaging modalities `(type: image)`

- **folder**: path to `.npz` feature files  
    - Expected filename format: `patientid-seriesnum.npz`

> ⚠️ If different, modify `extract_pid_from_filename()` in `ImageModality` to correctly extract patient ID from filename.

#### ▫️ Tabular modalities `(type: tabular)`

- **csv_path**: path to CSV file

> ⚠️ By default, missing features are filled with `-1` (`fillna(-1)`). If `-1` is a valid or meaningful value in your dataset, you should modify this in `TabularModality.__init__`
  
- **drop_cols**: additional columns to remove  
    - `patient_id_col` and `target_label` are dropped automatically, and don't need to be directly specified in `drop_cols`.  
---

## main.py

Configure training and model settings in the `params` dictionary.

Also define:
- **train_indices**
- **test_indices**

---

## Run

After configuring:

```bash
python3 main.py