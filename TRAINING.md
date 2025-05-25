## Training

The entire pipeline consists of 4 stages:

### 1. MViT

We first train the MViT model and use its weights to initialize the training of the Multi-term Frame Encoder (MTFE), reducing overall training time.

- **Bash files location:** `run_files/mvit`
- **Example command (GraSP dataset):**
  ```bash
  bash run_files/mvit/grasp_phases.sh

### 2. Multi-term Frame Encoder (MTFE)

We train the MTFE using the previously pretrained MViT model on the same dataset.

- **Bash files location:** `run_files/mmvit`
- **Important:**  If training from scratch, update the CHECKPOINT parameter in the bash script to point to the best model from the previous step.
- **Example command (GraSP dataset):**
  ```bash
  bash run_files/mmvit/grasp_phases.sh

### Feature Extraction 

We extract features that represent each keyframe using the trained MTFE.

- **Bash files location:** `run_files/extract_features`
- **Output directory for features:**  `./outputs/MuST_feats/`
- **Example command (GraSP dataset):**
  ```bash
  bash run_files/extract_features/grasp_phases.sh


### Long-term Transformer

We train the Long-Term Transformer module using the features extracted in the previous step.

- **Bash files location:** `run_files/long_term_transformer`
- **Important:**  Update the location of the extracted features in the bash script as needed. The recommended path is "./data/{dataset}/frames_features".
- **Example command (GraSP dataset):**
  ```bash
  bash run_files/long-term-transformer/grasp_phases.sh


