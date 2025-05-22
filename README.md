
# Project Title  
*StableMVS: Diffusion-Propelled Stable Multi-View Stereo via Epistemic and Morphological Priors for Urban Depth Ambiguity*  
**[Anonymous Project Page](https://anonymous.4open.science/r/StableMVS)**  

---

## Abstract  
This repository contains the official implementation of **StableMVS**, a robust multi-view stereo framework tailored for urban environments, introducing epistemic priors, morphology-guided propagation, and diffusion-based depth refinement to address low-texture, occlusion, and geometric ambiguity.

---

## Highlights
- **Epistemic Priors**: Integrates knowledge from large vision models into cost volume construction.
- **Morphological Optimization**: Uses structural contours to improve depth smoothness and edge preservation.
- **Diffusion-Driven Learning**: Enhances illumination-invariant feature learning via perturbation recovery.

---

## Overview

```text
Inputs: Multi-view aerial images
      ↓
Epistemic Prior Infusion
      ↓
Cost Volume Construction
      ↓
Morphological Contour Propagation
      ↓
Diffusion-Propelled Refinement
      ↓
Output: Robust & sharp depth map
```

---

## Installation

```bash
git clone https://anonymous.4open.science/r/StableMVS.git
cd StableMVS
conda create -n stablemvs python=3.10
conda activate stablemvs
pip install -r requirements.txt
```

---

## Dataset Preparation

Supported datasets:
- [ ] ETH3D
- [ ] Tanks and Temples
- [ ] UrbanStreet or UrbanScene3D

Instructions:
```bash
bash scripts/download_eth3d.sh
python scripts/preprocess_eth3d.py --input path/to/raw --output path/to/processed
```

---

## Inference

```bash
python infer.py \
  --config configs/eval_eth3d.yaml \
  --data_path path/to/data \
  --output_path ./outputs/
```

---

## Training

```bash
python train.py \
  --config configs/train_urban.yaml \
  --log_dir ./logs/
```

---

## Evaluation

```bash
python eval.py \
  --config configs/eval_eth3d.yaml \
  --load_ckpt checkpoints/stablemvs.pth
```

---



## Pretrained Models



## Results



More quantitative and qualitative results are available in the [paper](#).

---

## Citation

**Note:** This work is currently under double-blind peer review. Citation will be updated upon publication.

---

## Contact

For questions during the review process, please raise an issue in this repository or reach out through the anonymous communication channel provided in the supplementary.

---

