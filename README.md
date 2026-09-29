<h1 align="center">
  Two-Tier Visual Memory for Efficient Video Question Answering
</h1>

<h3 align="center">
  Training-free frame selection for edge-to-cloud Vision-Language Model inference
</h3>

<div align="center">
  <a href="https://scholar.google.com/citations?user=g9gcbl0AAAAJ&hl=en&oi=ao">Muhammad Fahad Nasir</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?hl=en&user=fZkn9poAAAAJ">Mobeen ur Rehman</a> &nbsp;
</div>

<h4 align="center">
  <!-- <a href=""><b>Paper</b></a> &nbsp;•&nbsp;  -->
  <a href="https://www.youtube.com/@Dr._Irfan_Robotics_Lab_KU"><b>Video</b></a>
</h4>

<div align="center">

<img height="65" alt="image" src="https://github.com/user-attachments/assets/f9af6b5d-b8f3-4ca9-9398-d1d01cea6262" />  &nbsp;&nbsp; <img height="65" alt="image" src="https://github.com/user-attachments/assets/5dd33fad-d340-4fa4-b47b-6b2c3e44819a"  />&nbsp;&nbsp; <img height="65" alt="image" src="https://github.com/user-attachments/assets/b50ab72b-f752-4941-9a6a-b0a0cfcecaa7" />


</div>

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg



## Overview

Training-free video question answering that reduces a video to **two kinds of visual evidence** before it reaches
the VLM:

- a **low-resolution summary** of the whole video (one frame per visual cluster, 320x180, ~66 visual tokens each), and
- a few **high-resolution close-ups** selected for the question (640x360, ~220 tokens each).

No fine-tuning, no extra training data. The answering model is `Qwen3-VL-4B-Instruct`, and everything runs on a
single 16 GB GPU.

**Headline result:** the same accuracy as the standard uniform-sampling baseline while sending **28% fewer input
tokens and 29% less image data**, and the best accuracy of seven methods at a tight 8-frame budget.

## Methodology
Our Research proposes a pipeline built for adaptive visual information transmission to Vision-Language Models, and evaluates it against four published training-free frame-selection methods on a 350-question subset of MVBench and Video-MME.

Key Research findings are not only how accurate a method is, but how much visual information it must transmit to reach that accuracy.

<img width="4176" height="1873" alt="Adaptive-VLM" src="https://github.com/user-attachments/assets/ff42de13-6956-4d13-bbfc-b284e45ef551" />


## Results

Evaluated on 350 questions (175 MVBench + 175 Video-MME short/medium), 5 categories x 35 questions per dataset.

### Main comparison (Qwen3-VL-4B, Our Subset Dataset)

| Method | Accuracy | MVBench | Video-MME | Input tokens | Frames seen | Data sent |
|---|---:|---:|---:|---:|---:|---:|
| **Ours (two-tier, 1 round)** | **65.7** | 65.1 | **66.3** | **4.5k** | 34.3 | 770 KB |
| Ours (up to 3 rounds, memory) | 65.1 | 64.6 | 65.7 | 5.7k | 35.4 | 797 KB |
| Uniform 32 frames @640x360 | 65.1 | **66.9** | 63.4 | 6.3k | 30.9 | 1084 KB |
| Uniform 64 frames @320x180 | 64.9 | 66.3 | 63.4 | 4.4k | 55.0 | 725 KB |
| Clusters ranked by size only | 64.9 | 64.6 | 65.1 | 5.6k | 35.4 | 798 KB |
| Uniform two-resolution (no clustering) | 64.3 | 63.4 | 65.1 | 4.9k | 39.3 | 863 KB |
| Summary tier only | 61.1 | 62.9 | 59.4 | 2.1k | 23.2 | 328 KB |
| Close-up tier only | 59.1 | 58.9 | 59.4 | 2.5k | 11.6 | 446 KB |

### Against published training-free selection methods

All ported into this harness and answered by the same model with the same prompts.

| Method | 8 frames | 16 frames | 32 frames |
|---|---:|---:|---:|
| **Ours (two-tier)** | **62.3** | 63.4 | 65.1 |
| AKS (CVPR 2025) | 61.7 | 64.3 | 65.1 |
| MaxInfo (WACV 2026) | 61.4 | 60.3 | 63.7 |
| Q-Frame multi-resolution (ICCV 2025) | 61.1 | **64.9** | 64.6 |
| Uniform sampling | 58.3 | 64.3 | 65.1 |
| Q-Frame fixed frames (ICCV 2025) | 57.7 | 58.6 | 63.4 |
| BOLT (CVPR 2025) | 56.6 | 62.0 | **66.6** |

Selection matters most when the budget is tight: the spread is 5.7 points at 8 frames and 3.2 points at 32.

---

## Repository layout

```text
notebooks/
  01_prepare_dataset.ipynb    download annotations, sample the subset, download those videos, extract frames
  02_frame_features.ipynb     SigLIP2 embeddings + motion / sharpness / contrast + question embeddings
  03_two_tier_method.ipynb    THE METHOD: two-tier selection, ablations, memory variant, embedding plots
  04_paper_baselines.ipynb    AKS, Q-Frame, BOLT, MaxInfo, uniform, ours at 8 / 16 / 32 budgets
asvlm_core.py                 shared harness used by notebooks 03 and 04
data/subset/manifest*.jsonl   the exact 350 test + 95 calibration questions used in the paper
results/two_tier/*.jsonl      per-question logs of the method, its ablations and our baselines
results/papers/*.jsonl        per-question logs of the published-method comparison
environment.yml               conda environment (Python 3.12, torch 2.11+cu128, transformers 5.17)
```

Frames, videos and features are **not** committed (they are large and the benchmarks have their own terms).
Notebooks 01 and 02 regenerate them.

---

## Setup

```bash
conda env create -f environment.yml
conda activate adaptive_vlm
python -m ipykernel install --user --name adaptive_vlm --display-name "Python (adaptive_vlm)"
```

Hardware used: one RTX 4080 SUPER (16 GB), Windows 11. A Hugging Face token is optional and only speeds up
downloads (`HF_TOKEN` environment variable).

Models pulled automatically on first run: `Qwen/Qwen3-VL-4B-Instruct` (~9 GB),
`google/siglip2-base-patch16-224`, and for notebook 04 `Salesforce/blip-itm-base-coco` and
`openai/clip-vit-large-patch14`.

---

## Method in short

**Offline, per video (no question needed)**
1. Extract frames: 5 fps for MVBench clips (5-60 s), 1 fps for Video-MME (up to 15 min).
2. Embed every frame with SigLIP2 (768-d, L2-normalised); also compute motion, sharpness and contrast.
3. Cluster the embeddings with spherical K-means, where
   `K = clip(round(8 * log2(1 + seconds/10)), 8, 48)`, so longer videos get fewer frames per minute
   (8 clusters at 10 s, 22 at 1 min, 40 at 5 min, 48 at 15 min). A cluster's representative is the sharpest of
   the three frames nearest its centre.

**Per question**
4. The VLM reads the question text only and predicts one of five task types (78.6% accurate).
5. **Tier 1 (summary):** one low-resolution frame per cluster, in time order, each labelled with its timestamp
   and how long that scene is on screen.
6. **Tier 2 (close-ups):** 8 frames for clips under a minute, 16 otherwise, at full box resolution.
   - object / spatial / event questions: top clusters by `size x question relevance`, best frame in each
   - counting / temporal questions: spread over time, because clustering merges repeated events
7. The VLM answers; the option letter is read from next-token probabilities, which also gives a confidence.
8. Optional (off in the final configuration): if confidence is below a calibrated threshold, the model writes a
   short memory note ("what I saw, what is still unclear, LOOK FOR ...") and 8 new close-ups are added, up to
   3 rounds. The threshold is tuned on the 95 calibration questions only.


## Acknowledgements

Benchmarks: [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) (CVPR 2024) and
[Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME).
Compared methods: [AKS](https://github.com/ncTimTang/AKS), [Q-Frame](https://github.com/xiaomi-research/q-frame),
[BOLT](https://github.com/sming256/BOLT), [MaxInfo](https://github.com/FusionBrainLab/MaxInfo). The two-tier
design is inspired by [Flash-VStream](https://github.com/IVGSZ/Flash-VStream) (ICCV 2025), reproduced here
without any fine-tuning.
