# Two-Tier Visual Memory for Efficient Video Question Answering

Training-free video question answering that reduces a video to **two kinds of visual evidence** before it reaches
the VLM:

- a **low-resolution summary** of the whole video (one frame per visual cluster, 320x180, ~66 visual tokens each), and
- a few **high-resolution close-ups** selected for the question (640x360, ~220 tokens each).

No fine-tuning, no extra training data. The answering model is `Qwen3-VL-4B-Instruct`, and everything runs on a
single 16 GB GPU.

**Headline result:** the same accuracy as the standard uniform-sampling baseline while sending **28% fewer input
tokens and 29% less image data**, and the best accuracy of seven methods at a tight 8-frame budget.

---

## Results

Evaluated on 350 questions (175 MVBench + 175 Video-MME short/medium), 5 categories x 35 questions per dataset.

### Main comparison (Qwen3-VL-4B, 350 questions)

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

### What the experiments support, and what they do not

Supported:
- **The two tiers need each other.** Either alone loses 4.6-6.6 points (p = 0.04 and p = 0.001).
- **Equal accuracy at lower cost** than uniform sampling: 28% fewer tokens, 29% less data.
- **The advantage grows with video length:** +2.9 on Video-MME, -1.8 on MVBench.
- **Mixing resolutions is the active ingredient.** Q-Frame's own multi-resolution variant beats its
  single-resolution variant by 3-6 points, which points the same way.

Not supported (reported as negative results):
- **Clustering is not essential.** Uniform frames at two resolutions ("uniform two-resolution", no clustering)
  tie with the full method.
- **The iterative memory loop does not pay off.** 93% of questions stop after one round; on the rest, extra
  rounds fixed 3 answers and broke 5.
- **A larger summary does not help**, and hurts on short clips.
- **Counting stays weak** (48-54%) under every selection rule tried.

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

## Reproducing the results

Run the notebooks in order with the `adaptive_vlm` kernel. Every stage is resumable: results are appended per
question and finished questions are skipped.

| Step | Notebook | What it does | Time |
|---|---|---|---|
| 1 | `01_prepare_dataset.ipynb` | Downloads MVBench + Video-MME annotations, samples 350 test and 95 calibration questions, downloads **only those** videos (~8 GB) out of the benchmark archives, extracts frames at 5 fps (MVBench) and 1 fps (Video-MME) | 1-2 h, mostly downloads |
| 2 | `02_frame_features.ipynb` | SigLIP2 embedding, motion, sharpness, contrast per frame; question embeddings | ~20 min |
| 3 | `03_two_tier_method.ipynb` | The method and all ablations | ~1 h |
| 4 | `04_paper_baselines.ipynb` | AKS / Q-Frame / BOLT / MaxInfo comparison | ~3.5 h (45 min of it one-time scoring) |

**Which run is the headline number.** Notebook 03 has two parts. Sections 1-12 are the version with the
iterative memory loop and its ablations. The **V2 section at the bottom** contains the final configuration:
**`v2_count_uniform`** = one round, cluster count from the video-length rule, uniform close-ups for counting
questions. That run is the 65.7% in the table above.

The committed `results/` logs let you check every number without a GPU: each line records the selected frames,
the answer, the confidence, the token counts and the bytes sent.

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

---

## Limitations

- **Token budgets in notebook 04 are approximate for our method.** The budget converter assumes a flat 220
  tokens per high-resolution frame, but a frame from a small source video costs less (~180 on MVBench). At the
  8-frame budget our method therefore spent ~15% more tokens than the fixed-frame baselines (2.0k vs 1.7k),
  shrinking to 3% at 32 frames. The code here is exactly what produced the reported numbers; a corrected
  converter would measure each frame's real cost first.
- **One model family.** All results use Qwen3-VL-4B. The findings are not yet confirmed on larger models.
- **No long videos.** Video-MME's long split (30-60 min) was not run, and that is where the summary tier should
  help most.
- **Subset, not the full benchmarks.** 350 questions, so differences below about 5 points are not significant;
  per category (35 questions) that widens to about 16 points.
- **Q-Frame's lowest resolution tier cannot be reproduced exactly** on Qwen3-VL, which floors images at
  256x256 px (~64 tokens). Its 1 : 2 : 8 frame ratio is kept and scaled instead.
- **AKS at small budgets:** its `num / 2^depth` rule allocates zero frames when the budget is 8 or 16, so the
  recursion depth is capped at `log2(k)` here.

## Licence

MIT, see [LICENSE](LICENSE). The benchmarks and pretrained models carry their own terms.

## Acknowledgements

Benchmarks: [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) (CVPR 2024) and
[Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME).
Compared methods: [AKS](https://github.com/ncTimTang/AKS), [Q-Frame](https://github.com/xiaomi-research/q-frame),
[BOLT](https://github.com/sming256/BOLT), [MaxInfo](https://github.com/FusionBrainLab/MaxInfo). The two-tier
design is inspired by [Flash-VStream](https://github.com/IVGSZ/Flash-VStream) (ICCV 2025), reproduced here
without any fine-tuning.
