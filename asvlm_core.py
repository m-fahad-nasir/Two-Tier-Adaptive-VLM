"""Shared harness for the Adaptive_Semantic_VLM experiments.

Shared by notebooks 03 and 04: data loading, clustering, prompts, the Qwen wrapper and the resumable
runner. This is the code that produced the reported results.

Typical use:

    import asvlm_core as core
    core.init()                                   # loads samples + question embeddings
    B = core.QwenAgent(core.QWEN_B)               # the answering VLM
    rec = core.answer(B, sample, frames)          # frames = [(index, box), ...]
"""

from __future__ import annotations

import functools
import gc
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --------------------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent          # the repository root (this file sits next to data/)
DATA_ROOT = PROJECT_ROOT / "data"
SUBSET_DIR = DATA_ROOT / "subset"
FRAMES_ROOT = DATA_ROOT / "frames"
FEATURES_ROOT = DATA_ROOT / "features"

# ---------------------------------------------------------------------------------- experiment
FPS = {"mvbench": 5.0, "videomme": 1.0}
DATASETS = ["mvbench", "videomme"]
LETTERS = "ABCDEFGH"
SEED = 42

K_SCALE, K_MIN, K_MAX = 8.0, 8, 48                 # cluster-count rule of the proposed method
SUMMARY_BOX = (320, 180)                           # ~66 visual tokens on Qwen3-VL
CLOSEUP_BOX = (640, 360)                           # ~220 visual tokens
LOW_BOX = (160, 90)                                # ~64 tokens (Qwen3-VL floors images at 256x256 px)
TOKENS_PER_FRAME = {CLOSEUP_BOX: 220, SUMMARY_BOX: 66, LOW_BOX: 64}   # measured, for budget planning

QWEN_B = "Qwen/Qwen3-VL-4B-Instruct"
SIGLIP_ID = "google/siglip2-base-patch16-224"

MVBENCH_SYSTEM = ("Carefully watch the video and pay attention to the cause and sequence of events, the detail and "
                  "movement of objects, and the action and pose of persons. Based on your observations, select the "
                  "best option that accurately addresses the question.")
ANSWER_PREFIX = "Best option: ("

ROUTER_PROMPT = """Classify the question about a video into exactly one category.
(A) temporal reasoning - order or sequence of events, what happened before/after/next
(B) object recognition - what object, person or attribute is present, whether something exists, what someone interacts with
(C) counting - how many things or how many times something happens
(D) spatial reasoning - positions, directions, locations, where things are or move relative to each other
(E) event understanding - why or how something happened, purpose or cause of actions, unexpected events, changes, what would happen if
Question: {question}
Answer with the letter only."""
ROUTER_LABELS = ["temporal_reasoning", "object_recognition", "counting", "spatial_reasoning", "event_understanding"]

_Q_EMB: dict[str, np.ndarray] = {}


# ------------------------------------------------------------------------------------ data
def fps_dirname(fps):
    return "fps_" + f"{fps:g}".replace(".", "p")


def load_samples(split="test", datasets=None, limit=None):
    """One dict per question, with its frames, per-frame features and question embedding."""
    if not _Q_EMB:
        with np.load(FEATURES_ROOT / "questions.npz") as z:
            _Q_EMB.update({q: a.astype(np.float32) for q, a in zip(z["qid"], z["q_emb"])})
    datasets = datasets or DATASETS
    name = {"test": "manifest.jsonl", "calib": "manifest_calib.jsonl"}[split]
    samples, missing = [], 0
    for line in (SUBSET_DIR / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["dataset"] not in datasets:
            continue
        fdir = FRAMES_ROOT / fps_dirname(FPS[r["dataset"]]) / r["dataset"] / r["frame_key"]
        fpath = FEATURES_ROOT / fps_dirname(FPS[r["dataset"]]) / r["dataset"] / f"{r['frame_key']}.npz"
        if not (fdir / "frames.json").exists() or not fpath.exists() or r["qid"] not in _Q_EMB:
            missing += 1
            continue
        meta = json.loads((fdir / "frames.json").read_text())
        with np.load(fpath) as z:
            feat = {k: z[k] for k in ("t", "emb", "motion", "sharp", "contrast")}
        feat["emb"] = feat["emb"].astype(np.float32)
        r.update(split=split, fps=FPS[r["dataset"]], frame_dir=str(fdir), feat=feat,
                 frames=[dict(f, path=str(fdir / f["file"])) for f in meta["frames"]],
                 span_s=float(feat["t"][-1] - feat["t"][0]) + 1.0 / FPS[r["dataset"]], q_emb=_Q_EMB[r["qid"]])
        samples.append(r)
    if limit:
        by = {}
        for s in samples:
            by.setdefault((s["dataset"], s["category"]), []).append(s)
        samples = [s for g in by.values() for s in g[:limit]]
    if missing:
        print(f"WARNING {split}: {missing} questions skipped (run notebooks 01 and 03 first)")
    return samples


def fit_box(im, box):
    s = min(1.0, box[0] / im.width, box[1] / im.height)
    return im if s == 1.0 else im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)


@functools.lru_cache(maxsize=2048)
def load_frame(path, box):
    return fit_box(Image.open(path).convert("RGB"), box)


@functools.lru_cache(maxsize=16384)
def frame_bytes(path, box, quality=90):
    buf = io.BytesIO()
    load_frame(path, box).save(buf, format="JPEG", quality=quality)
    return len(buf.getvalue())


def norm01(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def signals(s):
    """Per-frame relevance / quality / motion, normalised inside the video."""
    if "_sig" not in s:
        f = s["feat"]
        s["_sig"] = dict(rel=norm01(f["emb"] @ s["q_emb"]), q=norm01(np.log1p(f["sharp"])) * (f["contrast"] > 8.0),
                         motion=norm01(f["motion"]), emb=f["emb"], n=len(f["t"]))
    return s["_sig"]


# ------------------------------------------------------------------------------ clustering
def kmeans(x, k, seed, iters=60):
    rng = np.random.default_rng(seed)
    n = len(x)
    cent = [x[rng.integers(n)]]
    dist = 1.0 - x @ cent[0]
    for _ in range(1, k):
        p = np.maximum(dist, 0) ** 2
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1.0 / n)
        cent.append(x[rng.choice(n, p=p)])
        dist = np.minimum(dist, 1.0 - x @ cent[-1])
    cent = np.stack(cent)
    for _ in range(iters):
        lab = np.argmax(x @ cent.T, axis=1)
        new = np.stack([x[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-8
        if np.allclose(new, cent, atol=1e-6):
            break
        cent = new
    return np.argmax(x @ cent.T, axis=1), cent


def n_clusters(s, k_scale=K_SCALE, k_max=K_MAX):
    k = round(k_scale * math.log2(1 + s["span_s"] / 10.0))
    return int(max(1, min(np.clip(k, K_MIN, k_max), len(s["frames"]) // 2)))


def clusters(s, k_scale=K_SCALE, k_max=K_MAX):
    k = n_clusters(s, k_scale, k_max)
    key = f"_cl{k}"
    if key not in s:
        sig = signals(s)
        seed = int(hashlib.md5(s["frame_dir"].encode()).hexdigest()[:8], 16) ^ SEED
        lab, cent = kmeans(sig["emb"], k, seed)
        members, reps = [], []
        for j in range(k):
            m = np.flatnonzero(lab == j)
            if not len(m):
                continue
            near = m[np.argsort(-(sig["emb"][m] @ cent[j]))[:3]]
            reps.append(int(near[np.argmax(sig["q"][near])]))
            members.append(m)
        order = np.argsort([s["frames"][r]["t"] for r in reps])
        members, reps = [members[i] for i in order], [reps[i] for i in order]
        label = np.full(sig["n"], -1)
        for j, m in enumerate(members):
            label[m] = j
        s[key] = dict(members=members, reps=reps, label=label, size=np.array([len(m) for m in members]),
                      seconds=np.array([len(m) / s["fps"] for m in members]))
    return s[key]


def uniform_idx(n_total, k):
    k = max(1, min(k, n_total))
    return sorted({int(i) for i in ((np.arange(k) + 0.5) * n_total / k).astype(int)})


# --------------------------------------------------------------------------------- prompts
def mc_system(s):
    return MVBENCH_SYSTEM if s["dataset"] == "mvbench" else None


def mc_text(s):
    if s["dataset"] == "videomme":
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(s["options"]))
        return ("Select the best answer to the following multiple-choice question based on the video. "
                f"Respond with only the letter ({', '.join(LETTERS[:len(s['options'])])}) of the correct option.\n"
                f"{s['question']}\n{opts}\nThe best answer is:")
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(s["options"]))
    return f"Question: {s['question']}\nOptions:\n{opts}\nOnly give the best option."


def frame_parts(s, frames, header=None):
    """frames: list of (frame_index, box). Frames are sent in chronological order."""
    frames = sorted(frames, key=lambda fb: fb[0])
    parts = [{"type": "text", "text": header or
              f"The following {len(frames)} frames are sampled from the video in chronological order."}]
    for i, box in frames:
        parts += [{"type": "text", "text": f"Frame at {s['frames'][i]['t']:.1f}s:"},
                  {"type": "image", "path": s["frames"][i]["path"], "box": box}]
    return parts


def two_tier_parts(s, summary, closeups):
    """The proposed method's prompt: low-res scene summary, then high-res close-ups."""
    cl = clusters(s)
    rep_sec = {r: cl["seconds"][j] for j, r in enumerate(cl["reps"])}
    parts = [{"type": "text", "text": f"Overview of the whole video: {len(summary)} low-resolution frames, "
                                      "one per distinct scene, in chronological order."}]
    for i in sorted(summary):
        parts += [{"type": "text", "text": f"Scene at {s['frames'][i]['t']:.1f}s (visible ~{rep_sec.get(i, 0):.0f}s in total):"},
                  {"type": "image", "path": s["frames"][i]["path"], "box": SUMMARY_BOX}]
    if closeups:
        parts.append({"type": "text", "text": "Close-up frames at higher resolution, in chronological order:"})
        for i in sorted(closeups):
            parts += [{"type": "text", "text": f"Frame at {s['frames'][i]['t']:.1f}s:"},
                      {"type": "image", "path": s["frames"][i]["path"], "box": CLOSEUP_BOX}]
    return parts


def parts_bytes(parts):
    return sum(frame_bytes(p["path"], p["box"]) for p in parts if p["type"] == "image")


# ----------------------------------------------------------------------------------- model
def new_usage():
    return {"calls": 0, "in_tokens": 0, "out_tokens": 0, "images": 0}


# Windows memory fix: flash attention is not compiled into Windows PyTorch, and grouped-query attention then
# falls back to the math SDPA kernel, whose memory grows with the square of the token count. Expanding the
# key/value heads (and sending no attention mask, batch size 1) lets the memory-efficient kernel run.
import transformers.integrations.sdpa_attention as _sdpa  # noqa: E402
_sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False


class QwenAgent:
    def __init__(self, model_id=QWEN_B, load_in_4bit=False):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        kw = dict(dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        if load_in_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.name = model_id.split("/")[-1] + ("-4bit" if load_in_4bit else "")
        tok = self.processor.tokenizer
        self._letter_ids = []
        for L in LETTERS:
            ids = {tok.convert_tokens_to_ids(L)} | {tok.encode(" " + L, add_special_tokens=False)[0]}
            self._letter_ids.append(sorted(i for i in ids if i is not None and i != tok.unk_token_id))
        self.usage = new_usage()

    def _inputs(self, system, parts, prefix=""):
        content, images = [], []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                content.append({"type": "image"})
                images.append(load_frame(p["path"], p.get("box", CLOSEUP_BOX)))
        messages = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
                   [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + prefix
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt")
        inputs.pop("attention_mask", None)
        self.usage["calls"] += 1
        self.usage["images"] += len(images)
        self.usage["in_tokens"] += int(inputs["input_ids"].shape[1])
        return inputs.to(self.model.device, dtype=torch.bfloat16)

    @torch.inference_mode()
    def generate(self, system, parts, max_new_tokens=256):
        inputs = self._inputs(system, parts)
        n = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, top_k=None)
        self.usage["out_tokens"] += int(out.shape[1] - n)
        return self.processor.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def choose(self, system, parts, n_options, prefix=ANSWER_PREFIX):
        """Answer a multiple-choice question by reading the option letters' next-token probabilities."""
        inputs = self._inputs(system, parts, prefix)
        logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        per = torch.stack([torch.logsumexp(logp[ids], 0) for ids in self._letter_ids[:n_options]])
        probs = torch.softmax(per, 0).cpu().numpy()
        i = int(np.argmax(probs))
        return dict(pred=LETTERS[i], conf=float(probs[i]), probs=[round(float(p), 4) for p in probs],
                    letter_mass=round(float(per.exp().sum()), 4))

    def route(self, question):
        r = self.choose(None, [{"type": "text", "text": ROUTER_PROMPT.format(question=question)}], 5, prefix="(")
        return ROUTER_LABELS[LETTERS.index(r["pred"])], r["conf"]

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def route_cached(agent, s, cache_path):
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if s["qid"] not in cache:
        cache[s["qid"]] = list(agent.route(s["question"]))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=0))
    return cache[s["qid"]]


def answer(agent, s, parts, extra=None):
    """One multiple-choice pass; returns a result record with cost accounting."""
    agent.usage = new_usage()
    t0 = time.time()
    b = agent.choose(mc_system(s), parts + [{"type": "text", "text": mc_text(s)}], len(s["options"]))
    rec = dict(qid=s["qid"], split=s["split"], dataset=s["dataset"], category=s["category"], task=s["task"],
               n_frames_available=len(s["frames"]), answer=s["answer"], pred=b["pred"],
               correct=b["pred"] == s["answer"], conf=round(b["conf"], 4), probs=b["probs"],
               letter_mass=b["letter_mass"], usage=dict(agent.usage), payload_bytes=parts_bytes(parts),
               seconds=round(time.time() - t0, 2), model=agent.name)
    rec.update(extra or {})
    return rec


# ---------------------------------------------------------------------------------- runner
def load_jsonl(path):
    if not Path(path).exists():
        return pd.DataFrame()
    last = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["qid"]] = r
    return pd.DataFrame(list(last.values()))


def run_split(fn, samples, path, desc="", progress=None):
    """Resumable: appends one JSON line per question and skips questions already done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = load_jsonl(path)
    done = set() if prev.empty else set(prev.qid[prev["error"].isna()])
    todo = [s for s in samples if s["qid"] not in done]
    print(f"{path.name}: {len(done)} done, {len(todo)} to run")
    bar = (progress or (lambda x, **k: x))(todo, desc=desc or path.stem[:40])
    with open(path, "a", encoding="utf-8") as f:
        for s in bar:
            try:
                rec = fn(s)
                rec["error"] = None
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                rec = dict(qid=s["qid"], dataset=s["dataset"], category=s["category"], error=f"CUDA OOM: {e}"[:400])
            except Exception as e:  # noqa: BLE001
                rec = dict(qid=s["qid"], dataset=s["dataset"], category=s["category"], error=repr(e)[:400])
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
            f.flush()
    return load_jsonl(path)
