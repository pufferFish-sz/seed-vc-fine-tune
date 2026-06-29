"""Objective speaker-similarity metric (SECS) for voice-conversion outputs.

Computes **Speaker Embedding Cosine Similarity (SECS)** between audio files: each
clip is mapped to a fixed-length speaker-identity vector, and similarity is the
cosine of the angle between vectors (range -1..1; 1 = identical identity).
This is the standard objective metric for "does the converted audio sound like
the target speaker", used instead of / alongside subjective listening.

The reference voice can be either a single clip (--ref) or the **centroid** of an
entire folder of a character's clips (--ref-dir) -- the centroid (mean of the
per-clip embeddings) represents the character's voice in general and is the
rigorous target, equivalent to multi-utterance speaker enrollment.

Encoders:
  * campplus    - CAMPPlus speaker-verification model shipped with this repo
                  (also the encoder Seed-VC conditions timbre on -> slightly
                  optimistic, but a valid, zero-setup identity measure).
  * resemblyzer - independent GE2E speaker encoder (if installed). Unbiased
                  cross-check. Auto-skipped if not available.

Usage:
  python eval_similarity.py --ref-dir data/CthulhuFishH \
      --inputs output/compare/base/vc_*.wav output/compare/ft/vc_*.wav
  python eval_similarity.py --ref a.wav --inputs b.wav c.wav
  python eval_similarity.py --inputs a.wav b.wav c.wav        # pairwise matrix
"""
import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import glob
import argparse
import torch
import torchaudio.compliance.kaldi as kaldi
import librosa

from hf_utils import load_custom_model_from_hf
from modules.campplus.DTDNN import CAMPPlus


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- CAMPPlus (repo's own speaker encoder) ----------------
def load_campplus(device):
    ckpt = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    m = CAMPPlus(feat_dim=80, embedding_size=192)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval().to(device)
    return m


@torch.no_grad()
def campplus_embed(path, model, device):
    wav, _ = librosa.load(path, sr=16000)
    wav = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = model(feat.unsqueeze(0)).squeeze(0)
    return torch.nn.functional.normalize(emb, dim=-1)


# ---------------- Resemblyzer (optional, independent) ----------------
def try_load_resemblyzer():
    try:
        from resemblyzer import VoiceEncoder
        return VoiceEncoder(device="cpu")
    except Exception:
        return None


def resemblyzer_embed(path, encoder):
    from resemblyzer import preprocess_wav
    import numpy as np
    emb = encoder.embed_utterance(preprocess_wav(path))
    t = torch.from_numpy(np.asarray(emb)).float()
    return torch.nn.functional.normalize(t, dim=-1)


def centroid(paths, embed_fn):
    """Mean of per-clip embeddings, renormalized -> a speaker 'enrollment' vector."""
    embs = torch.stack([embed_fn(p) for p in paths], dim=0)
    return torch.nn.functional.normalize(embs.mean(dim=0), dim=-1)


def cos(a, b):
    return float(torch.dot(a, b).clamp(-1, 1))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ref", type=str, default=None, help="single reference clip")
    g.add_argument("--ref-dir", type=str, default=None, help="folder -> centroid reference")
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="clips to score")
    ap.add_argument("--encoder", choices=["campplus", "resemblyzer", "both"], default="both")
    args = ap.parse_args()

    device = _device()
    inputs = [p for p in dict.fromkeys(args.inputs) if os.path.exists(p)]
    if not inputs:
        raise SystemExit("No existing input files.")

    if args.ref_dir:
        ref_paths = sorted(glob.glob(os.path.join(args.ref_dir, "*.wav")))
        if not ref_paths:
            raise SystemExit(f"No .wav clips in {args.ref_dir}")
        ref_label = f"{os.path.basename(args.ref_dir.rstrip('/'))} centroid ({len(ref_paths)} clips)"
    elif args.ref:
        ref_paths = [args.ref]
        ref_label = os.path.basename(args.ref)
    else:
        ref_paths, ref_label = None, None

    def run(name, embed_fn):
        print(f"\n=== SECS via {name} ===")
        if ref_paths:
            ref_emb = centroid(ref_paths, embed_fn)
            print(f"Cosine similarity vs reference  [{ref_label}]   (higher = more similar)")
            print("-" * 60)
            for p in sorted(inputs, key=lambda q: cos(ref_emb, embed_fn(q)), reverse=True):
                print(f"  {cos(ref_emb, embed_fn(p)):+.4f}   {os.path.basename(p)}")
        else:
            embs = {p: embed_fn(p) for p in inputs}
            short = [os.path.basename(n)[:20] for n in inputs]
            print("Pairwise matrix:")
            print(" " * 22 + " ".join(f"{s:>20}" for s in short))
            for ni, si in zip(inputs, short):
                print(f"{si:>20}  " + " ".join(f"{cos(embs[ni], embs[nj]):>20.4f}" for nj in inputs))

    if args.encoder in ("campplus", "both"):
        m = load_campplus(device)
        run("CAMPPlus (repo speaker encoder)", lambda p: campplus_embed(p, m, device))
    if args.encoder in ("resemblyzer", "both"):
        enc = try_load_resemblyzer()
        if enc is None:
            if args.encoder == "resemblyzer":
                raise SystemExit("resemblyzer not installed. Run: pip install resemblyzer")
            print("\n[resemblyzer not installed -> skipping independent cross-check]")
        else:
            run("Resemblyzer (independent GE2E)", lambda p: resemblyzer_embed(p, enc))


if __name__ == "__main__":
    main()
