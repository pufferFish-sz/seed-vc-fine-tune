"""Objective speaker-similarity metric (SECS) for voice-conversion outputs.

Computes **Speaker Embedding Cosine Similarity (SECS)** between audio files: each
clip is mapped to a fixed-length speaker-identity vector, and similarity is the
cosine of the angle between vectors (range -1..1; 1 = identical identity).
This is the standard objective metric for "does the converted audio sound like
the target speaker", used instead of / alongside subjective listening.

Encoders:
  * campplus  - CAMPPlus speaker-verification model shipped with this repo.
                Note: this is also the speaker encoder Seed-VC conditions timbre
                on, so its scores are slightly optimistic for this model. Still a
                valid, zero-setup measure of speaker identity.
  * resemblyzer - independent GE2E speaker encoder (only if `resemblyzer` is
                installed: `pip install resemblyzer`). Use for an unbiased
                cross-check. Auto-skipped if not available.

Usage:
  # before/after: how close are base vs fine-tuned outputs to the target voice?
  python eval_similarity.py --ref data/Fox/VO_Fox_idle_1.wav \
      --inputs output/base/vc_*.wav output/ft/vc_*.wav

  # arbitrary set -> full pairwise matrix
  python eval_similarity.py --inputs a.wav b.wav c.wav
"""
import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
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


def cos(a, b):
    return float(torch.dot(a, b).clamp(-1, 1))


def report(embs, ref, inputs, encoder_name):
    print(f"\n=== SECS via {encoder_name} ===")
    if ref:
        print(f"Cosine similarity vs reference  [{os.path.basename(ref)}]")
        print("-" * 56)
        ranked = sorted(inputs, key=lambda p: cos(embs[ref], embs[p]), reverse=True)
        for p in ranked:
            print(f"  {cos(embs[ref], embs[p]):+.4f}   {os.path.basename(p)}")
    names = list(embs.keys())
    short = [os.path.basename(n)[:20] for n in names]
    print("\nPairwise matrix:")
    print(" " * 22 + " ".join(f"{s:>20}" for s in short))
    for ni, si in zip(names, short):
        row = " ".join(f"{cos(embs[ni], embs[nj]):>20.4f}" for nj in names)
        print(f"{si:>20}  {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=str, default=None, help="reference / target voice clip")
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="clips to score")
    ap.add_argument("--encoder", choices=["campplus", "resemblyzer", "both"], default="both")
    args = ap.parse_args()

    device = _device()
    # de-duplicate while keeping order
    paths = list(dict.fromkeys(([args.ref] if args.ref else []) + args.inputs))
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("Missing files:\n  " + "\n  ".join(missing))

    if args.encoder in ("campplus", "both"):
        m = load_campplus(device)
        embs = {p: campplus_embed(p, m, device) for p in paths}
        report(embs, args.ref, args.inputs, "CAMPPlus (repo speaker encoder)")

    if args.encoder in ("resemblyzer", "both"):
        enc = try_load_resemblyzer()
        if enc is None:
            if args.encoder == "resemblyzer":
                raise SystemExit("resemblyzer not installed. Run: pip install resemblyzer")
            print("\n[resemblyzer not installed -> skipping independent cross-check;"
                  " `pip install resemblyzer` to enable]")
        else:
            embs = {p: resemblyzer_embed(p, enc) for p in paths}
            report(embs, args.ref, args.inputs, "Resemblyzer (independent GE2E)")


if __name__ == "__main__":
    main()
