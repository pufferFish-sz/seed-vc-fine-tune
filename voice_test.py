"""Cross-character voice-conversion test with per-character centroid targets.

For each target character it builds:
  * a centroid  = mean speaker-embedding over ALL the character's clips (the
    rigorous "this character's voice in general" reference for scoring), and
  * a prompt    = a clean silence-trimmed concatenation of its clips (the
    reference fed to inference.py), via build_reference.py.

Then every source clip is converted into every target (with the base model and
the fine-tuned model), and each output is scored with SECS against EVERY
character centroid, using both CAMPPlus and the independent Resemblyzer.

  python voice_test.py                                  # defaults: CthulhuFishH, BatH
  python voice_test.py --targets CthulhuFishH BatH Lion --models base ft
  python voice_test.py --sources examples/source/VO_Alpaca_rest_2.wav

Adding a character is just adding its name to --targets: each gets its own
centroid automatically, so any character can be tested against any other.
"""
import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import argparse
import glob
import subprocess
import sys

import soundfile as sf


def pick_source(char, data_dir):
    fs = glob.glob(os.path.join(data_dir, char, "*.wav"))
    if not fs:
        raise SystemExit(f"No clips for character {char} in {data_dir}")
    return max(fs, key=lambda f: sf.info(f).frames)  # longest = most content


def expected_out(outdir, source, target, length, steps, cfg):
    s = os.path.basename(source).split(".")[0]
    t = os.path.basename(target).split(".")[0]
    return os.path.join(outdir, f"vc_{s}_{t}_{length}_{steps}_{cfg}.wav")


def run_inference(source, target, outdir, checkpoint=None, config=None,
                  steps=25, length=1.0, cfg=0.7):
    os.makedirs(outdir, exist_ok=True)
    out = expected_out(outdir, source, target, length, steps, cfg)
    if os.path.exists(out):
        return out  # idempotent: reuse if already generated
    cmd = [sys.executable, "inference.py",
           "--source", source, "--target", target, "--output", outdir,
           "--diffusion-steps", str(steps), "--length-adjust", str(length),
           "--inference-cfg-rate", str(cfg)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint, "--config", config]
    print(f">>> [{'fine-tuned' if checkpoint else 'base'}] "
          f"{os.path.basename(source)} -> {os.path.basename(target)}")
    subprocess.run(cmd, check=True)
    if not os.path.exists(out):
        raise SystemExit(f"Expected output not found: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["CthulhuFishH", "BatH"])
    ap.add_argument("--sources", nargs="+", default=None,
                    help="source clips; default = longest clip of each target character")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models", nargs="+", default=["base", "ft"], choices=["base", "ft"])
    ap.add_argument("--ft-checkpoint", default="runs/game_vc_ft/ft_model.pth")
    ap.add_argument("--ft-config",
                    default="runs/game_vc_ft/config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    ap.add_argument("--diffusion-steps", type=int, default=25)
    ap.add_argument("--out", default="output/voice_test")
    args = ap.parse_args()

    if "ft" in args.models:
        for p in (args.ft_checkpoint, args.ft_config):
            if not os.path.exists(p):
                raise SystemExit(f"Missing required file: {p}")

    from build_reference import build_reference

    # Per-target: centroid clip list + concat prompt for inference.
    centroid_paths, prompt = {}, {}
    for ch in args.targets:
        clips = sorted(glob.glob(os.path.join(args.data_dir, ch, "*.wav")))
        if not clips:
            raise SystemExit(f"No clips for target {ch}")
        centroid_paths[ch] = clips
        prompt[ch] = build_reference(os.path.join(args.data_dir, ch),
                                     os.path.join(args.out, f"ref_{ch}.wav"))

    sources = args.sources or [pick_source(ch, args.data_dir) for ch in args.targets]
    sources = [s for s in sources if os.path.exists(s)]
    src_label = {s: os.path.basename(s) for s in sources}

    model_cfg = {"base": (None, None),
                 "ft": (args.ft_checkpoint, args.ft_config)}

    # Run every (model, source, target) conversion.
    conv = {}
    for model in args.models:
        ckpt, cfg = model_cfg[model]
        for src in sources:
            for ch in args.targets:
                conv[(model, src, ch)] = run_inference(
                    src, prompt[ch], os.path.join(args.out, model),
                    checkpoint=ckpt, config=cfg, steps=args.diffusion_steps)

    # Score.
    from eval_similarity import (load_campplus, campplus_embed, try_load_resemblyzer,
                                 resemblyzer_embed, centroid, cos, _device)
    device = _device()

    def evaluate(name, embed_fn):
        cache = {}

        def emb(p):
            if p not in cache:
                cache[p] = embed_fn(p)
            return cache[p]

        cen = {ch: centroid(centroid_paths[ch], embed_fn) for ch in args.targets}

        # --- Accuracy: output vs its INTENDED target centroid (base vs ft) ---
        print("\n" + "=" * 78)
        print(f"ACCURACY  (output vs intended target centroid)   via {name}")
        print("higher = reached the target voice;  Δ = ft - base")
        print("-" * 78)
        head = f"{'source':>26} {'-> target':>14}"
        if "base" in args.models:
            head += f" {'base':>8}"
        if "ft" in args.models:
            head += f" {'ft':>8}"
        if {"base", "ft"} <= set(args.models):
            head += f" {'Δ':>8}"
        print(head)
        for src in sources:
            for ch in args.targets:
                row = f"{src_label[src][:26]:>26} {ch:>14}"
                vals = {}
                for model in args.models:
                    vals[model] = cos(cen[ch], emb(conv[(model, src, ch)]))
                    row += f" {vals[model]:>8.3f}"
                if {"base", "ft"} <= set(args.models):
                    row += f" {vals['ft'] - vals['base']:>+8.3f}"
                print(row)

        # --- Discrimination: ft output vs EVERY centroid ---
        if "ft" in args.models and len(args.targets) > 1:
            print("\n" + "-" * 78)
            print(f"DISCRIMINATION  (fine-tuned output vs every centroid)   via {name}")
            print("intended target marked *; correct if its score is the highest in the row")
            print("-" * 78)
            print(f"{'source':>26} {'-> target':>14}  " +
                  " ".join(f"{('vs '+ch)[:12]:>12}" for ch in args.targets))
            for src in sources:
                for ch in args.targets:
                    e = emb(conv[("ft", src, ch)])
                    scores = {c2: cos(cen[c2], e) for c2 in args.targets}
                    best = max(scores, key=scores.get)
                    cells = []
                    for c2 in args.targets:
                        mark = "*" if c2 == ch else (" " if c2 != best else "<")
                        cells.append(f"{scores[c2]:>11.3f}{mark}")
                    ok = "OK" if best == ch else "x"
                    print(f"{src_label[src][:26]:>26} {ch:>14}  " + " ".join(cells) + f"  {ok}")

    m = load_campplus(device)
    evaluate("CAMPPlus (repo speaker encoder)", lambda p: campplus_embed(p, m, device))

    enc = try_load_resemblyzer()
    if enc:
        evaluate("Resemblyzer (independent GE2E)", lambda p: resemblyzer_embed(p, enc))

    print("\nAudio for listening:")
    for model in args.models:
        print(f"  {model:4s} outputs -> {os.path.join(args.out, model)}")
    for ch in args.targets:
        print(f"  prompt[{ch}] -> {prompt[ch]}")


if __name__ == "__main__":
    main()
