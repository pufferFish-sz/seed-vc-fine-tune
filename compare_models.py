"""Before/after voice-conversion comparison: base Seed-VC vs fine-tuned model.

For each source clip, converts it toward the target voice with (a) the base
pretrained model and (b) the fine-tuned model, then scores every output against
the target with Speaker Embedding Cosine Similarity (SECS).
Higher SECS = the converted audio is closer to the target speaker's identity.

The target voice can be a single clip (--target) or an entire character folder
(--target-dir), which is the better choice for "make it like the whole voice":
  * inference prompt -> a clean silence-trimmed concatenation of the folder's
    clips (see build_reference.py), capped under inference.py's 25s limit;
  * metric -> the centroid (mean embedding) of ALL the folder's clips.

  python compare_models.py --target-dir data/CthulhuFishH     # recommended
  python compare_models.py --target data/Wolf/VO_Wolf_idle_1.wav
  python compare_models.py --target-dir data/Lion --sources examples/source/VO_BatP_cheer_2.wav
"""
import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import argparse
import glob
import subprocess
import sys

DEFAULT_SOURCES = [
    "examples/source/VO_CthulhuFishH_cheer_3.wav",
    "examples/source/VO_BatP_cheer_2.wav",
    "examples/source/VO_DeerH_eat_1.wav",
    "examples/source/VO_Alpaca_rest_2.wav",
    "examples/source/VO_ChameleonH_idle_1.wav",
]


def expected_out(outdir, source, target, length, steps, cfg):
    s = os.path.basename(source).split(".")[0]
    t = os.path.basename(target).split(".")[0]
    return os.path.join(outdir, f"vc_{s}_{t}_{length}_{steps}_{cfg}.wav")


def run_inference(source, target, outdir, checkpoint=None, config=None,
                  steps=25, length=1.0, cfg=0.7):
    os.makedirs(outdir, exist_ok=True)
    cmd = [sys.executable, "inference.py",
           "--source", source, "--target", target, "--output", outdir,
           "--diffusion-steps", str(steps), "--length-adjust", str(length),
           "--inference-cfg-rate", str(cfg)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint, "--config", config]
    tag = "fine-tuned" if checkpoint else "base"
    print(f">>> [{tag}] {os.path.basename(source)} -> {os.path.basename(target)}")
    subprocess.run(cmd, check=True)
    out = expected_out(outdir, source, target, length, steps, cfg)
    if not os.path.exists(out):
        raise SystemExit(f"Expected output not found: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--target", default=None, help="single target voice clip")
    g.add_argument("--target-dir", default=None, help="character folder (centroid + concat prompt)")
    ap.add_argument("--ft-checkpoint", default="runs/game_vc_ft/ft_model.pth")
    ap.add_argument("--ft-config",
                    default="runs/game_vc_ft/config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    ap.add_argument("--diffusion-steps", type=int, default=25)
    ap.add_argument("--out", default="output/compare")
    args = ap.parse_args()

    for p in (args.ft_checkpoint, args.ft_config):
        if not os.path.exists(p):
            raise SystemExit(f"Missing required file: {p}")

    # Resolve target: build a concat reference for the prompt, gather clips for the centroid.
    if args.target_dir:
        from build_reference import build_reference, list_clips
        ref_paths = list_clips(args.target_dir)
        if not ref_paths:
            raise SystemExit(f"No .wav clips in {args.target_dir}")
        char = os.path.basename(args.target_dir.rstrip("/\\"))
        prompt_clip = build_reference(args.target_dir,
                                      os.path.join(args.out, f"ref_{char}.wav"))
        ref_label = f"{char} centroid ({len(ref_paths)} clips)"
    else:
        target = args.target or "data/Wolf/VO_Wolf_idle_1.wav"
        if not os.path.exists(target):
            raise SystemExit(f"Missing target: {target}")
        prompt_clip = target
        ref_paths = [target]
        ref_label = os.path.basename(target)

    sources = [s for s in args.sources if os.path.exists(s)]
    if not sources:
        raise SystemExit("None of the requested source files exist.")

    print(f"\nTarget voice prompt : {prompt_clip}")
    print(f"Metric reference    : {ref_label}\n")

    runs = []  # (source, base_out, ft_out)
    for src in sources:
        base_out = run_inference(src, prompt_clip, os.path.join(args.out, "base"),
                                 steps=args.diffusion_steps)
        ft_out = run_inference(src, prompt_clip, os.path.join(args.out, "ft"),
                               checkpoint=args.ft_checkpoint, config=args.ft_config,
                               steps=args.diffusion_steps)
        runs.append((src, base_out, ft_out))

    from eval_similarity import (load_campplus, campplus_embed, try_load_resemblyzer,
                                 resemblyzer_embed, centroid, cos, _device)
    device = _device()

    def report(name, embed_fn):
        ref = centroid(ref_paths, embed_fn)
        print("\n" + "=" * 76)
        print(f"SECS vs target [{ref_label}]  via {name}")
        print("higher = closer to target voice;  Δ = fine-tuned minus base")
        print("-" * 76)
        print(f"{'source clip':32s} {'source':>8} {'base':>8} {'ft':>8} {'Δ(ft-base)':>12}")
        deltas = []
        for src, b, f in runs:
            ss, sb, sf_ = cos(ref, embed_fn(src)), cos(ref, embed_fn(b)), cos(ref, embed_fn(f))
            deltas.append(sf_ - sb)
            print(f"{os.path.basename(src)[:32]:32s} {ss:>8.3f} {sb:>8.3f} {sf_:>8.3f} {sf_-sb:>+12.3f}")
        avg = sum(deltas) / len(deltas)
        print("-" * 76)
        print(f"{'AVERAGE Δ (ft improvement over base)':>63s} {avg:>+12.3f}")

    m = load_campplus(device)
    report("CAMPPlus (repo speaker encoder)", lambda p: campplus_embed(p, m, device))

    enc = try_load_resemblyzer()
    if enc:
        report("Resemblyzer (independent GE2E)", lambda p: resemblyzer_embed(p, enc))

    print("\nListen & compare:")
    print(f"  base outputs -> {os.path.join(args.out, 'base')}")
    print(f"  ft   outputs -> {os.path.join(args.out, 'ft')}")
    print(f"  target prompt-> {prompt_clip}")


if __name__ == "__main__":
    main()
