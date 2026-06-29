"""Build a single clean reference clip from a folder of a character's clips.

Trims leading/trailing silence from each clip and concatenates them (with a
small gap) into one reference, capped at a target duration. Use the result as
the `--target` for inference.py so the voice prompt reflects many clips instead
of a single short utterance, while staying under inference.py's 25 s truncation
and 30 s context window.

  python build_reference.py --folder data/CthulhuFishH --out output/refs/CthulhuFishH_ref.wav
"""
import os
import glob
import argparse
import numpy as np
import librosa
import soundfile as sf


def list_clips(folder):
    return sorted(glob.glob(os.path.join(folder, "*.wav")))


def build_reference(folder, out_path, sr=22050, top_db=30, max_dur=18.0, gap=0.04):
    """Trim silence + concatenate clips up to max_dur seconds. Returns out_path."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    gap_samp = int(gap * sr)
    silence = np.zeros(gap_samp, dtype=np.float32)
    pieces, used, total = [], 0, 0.0
    for f in list_clips(folder):
        y, _ = librosa.load(f, sr=sr)
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        if len(yt) == 0:
            continue
        if total + len(yt) / sr > max_dur:
            break
        pieces.append(yt)
        pieces.append(silence)
        used += 1
        total += len(yt) / sr + gap
    if not pieces:
        raise SystemExit(f"No usable audio found in {folder}")
    wave = np.concatenate(pieces[:-1])  # drop trailing gap
    sf.write(out_path, wave, sr)
    print(f"wrote {out_path}  ({len(wave)/sr:.1f}s from {used} clips, silence-trimmed)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dur", type=float, default=18.0)
    ap.add_argument("--top-db", type=float, default=30.0)
    args = ap.parse_args()
    build_reference(args.folder, args.out, top_db=args.top_db, max_dur=args.max_dur)


if __name__ == "__main__":
    main()
