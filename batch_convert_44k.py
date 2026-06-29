"""Batch voice conversion for the f0-conditioned 44 kHz model.

Loads the model ONCE and converts every audio file in a source folder into the
target voice, using the f0-conditioned inference path (matches inference.py with
--f0-condition True). Writes 44.1 kHz WAVs. No ffmpeg needed.

  python batch_convert_44k.py --source-dir "examples/source/翻车鱼3阶段" \
      --target-dir data/sheep_angryman_test \
      --checkpoint runs/sheep_angryman_44k_ft/ft_model.pth \
      --config runs/sheep_angryman_44k_ft/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml
"""
import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # Chinese paths on Windows GBK console
import argparse
import glob

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import librosa
import soundfile as sf

import inference  # load_models(), crossfade(), adjust_f0_semitones()
from build_reference import build_reference


AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")


@torch.no_grad()
def convert_one(source, ref, models, sr, hop_length, device, fp16,
                steps, length_adjust, cfg_rate, f0_condition, auto_f0_adjust, pitch_shift):
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel = models
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(ref, sr=sr)[0]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        S_alt_list, buffer, traversed_time = [], None, 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
            S_alt = semantic_fn(chunk)
            S_alt_list.append(S_alt if traversed_time == 0 else S_alt[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = to_mel(source_audio.float())
    mel2 = to_mel(ref_audio.float())
    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(device)

    feat2 = kaldi.fbank(ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        F0_ori = torch.from_numpy(f0_fn(ori_waves_16k[0], thred=0.03)).to(device)[None]
        F0_alt = torch.from_numpy(f0_fn(converted_waves_16k[0], thred=0.03)).to(device)[None]
        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(F0_ori[F0_ori > 1] + 1e-5)
        voiced_log_f0_alt = torch.log(F0_alt[F0_alt > 1] + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = inference.adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)
    else:
        F0_ori = shifted_f0_alt = None

    cond, _, _, _, _ = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, _, _, _, _ = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    chunks = []
    previous_chunk = None
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition, torch.LongTensor([cat_condition.size(1)]).to(device),
                mel2, style2, None, steps, inference_cfg_rate=cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                chunks.append(vc_wave[0].cpu().numpy())
                break
            chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            chunks.append(inference.crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
            break
        else:
            chunks.append(inference.crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len))
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    return sr, np.concatenate(chunks).astype(np.float32)


def main(args):
    margs = argparse.Namespace(checkpoint=args.checkpoint, config=args.config,
                               fp16=not args.fp32, f0_condition=args.f0_condition)
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args = inference.load_models(margs)
    models = (model, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel)
    device = inference.device
    fp16 = not args.fp32
    sr = 44100 if args.f0_condition else 22050
    hop_length = 512 if args.f0_condition else 256

    # Reference prompt: build from the target character folder (or use a single clip).
    if args.target_dir:
        char = os.path.basename(args.target_dir.rstrip("/\\"))
        ref = build_reference(args.target_dir, os.path.join(args.out, f"ref_{char}.wav"),
                              sr=sr, max_dur=args.ref_dur)
    else:
        ref = args.target

    sources = sorted(f for f in glob.glob(os.path.join(args.source_dir, "*"))
                     if f.lower().endswith(AUDIO_EXT))
    if not sources:
        raise SystemExit(f"No audio files found in {args.source_dir}")
    os.makedirs(args.out, exist_ok=True)
    print(f"Converting {len(sources)} files -> {args.out}  (ref: {os.path.basename(ref)})")

    for i, src in enumerate(sources, 1):
        out_sr, wav = convert_one(src, ref, models, sr, hop_length, device, fp16,
                                  args.diffusion_steps, args.length_adjust, args.inference_cfg_rate,
                                  args.f0_condition, args.auto_f0_adjust, args.semi_tone_shift)
        name = "vc_" + os.path.splitext(os.path.basename(src))[0] + ".wav"
        out_path = os.path.join(args.out, name)
        sf.write(out_path, wav, out_sr)
        print(f"  [{i}/{len(sources)}] {os.path.basename(src)} -> {name}  ({len(wav)/out_sr:.2f}s)")

    print(f"Done. {len(sources)} files written to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True)
    p.add_argument("--target-dir", default=None, help="character folder -> concat reference prompt")
    p.add_argument("--target", default=None, help="single reference clip (if not using --target-dir)")
    p.add_argument("--checkpoint", default="runs/sheep_angryman_44k_ft/ft_model.pth")
    p.add_argument("--config", default="runs/sheep_angryman_44k_ft/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
    p.add_argument("--out", default="output/sheep44k_batch")
    p.add_argument("--diffusion-steps", type=int, default=30)
    p.add_argument("--length-adjust", type=float, default=1.0)
    p.add_argument("--inference-cfg-rate", type=float, default=0.7)
    p.add_argument("--f0-condition", action="store_true", default=True)
    p.add_argument("--auto-f0-adjust", action="store_true", default=True)
    p.add_argument("--semi-tone-shift", type=int, default=0)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--ref-dur", type=float, default=18.0)
    args = p.parse_args()
    if not args.target_dir and not args.target:
        args.target_dir = "data/sheep_angryman_test"
    main(args)
