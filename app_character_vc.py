"""Character Voice Conversion GUI.

Upload any audio (single file or a zip of many), pick a target character, convert.
Users can choose between two fine-tuned models:
  * 干声 (dry / no effects)  - trained on the -6 dBFS normalized, effect-free set
  * 含特效 (with effects)    - trained on the original (effected) set

Both are 44.1 kHz f0-conditioned models with identical architecture, so the heavy
shared components are loaded once and only the fine-tuned DiT weights are swapped
when the user switches models. Each model uses references + character list drawn
from its own training data. WAV output, no ffmpeg needed.

  python app_character_vc.py
  python app_character_vc.py --prebuild      # build all references up front
"""
import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import argparse
import glob
import shutil
import zipfile

import yaml
import gradio as gr
import soundfile as sf

import inference
from modules.commons import load_checkpoint
from build_reference import build_reference
from batch_convert_44k import convert_one

AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")

# Registry of selectable models. Both share the same 44 kHz f0 architecture.
MODELS = {
    "干声 (dry / 无特效)": {
        "ckpt": "runs/game_vc_44k_clean_ft/ft_model.pth",
        "config": "runs/game_vc_44k_clean_ft/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        "data": "data/_pgameVO_normalized",
        "refdir": "output/refs_44k_dry",
    },
    "含特效 (with effects)": {
        "ckpt": "runs/game_vc_44k_ft/ft_model.pth",
        "config": "runs/game_vc_44k_ft/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        "data": "data",
        "refdir": "output/refs_44k_fx",
    },
}
DEFAULT_MODEL = "干声 (dry / 无特效)"


def list_characters(data_dir):
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and glob.glob(os.path.join(data_dir, d, "*.wav"))
    )


def find_audio(root):
    found = []
    for dirpath, _, files in os.walk(root):
        if "__MACOSX" in dirpath:
            continue
        for f in files:
            if f.startswith("._"):
                continue
            if f.lower().endswith(AUDIO_EXT):
                found.append(os.path.join(dirpath, f))
    return sorted(found)


def main(args):
    # Keep only models whose checkpoint exists.
    available = {k: v for k, v in MODELS.items() if os.path.exists(v["ckpt"])}
    if not available:
        raise SystemExit("No model checkpoints found. Train a model first.")
    default_label = DEFAULT_MODEL if DEFAULT_MODEL in available else next(iter(available))

    # Load shared components once, using the default model's checkpoint/config.
    fp16 = not args.fp32
    cfg0 = available[default_label]["config"]
    f0_condition = yaml.safe_load(open(cfg0))["model_params"]["DiT"].get("f0_condition", False)
    margs = argparse.Namespace(checkpoint=available[default_label]["ckpt"],
                               config=cfg0, fp16=fp16, f0_condition=f0_condition)
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel, _ = inference.load_models(margs)
    models = (model, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel)
    device = inference.device
    sr = 44100 if f0_condition else 22050
    hop_length = 512 if f0_condition else 256

    state = {"current": default_label}

    def ensure_model(label):
        """Hot-swap the fine-tuned DiT weights if a different model was selected."""
        if label == state["current"]:
            return
        load_checkpoint(model, None, available[label]["ckpt"],
                        load_only_params=True, ignore_modules=[], is_distributed=False)
        state["current"] = label

    chars_for = {k: list_characters(v["data"]) for k, v in available.items()}
    ref_cache = {}

    def get_reference(label, character):
        key = (label, character)
        if key not in ref_cache:
            os.makedirs(available[label]["refdir"], exist_ok=True)
            out = os.path.join(available[label]["refdir"], f"{character}.wav")
            if not os.path.exists(out):
                build_reference(os.path.join(available[label]["data"], character),
                                out, sr=sr, max_dur=args.ref_dur)
            ref_cache[key] = out
        return ref_cache[key]

    if args.prebuild:
        for label in available:
            print(f"Pre-building references for {label} ({len(chars_for[label])} characters)...")
            for c in chars_for[label]:
                get_reference(label, c)

    def _vc(source, ref, steps, length_adjust, cfg_rate):
        return convert_one(source, ref, models, sr, hop_length, device, fp16,
                           int(steps), length_adjust, cfg_rate,
                           f0_condition, args.auto_f0_adjust, 0)

    def convert_single(model_label, source, character, steps, length_adjust, cfg_rate):
        if source is None:
            raise gr.Error("请先上传或录制一段源音频。")
        if not character:
            raise gr.Error("请选择一个目标角色。")
        ensure_model(model_label)
        return _vc(source, get_reference(model_label, character), steps, length_adjust, cfg_rate)

    def convert_zip(model_label, zip_path, character, steps, length_adjust, cfg_rate,
                    progress=gr.Progress()):
        if zip_path is None:
            raise gr.Error("请先上传一个 zip 压缩包。")
        if not character:
            raise gr.Error("请选择一个目标角色。")
        ensure_model(model_label)
        base = os.path.splitext(os.path.basename(zip_path))[0]
        run_dir = os.path.join(args.batch_out, f"{base}__to__{character}")
        in_dir, out_dir = os.path.join(run_dir, "input"), os.path.join(run_dir, "output")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(in_dir)
        except zipfile.BadZipFile:
            raise gr.Error("上传的文件不是有效的 zip 压缩包。")
        sources = find_audio(in_dir)
        if not sources:
            raise gr.Error("zip 中未找到音频文件（支持 wav/mp3/flac/ogg/m4a/opus）。")

        ref = get_reference(model_label, character)
        ok, fails = 0, []
        for i, src in enumerate(sources):
            progress(i / len(sources), desc=f"转换中 {i + 1}/{len(sources)}")
            try:
                out_sr, wav = _vc(src, ref, steps, length_adjust, cfg_rate)
                rel = os.path.relpath(src, in_dir)
                dst = os.path.join(out_dir, os.path.dirname(rel),
                                   "vc_" + os.path.splitext(os.path.basename(rel))[0] + ".wav")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                sf.write(dst, wav, out_sr)
                ok += 1
            except Exception as e:
                fails.append(f"{os.path.basename(src)}: {e}")
        progress(1.0, desc="打包中…")
        zip_out = os.path.join(run_dir, f"{base}__to__{character}.zip")
        shutil.make_archive(zip_out[:-4], "zip", out_dir)
        summary = f"完成：{ok}/{len(sources)} 个文件已用「{model_label}」转换为「{character}」。\n输出文件夹：{out_dir}"
        if fails:
            summary += f"\n失败 {len(fails)} 个：\n  " + "\n  ".join(fails[:10])
        return zip_out, summary

    def step_slider():
        return gr.Slider(1, 100, value=30, step=1, label="扩散步数", info="默认 30；50 以上质量最佳")

    def len_slider():
        return gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="长度调整", info="<1.0 加速，>1.0 减速")

    def cfg_slider():
        return gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="推理 CFG 系数")

    init_chars = chars_for[default_label]
    with gr.Blocks(title="角色变声") as demo:
        gr.Markdown(
            f"# 🎭 角色变声\n"
            f"将**任意**音频转换为某个**目标角色**的声音（44.1 kHz）。\n\n"
            f"先选择**模型**：**干声**（无特效、更干净）或**含特效**（原始素材）。"
            f"切换模型会自动更新可选角色列表并热加载对应权重。"
        )
        model_sel = gr.Radio(list(available.keys()), value=default_label, label="模型选择")
        with gr.Tabs():
            with gr.Tab("单文件转换"):
                with gr.Row():
                    with gr.Column():
                        src = gr.Audio(type="filepath", label="源音频（上传任意 .wav 文件）")
                        ch1 = gr.Dropdown(init_chars, value=init_chars[0], label="目标角色")
                        st1, ln1, cf1 = step_slider(), len_slider(), cfg_slider()
                        btn1 = gr.Button("开始转换 🎙️", variant="primary")
                    with gr.Column():
                        out1 = gr.Audio(label="转换后的音频", type="numpy")
                btn1.click(convert_single, [model_sel, src, ch1, st1, ln1, cf1], out1)

            with gr.Tab("批量转换（上传 zip）"):
                gr.Markdown("上传一个 **zip 压缩包**（可含子文件夹），批量转换为目标角色，完成后下载结果 zip。"
                            "支持 wav/mp3/flac/ogg/m4a/opus。")
                with gr.Row():
                    with gr.Column():
                        zip_in = gr.File(label="上传 zip 压缩包", file_types=[".zip"], type="filepath")
                        ch2 = gr.Dropdown(init_chars, value=init_chars[0], label="目标角色")
                        st2, ln2, cf2 = step_slider(), len_slider(), cfg_slider()
                        btn2 = gr.Button("开始批量转换 🎙️", variant="primary")
                    with gr.Column():
                        zip_out = gr.File(label="转换结果（zip 下载）")
                        summary = gr.Textbox(label="状态", lines=6)
                btn2.click(convert_zip, [model_sel, zip_in, ch2, st2, ln2, cf2], [zip_out, summary])

        def on_model_change(label):
            chs = chars_for[label]
            return (gr.update(choices=chs, value=chs[0] if chs else None),
                    gr.update(choices=chs, value=chs[0] if chs else None))

        model_sel.change(on_model_change, model_sel, [ch1, ch2])

    demo.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-out", type=str, default="output/batch_gui")
    parser.add_argument("--ref-dur", type=float, default=18.0)
    parser.add_argument("--auto-f0-adjust", action="store_true", default=True)
    parser.add_argument("--prebuild", action="store_true", help="build all references at startup")
    parser.add_argument("--share", action="store_true", help="create a public Gradio share link")
    parser.add_argument("--fp32", action="store_true",
                        help="use full fp32 inference (cleaner/less fuzzy, slower) instead of fp16")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
