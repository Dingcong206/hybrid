import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel

import tempfile
import subprocess
import soundfile as sf
from collections import Counter

# -------------------------
# 0) 路径配置（你只需要改这两行）
# -------------------------
# 建议把 BASE_DIR 指向真正存音频的目录（例如 Extracted_data）
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data/Extracted_data"
COSWARA_CSV = "/data/dingcong/hybrid/Coswara-Data/combined_data.csv"

SAVE_DIR = os.path.join(os.path.dirname(COSWARA_CSV), "coswara_hear_patches_expert")
OUT_CSV  = os.path.join(os.path.dirname(COSWARA_CSV), "coswara_metadata_expert.csv")

MODEL_ID = "google/hear-pytorch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

LABEL_MAP = {
    "healthy": 0,
    "positive_mild": 1,
    "positive_moderate": 1,
    "positive_asymp": 1,
}

# -------------------------
# 1) 导入 HeAR 源码工具（确认路径正确）
# -------------------------
sys.path.append("/data/dingcong/hybrid")
try:
    import hear.python.data_processing.audio_utils as audio_utils
except ImportError as e:
    raise SystemExit("❌ 无法导入 hear.python.data_processing.audio_utils：请确认 /data/dingcong/hybrid 下有 hear/ 源码目录") from e


# -------------------------
# 2) ffmpeg 解码：任何格式 -> 16k 单声道 wav -> torch [1, T]
# -------------------------
def load_audio_ffmpeg(path: str, target_sr: int = 16000) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(target_sr), tmp.name]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if p.returncode != 0:
            # 抛出带 stderr 的错误，方便上层统计
            raise RuntimeError(f"ffmpeg_failed: {p.stderr.strip()[-300:]}")

        audio, sr = sf.read(tmp.name)
        # audio: (T,) 或 (T,1)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, T]
        return audio


# -------------------------
# 3) 扫描任务：过滤垃圾文件/空文件/._ 文件
# -------------------------
def build_tasks(label_lookup: dict) -> list:
    tasks = []
    skip_dot = 0
    skip_small = 0
    skip_non_audio = 0
    skip_no_uid = 0
    skip_no_label = 0

    for root, dirs, files in os.walk(BASE_DIR):
        # 跳过你自己生成特征的目录
        if "coswara_hear_patches" in root:
            continue
        if "__MACOSX" in root:
            continue

        for f in files:
            # 1) 过滤 Mac 资源叉文件
            if f.startswith("._") or f == ".DS_Store":
                skip_dot += 1
                continue

            # 2) 只保留音频后缀
            if not f.lower().endswith((".wav", ".webm")):
                skip_non_audio += 1
                continue

            full_path = os.path.join(root, f)

            # 3) 过滤空/极小文件（很多损坏文件会 < 1KB）
            try:
                if os.path.getsize(full_path) < 1024:
                    skip_small += 1
                    continue
            except OSError:
                skip_small += 1
                continue

            # 4) 从路径里找 u_id（Coswara id 通常很长）
            parts = root.split(os.sep)
            u_id = next((p for p in parts if len(p) > 20), None)
            if not u_id:
                skip_no_uid += 1
                continue

            # 5) 必须有标签
            status = label_lookup.get(u_id, None)
            if status is None or status not in LABEL_MAP:
                skip_no_label += 1
                continue

            tasks.append({
                "u_id": u_id,
                "f_name": f,
                "path": full_path,
                "label": LABEL_MAP[status]
            })

    print("\n📌 扫描统计：")
    print(f"  跳过 ._* / .DS_Store: {skip_dot}")
    print(f"  跳过 非音频文件:      {skip_non_audio}")
    print(f"  跳过 小文件(<1KB):   {skip_small}")
    print(f"  跳过 无法解析u_id:   {skip_no_uid}")
    print(f"  跳过 无标签/不在映射: {skip_no_label}")
    return tasks


def main():
    print("🚀 正在启动 HeAR 专家模式 (带声学检测器)...")

    # 0) 快速自检：CSV 是否存在
    if not os.path.exists(COSWARA_CSV):
        raise SystemExit(f"❌ 找不到 combined_data.csv：{COSWARA_CSV}")

    # 1) 加载模型
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()

    # 2) 读取元数据并建索引
    df_raw = pd.read_csv(COSWARA_CSV)
    df_raw["id"] = df_raw["id"].astype(str).str.strip()
    df_raw["covid_status"] = df_raw["covid_status"].astype(str).str.strip()
    label_lookup = dict(zip(df_raw["id"], df_raw["covid_status"]))

    # 3) 扫描任务
    tasks = build_tasks(label_lookup)
    print(f"\n📊 匹配成功：{len(tasks)} 个音频。开始执行检测与提取...")

    meta_data = []
    stats = {"success": 0, "skip_no_event": 0, "decode_fail": 0, "other_fail": 0}
    err_counter = Counter()

    pbar = tqdm(total=len(tasks))

    with torch.no_grad():
        for task in tasks:
            try:
                # A) 解码
                try:
                    waveform = load_audio_ffmpeg(task["path"])  # [1, T] float32
                except Exception as e:
                    stats["decode_fail"] += 1
                    msg = str(e)
                    # 粗分类
                    if "Invalid data found" in msg:
                        err_counter["invalid_data"] += 1
                    elif "Permission denied" in msg:
                        err_counter["permission"] += 1
                    elif "No such file" in msg:
                        err_counter["missing"] += 1
                    else:
                        err_counter["decode_other"] += 1

                    # 只打印前5个解码错误
                    if stats["decode_fail"] <= 5:
                        print("\n❌ decode failed:", task["path"])
                        print("   ", msg[:200])
                    continue

                # B) HeAR 预处理（AED + 裁剪/对齐）
                spec = audio_utils.preprocess_audio(waveform)  # 期望 [1, 32000] 或 None
                if spec is None:
                    stats["skip_no_event"] += 1
                    continue

                # C) 提取 embeddings（你要的 ViT 前 patch embedding）
                audio_input = spec.to(DEVICE)
                out = model.embeddings(audio_input)
                x = out[0] if isinstance(out, (tuple, list)) else out

                feat = x.squeeze(0).cpu().numpy()

                # D) 保存
                save_name = f"{task['u_id']}_{task['f_name'].replace('.', '_')}.npy"
                save_path = os.path.join(SAVE_DIR, save_name)
                np.save(save_path, feat)

                meta_data.append({
                    "user_id": task["u_id"],
                    "original_wav": task["f_name"],
                    "feature_path": save_path,
                    "label": task["label"],
                    "shape0": feat.shape[0],
                    "shape1": feat.shape[1] if feat.ndim == 2 else -1,
                })
                stats["success"] += 1

            except Exception as e:
                stats["other_fail"] += 1
                if stats["other_fail"] <= 3:
                    print("\n❌ other failed:", task["path"])
                    print("   ", repr(e))

            finally:
                pbar.update(1)

    pbar.close()

    # 4) 输出总结
    print("\n✨ 提取总结:")
    print(f"✅ 成功特征数: {stats['success']}")
    print(f"🚫 检测器过滤(无有效音): {stats['skip_no_event']}")
    print(f"❌ 解码失败: {stats['decode_fail']}")
    print(f"❌ 其他失败: {stats['other_fail']}")
    if err_counter:
        print("📌 解码失败原因Top:", err_counter.most_common(10))

    if meta_data:
        pd.DataFrame(meta_data).to_csv(OUT_CSV, index=False)
        print(f"\n📄 已写出: {OUT_CSV}")
        print(f"📁 特征目录: {SAVE_DIR}")
    else:
        print("❌ 最终未生成任何特征：请检查 BASE_DIR / CSV / 解码错误类型统计。")


if __name__ == "__main__":
    main()
