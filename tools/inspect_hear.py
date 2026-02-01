import os
import tensorflow as tf

BASE = "/home/guest1/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"

MODEL_DIRS = [
    os.path.join(BASE, "event_detector", "spectrogram_frontend"),
    os.path.join(BASE, "event_detector", "event_detector_small"),
    os.path.join(BASE, "event_detector", "event_detector_large"),
]

def inspect(model_dir):
    print("\n==============================")
    print("MODEL:", model_dir)
    m = tf.saved_model.load(model_dir)
    print("signatures:", list(m.signatures.keys()))
    fn = m.signatures["serving_default"]
    print("input_signature:", fn.structured_input_signature)

    kw = fn.structured_input_signature[1]

    # 尝试用 audio_wav 喂 2 秒音频
    feed = {}
    if "audio_wav" in kw:
        feed["audio_wav"] = tf.zeros([1, 32000], tf.float32)
    else:
        # 若没有 audio_wav，则用第一个输入名，按它的shape造一个全零输入
        first_k = list(kw.keys())[0]
        feed[first_k] = tf.zeros(kw[first_k].shape, tf.float32)

    out = fn(**feed)

    print("outputs:")
    for k, v in out.items():
        print("  ", k, v.shape, v.dtype)

for d in MODEL_DIRS:
    try:
        inspect(d)
    except Exception as e:
        print("\n==============================")
        print("MODEL:", d)
        print("FAILED:", repr(e))
