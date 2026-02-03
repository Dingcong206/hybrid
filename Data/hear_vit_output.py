
import argparse
import numpy as np
import torch
import torchaudio
from torchaudio import transforms as T
import torch.nn.functional as F
from transformers import AutoModel

SR = 16000
TARGET_SAMPLES = 32000

def load_wav_2s(path: str):
    wav, orig_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != SR:
        wav = T.Resample(orig_sr, SR)(wav)

    n = wav.shape[-1]
    if n > TARGET_SAMPLES:
        wav = wav[:, :TARGET_SAMPLES]
    elif n < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - n
        wav = F.pad(wav, (pad//2, pad - pad//2))
    return wav  # (1, 32000)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--wav", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(device)
    model.eval()

    x = load_wav_2s(args.wav).to(device).float()  # (1,32000)

    out = model(x, return_dict=True)

    print("\n=== Model output fields ===")
    for k in out.keys():
        v = out[k]
        if hasattr(v, "shape"):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {type(v)}")

    # 1) pooler_output (best)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        emb = out.pooler_output.squeeze(0)
        source = "pooler_output"
    else:
        # 2) CLS from last_hidden_state
        if not hasattr(out, "last_hidden_state") or out.last_hidden_state is None:
            raise RuntimeError("No pooler_output and no last_hidden_state. Cannot get ViT embedding.")
        emb = out.last_hidden_state[:, 0, :].squeeze(0)
        source = "cls_from_last_hidden_state"

    emb_np = emb.detach().cpu().numpy()
    print(f"\n✅ ViT embedding source: {source}")
    print("✅ embedding shape:", emb_np.shape)
    print("✅ mean/std:", float(emb_np.mean()), float(emb_np.std()))
    print("✅ nan:", bool(np.isnan(emb_np).any()), "inf:", bool(np.isinf(emb_np).any()))

if __name__ == "__main__":
    main()

