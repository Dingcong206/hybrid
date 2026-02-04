import torch
from pathlib import Path


def scan_checkpoints(root_dir="/data/dingcong/hybrid"):
    print(f"{'Folder Name':<35} | {'ICBHI':<8} | {'Epoch':<5} | {'Last Modified'}")
    print("-" * 75)

    # 搜索目录下所有的 .pt 文件
    for pt_file in Path(root_dir).rglob("*.pt"):
        try:
            # map_location='cpu' 确保没有 GPU 也能读
            ckpt = torch.load(pt_file, map_location='cpu')

            # 提取你在代码中保存的字段
            icbhi = ckpt.get('best_icbhi', ckpt.get('best_acc', 0.0))
            epoch = ckpt.get('epoch', 'N/A')
            mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(pt_file.stat().st_mtime))

            print(f"{pt_file.parent.name:<35} | {icbhi:<8.4f} | {epoch:<5} | {mtime}")
        except Exception:
            # 略过非模型文件或损坏的文件
            continue


if __name__ == "__main__":
    import time

    scan_checkpoints()