import torch
from pathlib import Path


def find_all_results(search_path="/data/dingcong/hybrid"):
    print(f"{'文件夹名称':<40} | {'最佳 ICBHI':<10} | {'Epoch'}")
    print("-" * 65)

    # 递归搜索目录下所有的 .pt 文件
    for pt_file in Path(search_path).rglob("*.pt"):
        try:
            ckpt = torch.load(pt_file, map_location='cpu')
            if 'best_icbhi' in ckpt:
                folder_name = pt_file.parent.name
                icbhi = ckpt['best_icbhi']
                epoch = ckpt['epoch']
                print(f"{folder_name:<40} | {icbhi:<10.4f} | {epoch}")
        except:
            continue


if __name__ == "__main__":
    find_all_results()