from torch.utils.data import DataLoader

from .dataset import ICBHIASTTokenDataset


def build_dataloader(
    csv_path,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False
):
    """
    构建 ICBHI AST tokens 的 DataLoader。

    Args:
        csv_path: train_sub.csv / val.csv / test_index.csv 路径
        batch_size: 每个 batch 的样本数
        shuffle: 是否打乱数据
        num_workers: 数据读取进程数
        pin_memory: GPU 训练时建议 True
        drop_last: 是否丢弃最后一个不完整 batch

    Returns:
        DataLoader
    """

    dataset = ICBHIASTTokenDataset(csv_path=csv_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return loader


def build_train_val_loaders(
    train_csv,
    val_csv,
    batch_size=32,
    num_workers=2
):
    """
    构建训练集和验证集 DataLoader。
    """

    train_loader = build_dataloader(
        csv_path=train_csv,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = build_dataloader(
        csv_path=val_csv,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader