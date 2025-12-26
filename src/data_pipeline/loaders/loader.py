






from psutil import num
def get_dataloader(config) -> DataLoader:
    """
    返回一个DataLoader，用于加载数据集。
    """
    train_dataset = datasets
    test_dataset = 
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)
    return train_loader, test_loader
