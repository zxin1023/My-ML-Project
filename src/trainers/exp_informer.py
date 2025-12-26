import os
import optim


from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml
import wandb

import argparse
from utils.config import load_config


def trainer(model, optimizer, criterion, dataset, config):
    model = Informer()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(config['train']['epochs']):
        train_loss = 0.0
        val_acc = evaluate(model, val_loader)
        
        # wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_acc': val_acc
        })
        
        
        
        model.train()
        for batch in tqdm(DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)):
            x, y = batch
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
        # 保存模型
        if epoch % config['train']['save_every'] == 0:
            torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], f'model_{epoch}.pth'))
            wandb.save(f"{config['train']['save_dir']}/model_{epoch}.pth")
        
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        config=config
    )




 