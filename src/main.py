from dataset import TimeSeriesDataset
import pandas as pd
import torch
from utils import UtilFunction
import tqdm
import torch.nn as nn
import torch.optim as optim
from models import TransformerAE


def train_AE(model, train_loader, optimizer, criterion, n_epochs, device):
    train_losses = []
    best_model = {
        "loss": float('inf'),
        "state": None,
        "epoch": 0
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch") as t:
            for batch in t:
                inputs = batch["input"].to(device)
                original_hidden, reconstructed_hidden = model(inputs) # [ Batch_size, HIDDEN_DIM_LSTM ]

                loss = criterion(reconstructed_hidden, original_hidden)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                t.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        print(f"Epoch {epoch + 1}/{n_epochs}, Average Train Loss: {avg_epoch_loss:.8f}")
        
        if avg_epoch_loss < best_model["loss"]:
            best_model["state"] = model.state_dict()
            best_model["loss"] = avg_epoch_loss
            best_model["epoch"] = epoch + 1

    return train_losses, best_model

config = UtilFunction.load_config('./src/config.yaml')
CFG = config['CFG']

df_A = pd.read_csv("./open/train/TRAIN_A.csv")
df_B = pd.read_csv("./open/train/TRAIN_B.csv")

train_dataset_A = TimeSeriesDataset(df_A, stride=60)
train_dataset_B = TimeSeriesDataset(df_B, stride=60)
train_dataset_A_B = torch.utils.data.ConcatDataset([train_dataset_A, train_dataset_B])

train_loader = torch.utils.data.DataLoader(train_dataset_A_B, 
                                            batch_size=CFG['BATCH_SIZE'], 
                                            shuffle=True)

model = TransformerAE(CFG['HIDDEN_DIM_TRANSFORMER'],CFG['NUM_HEADS'],CFG['DROPOUT'],CFG['NUM_LAYERS']).cuda()

# Optimizer 및 Loss Function 설정
optimizer = optim.Adam(model.parameters(), lr=CFG['  LEARNING_RATE'])
criterion = nn.MSELoss()

if __name__ == '__main__':
    train_AE(model,train_loader,optimizer,criterion,CFG['EPOCHS'],CFG['DEVICE'])