import torch
import tqdm

class Trainer():
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