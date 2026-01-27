import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn
import torch.optim as optim

from models.residual_nn import ResidualBlock, MLP

num_epochs = 100
batch_size = 2048
learning_rate = 1e-4
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def train_model(
        train_loader,
        val_loader,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.05,
        emb_dim=16,
        emb_num=127,
        num_epochs=100,
        learning_rate=1e-4,
        device=device):
    ## defining parameters
    in_dim = train_loader.dataset[0][0].shape[-1]
    model = MLP(in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=dropout,
                emb_dim=emb_dim,
                emb_num=emb_num).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = MLP.criterion

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=1e-4
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, stock_id, y in train_loader:
            x, stock_id, y = x.to(device), stock_id.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, stock_id)
            loss = criterion(output, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        for x, stock_id, y in val_loader:
            x, stock_id, y = x.to(device), stock_id.to(device), y.to(device)
            with torch.no_grad():
                output = model(x, stock_id)
                loss = criterion(output, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        print(f'Epoch: {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss}')

        if current_lr := scheduler.get_last_lr()[0] < 1e-7:
            print(f'LR = {current_lr} is too small! Early stopping the training.')
            break

    return model


if __name__ == '__main__':
    df = pd.read_csv('../baseline4.csv', index_col=False).iloc[:, 1:]
    X = df.drop(columns=['time_id', 'target', 'stock_id']) * 10
    stock = df.loc[:, ['stock_id']]
    y = df['target'] * 10

    X = X.to_numpy()
    stock = stock.to_numpy()
    y = y.to_numpy()

    X = torch.tensor(X, dtype=torch.float32)
    stock = torch.tensor(stock, dtype=torch.int).squeeze(-1)
    y = torch.tensor(y, dtype=torch.float32)

    from sklearn.model_selection import train_test_split

    X_train, X_test, stock_train, stock_test, y_train, y_test = train_test_split(X, stock, y, test_size=0.3,
                                                                                 shuffle=True, random_state=1)
    X_test, X_val, stock_test, stock_val, y_test, y_val = train_test_split(X_test, stock_test, y_test, test_size=0.5,
                                                                           shuffle=True, random_state=1)

    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(X_train, stock_train, y_train)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, pin_memory=True)

    val_dataset = TensorDataset(X_val, stock_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=2048, pin_memory=True)

    model = train_model(dataloader, val_dataloader, learning_rate=1e-3)

    model.eval()
    with torch.no_grad():
        model = model.to('cpu')
        test_pred = model(X_test, stock_test)
        test_loss = MLP.criterion(test_pred, y_test).item()
    print(test_loss)
    torch.save(model.state_dict(), f'first_model_loss_{test_loss:.4f}.pth')









