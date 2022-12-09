from cnn_1d_machine import CNN1DMachine
from lstm_machine import LSTMMachine
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from sm_dataset import SM_Dataset

def train():
    IS_LSTM = False
    NUM_EPOCHS = 200
    BATCH_SIZE = 500

    dataset = SM_Dataset(is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = CNN1DMachine()
    if IS_LSTM:
        model = LSTMMachine()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss(reduction='mean')

    for epoch in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            optimizer.zero_grad()
            y_pred = model(data)
            y_pred = y_pred.reshape(-1)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    print("Training done. Machine saved to models/saha.h5")
    torch.save(model.state_dict(), 'model.h5')
    return model


if __name__ == "__main__":
    train()
