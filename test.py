import torch
from torch.utils.data import DataLoader
from sm_dataset import SM_Dataset
from lstm_machine import Machine
from sklearn.metrics import r2_score


def test():
    BATCH_SIZE = 2000
    dataset = SM_Dataset(is_train=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = Machine()
    model.load_state_dict(torch.load("model.h5"))
    model.eval()
    print(f"Test started ...")
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            print(r2_score(y_true, y_pred))

if __name__ == "__main__":
    test()
