import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.io import arff

class LoadDataset(Dataset):

    def __init__(self, dataset='CBF', pack='TRAIN'):
        
        filepath = f'{dataset}/{dataset}_{pack}.arff'
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        df['target'] = [int(str(x).split("'")[1]) for x in df['target']]
        self.signals = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx].unsqueeze(0)
        label = self.labels[idx]
        return signal, label

