import time

import numpy as np
import torch
from torch.utils.data import Dataset

from nlp.dataset_utils import load_dataset
from util import format_time


class EmojiPredictionDataset(Dataset):
    def __init__(self, dataset, transform=None):
        if isinstance(dataset, str):
            self.dataset_dict = load_dataset(base_path)
        else:
            self.dataset_dict = dataset
        self.transform = transform
    

    def __len__(self):
        return len(self.dataset_dict["label_list"])
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # print("idx:", idx)
        # print("type(idx):", type(idx))
        # print("type(idx[0]):", type(idx[0]))
        
        text_array = np.array(self.dataset_dict["text_list"][idx])
        label_array = np.array(self.dataset_dict["label_list"][idx], dtype=int)
        sample = { 'x': text_array, 'y': label_array }

        if self.transform:
            sample = self.transform(sample)

        return sample
