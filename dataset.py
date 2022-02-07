import torch as tc


class SequenceDataset(tc.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.n_seq_per_env = data.shape[0]
        self.n_env = data.shape[1]
        self.seq_len = data.shape[-1]

    def __get_item__(self, idx_item, idx_env):
        return self.data[idx_item, idx_env]

    def __len__(self):
        return len(self.data)

