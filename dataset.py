from dataset_generator import Generator
from torch.utils.data import Dataset
class Dataset(Dataset):

    def __init__(self, batch_num, batch_size, initial_intensity, phase_len, device, dtype, max_order=10, max_value=10):
        self.generator = Generator(batch_num, batch_size, initial_intensity, phase_len, device, dtype, max_order=10, max_value=10)
        self.num = batch_num*batch_size
        self.init_intensity = initial_intensity
    def __len__(self):
        return self.num
    def __getitem__(self, idx):
        return self.generator.data[idx]