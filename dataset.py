from dataset_generator import Generator
from torch.utils.data import Dataset
class Dataset(Dataset):

    def __init__(self, num, max_order=10, max_value=None):
        self.generator = Generator(num, max_order)
        self.num = num
    def __len__(self):
        return self.num
    def __getitem__(self, idx):
        return self.generator.phase_gen()