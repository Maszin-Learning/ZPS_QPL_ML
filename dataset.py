from dataset_generator import Generator
from torch.utils.data import Dataset
import os
import numpy as np
import torch

    
class Dataset_train(Dataset):
    def __init__(self, root ,transform):
        self.labels = []
        self.dirName_1 = os.path.join('data','train_intensity') # path to training data with intensity of spectrum
        self.dirName_2 = os.path.join('data','train_phase') # path to training data with pahse used to transformate gause of spectrum
        self.files_names_intensity = []
        self.files_names_phase = []
        
        self.transform = transform
        self.load_signals()
    

    def load_signals(self):

        # create list of paths to images
        self.files_names_intensity = [os.path.join(self.dirName_1, f) for f in os.listdir(self.dirName_1) if os.path.isfile(os.path.join(self.dirName_1, f))]
        self.files_names_phase = [os.path.join(self.dirName_2, f) for f in os.listdir(self.dirName_2) if os.path.isfile(os.path.join(self.dirName_2, f))]
        


    def __len__(self):
        return len(self.files_names_intensity)

    def __getitem__(self, index):
        intensity = np.loadtxt(self.files_names_intensity[index],
                 delimiter=" ", dtype=np.float32)
        phase = np.loadtxt(self.files_names_phase[index],
                 delimiter=" ", dtype=np.float32)
        
        if self.transform:
            intensity = torch.tensor(intensity, requires_grad=True)
            phase = torch.tensor(phase, requires_grad=True)

        return (intensity, phase)