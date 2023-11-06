from dataset_generator import Generator
import spectral_analysis as sa
import torch
import numpy as np
import torchvision.transforms as transforms  # Transformations and augmentations
from dataset import Dataset_train


# data type
my_device = torch.device('cpu')
my_dtype = np.float32

pulse = sa.hermitian_pulse(0, [190, 196], 193, 1)
intensity = pulse.Y.copy()
the_generator = Generator(data_num = 10,
                          initial_intensity = intensity,
                          phase_len = 56,
                          device = my_device,
                          dtype = my_dtype
                          )

the_generator.generate_and_save()

_transform = transforms.Compose([transforms.ToTensor()])
dataset_train = Dataset_train(root='', transform=_transform)