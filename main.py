# modules

import spectral_analysis as sa
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from math import floor
from utilities import evolve_pt, np_to_complex_pt, evolve_np, plot_dataset
from test import test
from dataset import Dataset
from torch.utils.data import DataLoader #Dataloader module
from test import create_test_pulse
import torchvision.transforms as transforms  # Transformations and augmentations
from dataset import Dataset_train
from dataset_generator import Generator
import utilities
import argparse
import wandb
import shutil
import warnings


def main(_learning_rate,
         _epoch_num,
         _batch_size,
         _plot_freq,
         _dataset_size,
         _generate,
         _cpu,
         _node_number,
         _net_architecture,
         _criterion,
         _optimalizer,
         _test_signal,
         _weight_decay):
    #hyperparameters
    print('\n',
          'learning_rate:', _learning_rate,'\n',
          'epoch_number:', _epoch_num,'\n',
          'batch_size:', _batch_size,'\n',
          'plot_frequency:', _plot_freq,'\n',
          'dataset_size:', _dataset_size,'\n',
          'generate:', _generate,'\n',
          'node_number:', _node_number, '\n',
          'architecture:', _net_architecture, '\n',
          'criterion:', _criterion, '\n',
          'optimalizer:', _optimalizer, '\n',
          'test_signal:', _test_signal, '\n',
          'weight_decay:', _weight_decay, '\n')
    
    
    ### Chose architecture 
    if _net_architecture == 'network_1':
        from nets import network_1 as network
    if _net_architecture == 'network_2':
        from nets import network_2 as network
    if _net_architecture == 'network_3':
        from nets import network_3 as network
    if _net_architecture == 'network_4':
        from nets import network_4 as network
    if _net_architecture == 'network_5':
        from nets import network_5 as network
    if _net_architecture == 'network_6':
        from nets import network_6 as network
    if _net_architecture == 'network_7':
        from nets import network_7 as network
    if _net_architecture == 'network_8':
        from nets import network_8 as network
    if _net_architecture == 'network_9':
        from nets import network_9 as network
    if _net_architecture == 'network_11':
        from nets import network_11 as network

    ### Chose device, disclaimer! on cpu network will not run due to batch normalization

    if _cpu:
        print('Forced cpu')
        my_device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            my_device = torch.device("cuda")
            print (f"Using {my_device}")
            print('allocated CUDA memory: ',torch.cuda.memory_allocated())      # Checking GPU RAM allocated memory
            print('cached CUDA memory: ',torch.cuda.memory_reserved())
            torch.cuda.empty_cache()                                            # clear CUDA memory
            torch.backends.cudnn.benchmark = True                               # let cuda chose the most efficient way of calculating Convolutions
        elif torch.backends.mps.is_available():
            print ("CUDA device not found.")
            my_device = torch.device("mps")
            print (f"Using {my_device}")
        else:
            print ("MPS device not found.")
            my_device = torch.device("cpu")
            print (f"Using {my_device}")

    # data type

    my_dtype = torch.float32

    # initial pulse (that is transformed by some phase)

    input_dim = 5000 # number of points in a single pulse
    zeroes_num = 5000

    bandwidth = [190, 196]
    centre = 193
    FWHM = 0.4

    # this is our input for the net
    initial_pulse = sa.hermitian_pulse(pol_num = 0, # 0 for gauss signal
                                    bandwidth = bandwidth,
                                    centre = centre,
                                    FWHM = FWHM,
                                    num = input_dim)
    
    # this serves only to generate FT pulse
    long_pulse = initial_pulse.zero_padding(length = zeroes_num, inplace = False)
    long_pulse_2 = long_pulse.copy()    
    Y_initial = initial_pulse.Y.copy()

    # we want to find what is the bandwidth of intensity after FT, to estimate output dimension of NN

    long_pulse.fourier()
    x_start = long_pulse.quantile(0.001)
    x_end = long_pulse.quantile(0.999)
    idx_start = np.searchsorted(long_pulse.X, x_start)
    idx_end = np.searchsorted(long_pulse.X, x_end)
    output_dim = idx_end - idx_start    # number of points of non-zero FT-intensity
    if output_dim % 2 == 1:
        output_dim += 1

    print("\ninput_dim (spectrum length) = {}".format(input_dim))  
    print("output_dim (phase length) = {}".format(output_dim))

    # stuff for plots

    pulse_ft = long_pulse_2.copy()
    pulse_ft.fourier()
    pulse_ft.X = np.real(pulse_ft.X)
    pulse_ft.Y = np.abs(pulse_ft.Y)
    pulse_ft.cut(inplace = True, start = idx_start, end = idx_end, how = "index")    

    # test pulse

    test_pulse, test_phase = create_test_pulse(_test_signal, initial_pulse, output_dim, my_device, my_dtype)

    # create dataset and wrap it into dataloader

    if _generate:
        print("\nCreating training set...")
        
        the_generator = Generator(data_num = _dataset_size,
                                initial_intensity = Y_initial,
                                phase_len = output_dim,
                                device = my_device,
                                dtype = np.float32
                                )

        the_generator.generate_and_save()
        print("Successfully created training set containing {} spectra.\n".format(len(os.listdir('data/train_intensity'))))

        plot_dataset(100, pulse = initial_pulse, ft_pulse = pulse_ft)

        exit()

    # recreate pics folder if exists and create it otherwise

    if os.path.isdir("pics"):
        shutil.rmtree('pics') # clear pictures folder
        os.mkdir("pics")
    else:
        os.mkdir("pics")

    # create NN

    model = network(input_size = input_dim, 
                n = _node_number, 
                output_size = output_dim)
    model.to(device = my_device, dtype = my_dtype)

    # choose optimizer

    if _optimalizer =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = _learning_rate, weight_decay=_weight_decay)
    if _optimalizer =='NAdam':
        optimizer = torch.optim.NAdam(model.parameters(), lr = _learning_rate)
    if _optimalizer =='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = _learning_rate)
    if _optimalizer =='RSMprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = _learning_rate)
    
    # choose loss function
    
    if _criterion =='MSE':
        criterion = torch.nn.MSELoss()
    if _criterion =='L1':
        criterion = torch.nn.L1Loss()
    
    dataset_train = Dataset_train(root='', transform=True, device = my_device)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=_batch_size, num_workers=0, shuffle=True)
    
    # learning loop

    loss_list = []
    wandb.watch(model, criterion, log="all", log_freq=400)

    for epoch in tqdm(range(_epoch_num)):
        for pulse, _ in dataloader_train:
            # pulse = pulse.to(my_device) # the pulse is already created on device by dataset, uncomment if not using designated dataset for this problem
            
            # predict phase that will transform gauss into this pulse
            predicted_phase = model(pulse)

            # transform gauss into something using this phase
            initial_intensity = np_to_complex_pt(long_pulse_2.Y.copy(), device = my_device, dtype = my_dtype)
            reconstructed_intensity = evolve_pt(initial_intensity, predicted_phase, device = my_device, dtype = my_dtype)

            # a bit of calculus
            loss = criterion(reconstructed_intensity.abs()[:,zeroes_num: input_dim + zeroes_num], pulse) # pulse intensity
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # stats
            _loss = loss.clone().cpu().detach().numpy()
            wandb.log({"loss": _loss}) #log loss to wandb
            loss_list.append(_loss)

        if epoch%_plot_freq == 0: # plot and test model
            model.eval()

            print("Epoch no. {}. Loss {}.".format(epoch, np.mean(np.array(loss_list[epoch*len(dataloader_train): (epoch+1)*len(dataloader_train)]))))
            fig, test_loss=test(model = model,
                    test_pulse = test_pulse,
                    test_phase = test_phase,
                    initial_pulse = long_pulse_2.copy(),
                    device = my_device, 
                    dtype = my_dtype,
                    iter_num = epoch)
            
            wandb.log({"chart": fig})
            print('test_loss',test_loss)
            wandb.log({"test_loss": test_loss})
            fig.close()
            model.train()

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning) # ignore warnings from plotly
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default=1e-5, type=float)
    parser.add_argument('-en', '--epoch_num', default=10, type=int)
    parser.add_argument('-bs', '--batch_size', default=50, type=int)
    parser.add_argument('-pf', '--plot_freq', default=3, type=int)
    parser.add_argument('-ds', '--dataset_size', default=10000, type=int)
    parser.add_argument('-g', '--generate', action='store_true') # only generate, training will not run, wandb will be offline
    parser.add_argument('-fc', '--force_cpu', action='store_true')
    parser.add_argument('-tr', '--test_run', action='store_true')
    parser.add_argument('-nn', '--node_number', default=100, type=int)
    parser.add_argument('-ar', '--architecture', default='network_1', type=str)
    parser.add_argument('-cr', '--criterion', default='MSE', type=str)
    parser.add_argument('-op', '--optimalizer', default='Adam', type=str,)
    parser.add_argument('-ts', '--test_signal', default='hermite', type=str,)
    parser.add_argument('-wd', '--weight_decay', default=0, type=float)
    args = parser.parse_args()
    config={}
    
    ### WANDB config
    # start a new wandb run to track this script
    
    # for forced offline work -tf
    if args.test_run or args.generate:
        print('WANDB WORKING OFFLINE')
        wandb.init(mode="disabled") # for offline work
    else:
        wandb.init(
        # set the wandb project where this run will be logged
        project = "platypus",
        entity = "zps_qpl_ml",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.learning_rate,
        "epochs": args.epoch_num,
        "batch_size": args.batch_size,
        'dataset_size': args.dataset_size,
        "architecture": args.architecture,
        "dataset": "defalut",
        "node_number": args.node_number,
        "test_signal": args.test_signal,
        "weight_decay": args.weight_decay
        }
        )
    
    main(args.learning_rate,
         args.epoch_num,
         args.batch_size,
         args.plot_freq,
         args.dataset_size,
         args.generate,
         args.force_cpu,
         args.node_number,
         args.architecture,
         args.criterion,
         args.optimalizer,
         args.test_signal,
         args.weight_decay)
