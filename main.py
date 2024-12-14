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
import utilities
from dataset import Dataset
from torch.utils.data import DataLoader #Dataloader module
import torchaudio
from test import create_target_pulse, test, create_initial_pulse
import torchvision.transforms as transforms  # Transformations and augmentations
from dataset import Dataset_train
from dataset_generator import Generator
import utilities as u
import argparse
import wandb
import shutil
import warnings
from utilities import MSEsmooth, MSEsmooth2, MSElowpass

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
         _target_signal,
         _initial_signal,
         _weight_decay,
         _axis_type):
    
    # hyperparameters

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
          'target_signal:', _target_signal, '\n',
          'initial_signal:', _initial_signal, '\n',
          'weight_decay:', _weight_decay, '\n',
          'axis_type:', _axis_type, '\n')
    
    # Choose architecture 

    if _net_architecture == 'network_0':
        from nets import network_0 as network #DEVELOPMENT ARCHITECTURE

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
    if _net_architecture == 'network_12':
        from nets import network_12 as network
    if _net_architecture == 'network_UNET_1D':
        from nets import UNET_1D as network  

    # Choose device, disclaimer! on cpu network will not run due to batch normalization

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
    
    #clear data folders
    utilities.clear_folder('saved_models')
    utilities.clear_folder('pics')
        
    model_save_PATH_dir='saved_models'

    # initial pulse (that is to be transformed by some phase)

    input_dim = 5000    # number of points in a single pulse
    zeroes_num = 5000   # number of zeroes we add on the left and on the right of the main pulse (to make FT intensity broader)

    bandwidth = [0, input_dim*11] # 11 ps is the resolution of EOPM
    centre_init = 500       # not used if initial signal is exponential
    width_init = 100        # not used if initial signal is exponential

    centre_target = 500     # centre of the target pulse defined in dataset_generator -> pulse_gen
    width_target = 200      # centre of the target pulse defined in dataset_generator -> pulse_gen

    convolution_width = 0.1   # width of the gaussian convolved with the main signal

    initial_pulse = create_initial_pulse(bandwidth = bandwidth,
                                         centre = centre_init,
                                         FWHM = width_init,
                                         num = input_dim,
                                         pulse_type = _initial_signal)
    
    # additional pulse to add to exp (gauss) so it makes it more physical

    signal_correction = create_initial_pulse(bandwidth = bandwidth,
                                         centre = bandwidth[0]/2 + bandwidth[1]/2,
                                         FWHM = convolution_width,
                                         num = input_dim,
                                         pulse_type = 'gauss')
    
    initial_pulse.Y = np.convolve(initial_pulse.Y, signal_correction.Y, mode='same')
    Y_initial = initial_pulse.Y.copy()

    # normalize it in L2

    initial_pulse.Y = initial_pulse.Y / np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))

    # this serves only to generate FT pulse

    long_pulse = initial_pulse.zero_padding(length = zeroes_num, inplace = False) 

    # we want to find what is the bandwidth of intensity after FT, to estimate output dimension of NN

    trash_fraction = 0.001 # percent of FT transformed to be cut off - it will contribute to the noise

    long_pulse_ft = long_pulse.inv_fourier(inplace = False)
    fwhm_init_F = u.comp_FWHM(u.comp_std(initial_pulse.inv_fourier(inplace = False).X, initial_pulse.inv_fourier(inplace = False).Y))
    x_start = long_pulse_ft.quantile(trash_fraction/2, norm = "L2")
    x_end = long_pulse_ft.quantile(1-trash_fraction/2, norm = "L2")
    idx_start = np.searchsorted(long_pulse_ft.X, x_start)
    idx_end = np.searchsorted(long_pulse_ft.X, x_end)
    if (idx_end - idx_start) % 2 == 1:
        idx_end += 1
    output_dim = idx_end - idx_start    # number of points of non-zero FT-intensity

    print("\ninput_dim (spectrum length) = {}".format(input_dim))  
    print("output_dim (phase length) = {}".format(output_dim))

    # stuff for plots

    pulse_ft = long_pulse.copy()
    pulse_ft.inv_fourier()
    pulse_ft.X = np.real(pulse_ft.X)
    pulse_ft.Y = np.abs(pulse_ft.Y)
    pulse_ft.cut(inplace = True, start = idx_start, end = idx_end, how = "index")    

    # generate training data

    if _generate:
        print("\nCreating training set...")
        
        the_generator = Generator(data_num = _dataset_size,
                                initial_intensity = Y_initial,
                                FT_X = pulse_ft.X,
                                phase_len = output_dim,
                                device = my_device,
                                dtype = np.float32,
                                target_type = _target_signal,
                                target_metadata = [centre_target, width_target, bandwidth[0], bandwidth[1]] #czemu do cholery jak zmienie na center i width to przestaje się uczyć XDDD
                                )

        the_generator.generate_and_save()
        print("Successfully created training set containing {} spectra.\n".format(len(os.listdir('data/train_intensity'))))

        #u.plot_dataset(_batch_size, pulse = initial_pulse, ft_pulse = pulse_ft)

        print("Calculating mean Time-Bandwidth Product of the training set...")
        TBP_mean, TBP_std = u.comp_mean_TBP(initial_pulse.X, fwhm_init_F)
        print("Mean TBP of the transformation from initial pulse to a spectrum from dataset is equal to {} +- {}.\n".format(round(TBP_mean, 5), round(TBP_std, 5)))   

        exit()

    # create dataset and dataloader
    
    dataset_train = Dataset_train(root='', transform=True, device = my_device)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=_batch_size, num_workers=0, shuffle=True)

    # target pulse

    target_pulse = dataset_train[0]
    target_pulse = target_pulse*(1 + trash_fraction)
    fwhm_target = u.comp_FWHM(u.comp_std(initial_pulse.X.copy(), target_pulse.clone().detach().cpu().numpy().ravel()))
    print("\nTime-bandwidth product of the transformation from the initial pulse to the target pulse is equal to {}.\n".format(round(fwhm_target*fwhm_init_F/2, 5)))   # WARNING: This "/2" is just empirical correction
    if fwhm_target*fwhm_init_F/2 < 0.44:
        print("TRANSFORMATION IMPOSSIBLE\n")

    # create NN

    model = network(input_size = input_dim, 
                n = _node_number, 
                output_size = output_dim)
    model.to(device = my_device, dtype = my_dtype)
    

    print("Model parameters: {}\n".format(utilities.count_parameters(model)))

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

    filter_threshold = 0.15 # write 1 if you don't want to filter out anything
    filter_mask = u.gen_filter_mask(threshold = filter_threshold, num = output_dim, device = my_device)

    
    if _criterion =='MSE':
        criterion = torch.nn.MSELoss()
    if _criterion =='L1':
        criterion = torch.nn.L1Loss()
    if _criterion =='MSEsmooth':
        criterion = MSEsmooth(device = my_device, dtype = my_dtype, c_factor = 0.6)
    if _criterion =='MSEsmooth2':
        criterion = MSEsmooth2(device = my_device, dtype = my_dtype, c_factor = 0.5, s_factor = 0.5)
    if _criterion =='MSElowpass':
        criterion = MSElowpass(device = my_device, dtype = my_dtype, penalty_strength = 1, filter_mask = filter_mask)
    
    # prepare initial pulses (with nontrivial phases)

    initial_intensity = long_pulse.copy()
    initial_phase = np.zeros(len(long_pulse))      # HERE YOU CAN ADD WHATEVER YOU WANT
    initial_intensity.Y = initial_intensity.Y*np.exp(1j*initial_phase)
    initial_intensity_pt = u.np_to_complex_pt(initial_intensity.Y, device = my_device, dtype = my_dtype)

    # learning loop

    loss_list = []
    test_loss_global = 10000
    wandb.watch(model, criterion, log="all", log_freq=400)

    for epoch in range(_epoch_num):
        for pulse, _ in tqdm(dataloader_train):

            # pulse = pulse.to(my_device) # the pulse is already created on device by dataset, uncomment if not using designated dataset for this problem
            predicted_phase = model(pulse)

            # transform gauss into something using this phase
            reconstructed_intensity = u.evolve_pt(initial_intensity_pt, predicted_phase, device = my_device, dtype = my_dtype)

            # calculating back-propagation
            if _criterion in ["MSEsmooth", "MSEsmooth2", "MSElowpass"]:
                loss = criterion((predicted_phase, reconstructed_intensity.abs()[:,zeroes_num: input_dim + zeroes_num]), pulse)
            else:
                loss = criterion(reconstructed_intensity.abs()[:,zeroes_num: input_dim + zeroes_num], pulse) # pulse intensity
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # stats
            _loss = loss.clone().cpu().detach().numpy()
            wandb.log({"loss": _loss}) # log loss to wandb
            loss_list.append(_loss)

        if epoch%_plot_freq == 0: # plot and test model
            model.eval()

            print("Epoch no. {}. Loss {}.".format(epoch, np.mean(np.array(loss_list[epoch*len(dataloader_train): (epoch+1)*len(dataloader_train)]))))

            fig, test_loss = test(model = model,
                    target_pulse = target_pulse,
                    initial_pulse = initial_intensity,
                    device = my_device, 
                    dtype = my_dtype,
                    iter_num = epoch,
                    save = True,
                    x_type = _axis_type,
                    filter_threshold = filter_threshold)
            
            cont_penalty = torch.sqrt(torch.sum(torch.square(u.diff_pt(u.unwrap(predicted_phase), device = my_device, dtype = my_dtype))))
            print("phase's variation MSE: {}.".format(cont_penalty))

            if test_loss < test_loss_global:
                # shutil.rmtree(model_save_PATH_dir)
                utilities.clear_folder('saved_models')
                torch.save(model.state_dict(), os.path.join(model_save_PATH_dir, f'{_net_architecture}_ep{epoch}.pt'))
            test_loss_global = test_loss
            wandb.log({"chart": fig})
            print('test_loss',test_loss)
            wandb.log({"test_loss": test_loss})
            fig.close()

            model.train()

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning) # ignore warnings from plotly
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default=5e-4, type=float) # designed for network_1
    parser.add_argument('-en', '--epoch_num', default=10, type=int)
    parser.add_argument('-bs', '--batch_size', default=2, type=int)
    parser.add_argument('-pf', '--plot_freq', default=3, type=int)
    parser.add_argument('-ds', '--dataset_size', default=100, type=int)
    parser.add_argument('-g', '--generate', action='store_true') # only generate, training will not run, wandb will be offline
    parser.add_argument('-fc', '--force_cpu', action='store_true')
    parser.add_argument('-tr', '--test_run', action='store_true')
    parser.add_argument('-nn', '--node_number', default=100, type=int)
    parser.add_argument('-ar', '--architecture', default='network_0', type=str)
    parser.add_argument('-cr', '--criterion', default='MSEsmooth', type=str)
    parser.add_argument('-op', '--optimalizer', default='Adam', type=str,)
    parser.add_argument('-ts', '--target_signal', default='gauss', type=str,)
    parser.add_argument('-is', '--initial_signal', default='exponential', type=str,)
    parser.add_argument('-wd', '--weight_decay', default=0, type=float)
    parser.add_argument('-ax', '--axis_type', default="freq", type=str)
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
        "target_signal": args.target_signal,
        "initial_signal": args.initial_signal,
        "criterion": args.criterion,
        "optimizer": args.optimalizer,
        "weight_decay": args.weight_decay,
        "axis_type": args.axis_type
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
         args.target_signal,
         args.initial_signal,
         args.weight_decay,
         args.axis_type)
