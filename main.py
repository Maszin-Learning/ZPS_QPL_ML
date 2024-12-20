# modules

print("Loading modules...")

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
import loss_functions as lf
from loss_functions import MSEsmooth, MSEsmooth2, MSElowpass, MSEdouble

print("Modules loaded successfully!")

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

    # metaparameters

    bandwidth = [-2500, 2500]   # (ps)

    spectral_phase_len = 40     # so, assuming 1.5 GHz of pulse shaper's resolution, we get 60 GHz of bandwidth
    temporal_phase_len = 200    # so, assuming 11 ps of modulator's resolution, we get 2200 ps of bandwidth

    comp_time_resolution = 1        # (ps) to avoid border effects we compute with higher resolution than the one of the modulator's
    comp_freq_resolution = 0.0001   # (THz) as above

    time_num = floor((bandwidth[1]-bandwidth[0])/comp_time_resolution)             # number of points in the initial pulse

    centre_init = 500           # not used if initial signal is exponential
    width_init = 100            # not used if initial signal is exponential

    centre_target = 0           # (ps) centre of the target pulse defined in dataset_generator -> pulse_gen
    width_target = 200          # (ps) FWHM of the target pulse defined in dataset_generator -> pulse_gen

    convolution_width = 0.1   # width of the gaussian convolved with the main signal

    # initial pulse

    initial_pulse = create_initial_pulse(bandwidth = bandwidth,
                                         centre = centre_init,
                                         FWHM = width_init,
                                         num = time_num,
                                         pulse_type = _initial_signal)
    
    # additional pulse to add to exp (gauss) so it makes it more physical

    signal_correction = create_initial_pulse(bandwidth = bandwidth,
                                         centre = bandwidth[0]/2 + bandwidth[1]/2,
                                         FWHM = convolution_width,
                                         num = time_num,
                                         pulse_type = 'gauss')
    
    initial_pulse.Y = np.convolve(initial_pulse.Y, signal_correction.Y, mode='same')
    initial_pulse.Y = initial_pulse.Y / np.sum(initial_pulse.Y)

    Y_initial = initial_pulse.Y.copy()
    initial_intensity_pt = u.np_to_complex_pt(initial_pulse.Y, device = my_device, dtype = my_dtype)

    # we compute resolution stuff to be able later to match phase and intensity

    initial_pulse_FT = initial_pulse.inv_fourier(inplace = False)
    init_freq_resolution = initial_pulse_FT.calc_spacing()
    increase_freq_res = init_freq_resolution/comp_freq_resolution # at the beginning, we dont control the frequency resolution and later we will want to increase it to given level

    temp_idx_start = np.searchsorted(initial_pulse.X, initial_pulse.quantile(1e-5, "L1")-10) # extra 10 ps just to be sure; from this index we start multiplication of phase

    # generate training data

    if _generate:
        print("\nCreating training set...")
        
        the_generator = Generator(data_num = _dataset_size,
                                initial_intensity = Y_initial,
                                FT_X = None,        # this feature isn't called anymore
                                phase_len = None,   # this feature isn't called anymore
                                device = my_device,
                                dtype = np.float32,
                                target_type = _target_signal,
                                target_metadata = [centre_target, width_target, bandwidth[0], bandwidth[1]]
                                )

        the_generator.generate_and_save()
        print("Successfully created training set containing {} spectra.\n".format(len(os.listdir('data/train_intensity'))))

        exit()

    # create dataset and dataloader
    
    dataset_train = Dataset_train(root='', transform=True, device = my_device)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=_batch_size, num_workers=0, shuffle=True)
 
    # create NN

    model = network(input_size = time_num, 
                n = _node_number, 
                spectral_phase_len = spectral_phase_len,
                temporal_phase_len = temporal_phase_len)
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

    #filter_threshold = 0.15 # write 1 if you don't want to filter out anything
    #filter_mask = lf.gen_filter_mask(threshold = filter_threshold, num = None, device = my_device)   # what is the num?
    
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
    if _criterion =='MSEdouble':
        criterion = MSEdouble(device = my_device, dtype = my_dtype)

    # prepare targets

    temp_intens_target = dataset_train[0]
    temp_intens_target = torch.tensor(temp_intens_target, requires_grad = False, device = my_device, dtype = my_dtype)  # well, it was a tensor even before, but now we know its properties
    temp_intens_target = temp_intens_target/np.sum(temp_intens_target.clone().detach().cpu().numpy())

    spectr_intens_target = u.fourier(temp_intens_target)
    spectr_intens_target = u.cut(spectr_intens_target, 50/init_freq_resolution) # we leave central 50 GHz, we delete the rest in order to save GPU
    spectr_intens_target = u.increase_resolution(spectr_intens_target, increase_freq_res, device = my_device, dtype = my_dtype)
    
    # learning loop

    loss_list = []
    test_loss_global = 10000
    wandb.watch(model, criterion, log="all", log_freq=400)

    for epoch in range(_epoch_num):
        for pulse, _ in tqdm(dataloader_train):

            temp_phase_pred, spectr_phase_pred = model(pulse)

            # we apply temporal phase
            temp_phase_pred = u.increase_resolution(temp_phase_pred, 11/comp_time_resolution, device = my_device, dtype = my_dtype) # 11 ps is the resolution of EOPM
            temp_intens_pred = u.multiply_by_phase(initial_intensity_pt, temp_phase_pred, index_start = temp_idx_start, device = my_device, dtype = my_dtype)

            # we apply spectral phase
            spectr_intens_pred = u.fourier(temp_intens_pred)
            spectr_intens_pred = u.cut(spectr_intens_pred, 50/init_freq_resolution) # we leave central 50 GHz, we delete the rest in order to save GPU
            spectr_intens_pred = u.increase_resolution(spectr_intens_pred, increase_freq_res, device = my_device, dtype = my_dtype)
            spectr_phase_pred = u.increase_resolution(spectr_phase_pred, 0.0015/comp_freq_resolution, device = my_device, dtype = my_dtype)  # 1.5 GHz is the resolution of the pulse shaper
            spectr_intens_pred = u.multiply_by_phase(spectr_intens_pred, spectr_phase_pred, index_start = floor((spectr_intens_pred.shape[-1]-spectr_phase_pred.shape[-1])/2), device = my_device, dtype = my_dtype)

            # and back to time domain
            print(np.sum(spectr_intens_pred.clone().abs().cpu().detach().numpy()))
            temp_intens_pred = u.inv_fourier(spectr_intens_pred)
            print(np.sum(temp_intens_pred.clone().abs().cpu().detach().numpy()))
            temp_intens_pred = u.cut(temp_intens_pred, np.array(temp_intens_target.shape)[-1])

            # calculating back-propagation
            loss = criterion(temp_phase_pred, spectr_phase_pred, temp_intens_pred, spectr_intens_pred, temp_intens_target, spectr_intens_target) # define fuckers
            
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
            
            plt.plot(range(np.array(temp_intens_pred.shape)[-1]), temp_intens_pred.clone().cpu().detach().abs().numpy()[0,:], color = "red")
            plt.plot(range(np.array(temp_intens_target.shape)[-1]), temp_intens_target.clone().cpu().detach().abs().numpy(), color = "green")
            plt.grid()
            plt.show()

            '''
            fig, test_loss = test(model = model,
                    target_pulse = target_pulse,
                    initial_pulse = initial_pulse,
                    device = my_device, 
                    dtype = my_dtype,
                    iter_num = epoch,
                    save = True,
                    x_type = _axis_type,
                    filter_threshold = filter_threshold)
            
            cont_penalty = 0
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
            '''
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
