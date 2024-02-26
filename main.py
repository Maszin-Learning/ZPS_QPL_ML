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
import utilities as u
from dataset import Dataset
from torch.utils.data import DataLoader #Dataloader module
import torchaudio
from test import create_test_pulse, test, create_test_set, create_initial_pulse
import torchvision.transforms as transforms  # Transformations and augmentations
from dataset import Dataset_train
from dataset_generator import Generator
import utilities
import argparse
import wandb
import shutil
import warnings
from utilities import MSEsmooth

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
          'test_signal:', _test_signal, '\n',
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
    zeroes_num = 2500   # number of zeroes we add on the left and on the right of the main pulse (to make FT intensity broader)

    bandwidth = [0, 1000]
    centre = 500
    width = 100

    initial_pulse = create_initial_pulse(bandwidth = bandwidth,
                                         centre = centre,
                                         FWHM = width,
                                         num = input_dim,
                                         pulse_type = _initial_signal)
    


    # normalize it in L2

    initial_pulse.Y = initial_pulse.Y / np.sqrt(np.sum(initial_pulse.Y*np.conjugate(initial_pulse.Y)))

    # this serves only to generate FT pulse

    long_pulse = initial_pulse.zero_padding(length = zeroes_num, inplace = False) 

    
    # additional pulse to add to exp (gauss) so it makes it more physical
    signal_correction = create_initial_pulse(bandwidth = bandwidth,
                                         centre = centre,
                                         FWHM = width/10,
                                         num = long_pulse.Y.shape[0],
                                         pulse_type = 'gauss')
    
    
    
    long_pulse_2 = long_pulse.copy()    
    long_pulse_2.Y = np.convolve(long_pulse_2.Y, signal_correction.Y, mode='same')
    long_pulse_2.Y = long_pulse_2.Y / np.sqrt(np.sum(long_pulse_2.Y*np.conjugate(long_pulse_2.Y)))
    Y_initial = initial_pulse.Y.copy()

    # we want to find what is the bandwidth of intensity after FT, to estimate output dimension of NN

    trash_fraction = 0.005 # percent of FT transformed to be cut off - it will contribute to the noise

    long_pulse.fourier()
    fwhm_init_F = u.comp_FWHM(u.comp_std(initial_pulse.fourier(inplace = False).X, initial_pulse.fourier(inplace = False).Y))
    x_start = long_pulse.quantile(trash_fraction/2, norm = "L2")
    x_end = long_pulse.quantile(1-trash_fraction/2, norm = "L2")
    idx_start = np.searchsorted(long_pulse.X, x_start)
    idx_end = np.searchsorted(long_pulse.X, x_end)
    if (idx_end - idx_start) % 2 == 1:
        idx_end += 1
    output_dim = idx_end - idx_start    # number of points of non-zero FT-intensity

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
    #test_pulse = test_pulse * 1.05
    fwhm_test = u.comp_FWHM(u.comp_std(initial_pulse.X.copy(), test_pulse.clone().detach().cpu().numpy().ravel()))
    print("\nTime-bandwidth product of the transformation from the initial pulse to the test pulse is equal to {}.\n".format(round(fwhm_test*fwhm_init_F/2, 5)))   # WARNING: This "/2" is just empirical correction
    if fwhm_test*fwhm_init_F/2 < 0.44:
        print("TRANSFORMATION IMPOSSIBLE\n")
    test_set = create_test_set(initial_pulse, output_dim, my_device, my_dtype)

    # create dataset and wrap it into dataloader

    if _generate:
        print("\nCreating training set...")
        
        the_generator = Generator(data_num = _dataset_size,
                                initial_intensity = Y_initial,
                                FT_X = pulse_ft.X,
                                phase_len = output_dim,
                                device = my_device,
                                dtype = np.float32,
                                target_type = _test_signal,
                                target_metadata = [500, 100, bandwidth[0], bandwidth[1]]
                                )

        the_generator.generate_and_save()
        print("Successfully created training set containing {} spectra.\n".format(len(os.listdir('data/train_intensity'))))

        u.plot_dataset(100, pulse = initial_pulse, ft_pulse = pulse_ft)

        print("Calculating mean Time-Bandwidth Product of the training set...")
        TBP_mean, TBP_std = u.comp_mean_TBP(initial_pulse.X, fwhm_init_F)
        print("Mean TBP of the transformation from initial pulse to a spectrum from dataset is equal to {} +- {}.\n".format(round(TBP_mean, 5), round(TBP_std, 5)))   

        exit()

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
    
    if _criterion =='MSE':
        criterion = torch.nn.MSELoss()
    if _criterion =='L1':
        criterion = torch.nn.L1Loss()
    if _criterion =='MSEsmooth':
        criterion = MSEsmooth(device = my_device, dtype = my_dtype, c_factor = 0.6)
    
    # create dataset and dataloader
    
    dataset_train = Dataset_train(root='', transform=True, device = my_device)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=_batch_size, num_workers=0, shuffle=True)
    
    # learning loop

    loss_list = []
    test_loss_global = 10000
    wandb.watch(model, criterion, log="all", log_freq=400)

    for epoch in range(_epoch_num):
        for pulse, _ in tqdm(dataloader_train):
            # pulse = pulse.to(my_device) # the pulse is already created on device by dataset, uncomment if not using designated dataset for this problem
            
            # predict phase that will transform gauss into this pulse
            #predicted_phase = utilities.unwrap(model(pulse))
            predicted_phase = model(pulse)
            #print(predicted_phase_t)
            #print(pulse.shape)
            #utilities.unwrap(model(pulse))
            #if epoch > 0.5*_epoch_num:
            #    print('FILTERING')
            #    predicted_phase = torchaudio.functional.lowpass_biquad(waveform=predicted_phase, sample_rate=1, cutoff_freq=200)
            #predicted_phase = predicted_phase  % (2*np.pi)

            # transform gauss into something using this phase
            initial_intensity = u.np_to_complex_pt(long_pulse_2.Y.copy(), device = my_device, dtype = my_dtype)
            reconstructed_intensity = u.evolve_pt(initial_intensity, predicted_phase, device = my_device, dtype = my_dtype)

            # calculating back-propagation
            if _criterion == "MSEsmooth":
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
                    test_pulse = test_pulse,
                    test_phase = test_phase,
                    initial_pulse = long_pulse_2.copy(),
                    device = my_device, 
                    dtype = my_dtype,
                    iter_num = epoch,
                    save = True,
                    x_type = _axis_type)
            
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

            test_set_losses = []
            for test_signal in test_set:
                fig, test_loss_temp = test(model = model,
                                test_pulse = test_signal,
                                test_phase = test_phase,
                                initial_pulse = long_pulse_2.copy(),
                                device = my_device, 
                                dtype = my_dtype,
                                iter_num = epoch,
                                save = False)
                test_set_losses.append(test_loss_temp)
                fig.close()

            wandb.log({"test_set_loss": np.mean(test_set_losses)})
            print('test_set_loss', np.mean(test_set_losses))

            model.train()


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning) # ignore warnings from plotly
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', default=5e-4, type=float) # designed for network_1
    parser.add_argument('-en', '--epoch_num', default=10, type=int)
    parser.add_argument('-bs', '--batch_size', default=2, type=int)
    parser.add_argument('-pf', '--plot_freq', default=3, type=int)
    parser.add_argument('-ds', '--dataset_size', default=5000, type=int)
    parser.add_argument('-g', '--generate', action='store_true') # only generate, training will not run, wandb will be offline
    parser.add_argument('-fc', '--force_cpu', action='store_true')
    parser.add_argument('-tr', '--test_run', action='store_true')
    parser.add_argument('-nn', '--node_number', default=100, type=int)
    parser.add_argument('-ar', '--architecture', default='network_1', type=str)
    parser.add_argument('-cr', '--criterion', default='MSEsmooth', type=str)
    parser.add_argument('-op', '--optimalizer', default='Adam', type=str,)
    parser.add_argument('-ts', '--test_signal', default='gauss', type=str,)
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
        "test_signal": args.test_signal,
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
         args.test_signal,
         args.initial_signal,
         args.weight_decay,
         args.axis_type)
