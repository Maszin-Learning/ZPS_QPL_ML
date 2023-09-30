import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from dcgan import Discriminator, Generator, weights_init
from preprocessing import Dataset
import spectral_analysis as sa


lr = 5e-4#5e-4
beta1 = 0.3
beta2 = 0.80
epoch_num = 70
batch_size = 512
nz = 100  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
dtype = torch.float32

### GAUSS
#lr = 5e-4#5e-4
#beta1 = 0.3
#beta2 = 0.80
#epoch_num = 70
#batch_size = 512
#nz = 100  # length of noise
##ngpu = 0
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps")

#lr = 5e-4#5e-4
#beta1 = 0.3
#beta2 = 0.90
#epoch_num = 10000
#batch_size = 128
#nz = 100  # length of noise



pulse_1 = sa.gaussian_pulse((1550,1560), 1555, 4, x_type='freq')
pulse_1.x_type = "wl"
pulse_1.wl_to_freq()
pulse_1.Y = pulse_1.Y
signal_len=len(pulse_1)
print(signal_len)

plt.plot(pulse_1.Y)
plt.savefig('gauss_GAN.png')
plt.close()

pulse_2 = sa.hermitian_pulse((1550,1560), 1555, 4, x_type='freq')
pulse_2.x_type = "wl"
pulse_2.wl_to_freq()
pulse_2.Y = pulse_2.Y

plt.plot(pulse_2.Y)
plt.savefig('hermit_GAN.png')
plt.close()



pulse_2_Y_real=pulse_2.Y.real
pulse_2_Y_imag=pulse_2.X.imag
pulse_2_Y_abs_tensor = torch.tensor(np.abs(pulse_2.Y), requires_grad=True, device=device, dtype=dtype).reshape(1,signal_len)


pulse_1.fourier()




    x = torch.tensor([pulse_1.Y.real, pulse_1.Y.imag], requires_grad=True, device=device_, dtype=dtype_)
    x = x.reshape(signal_len, 2)
    pulse_1_torch_Y = torch.view_as_complex(x)
    pulse_1_torch_Y = pulse_1_torch_Y.reshape(1, signal_len)
    #print(pulse_1_torch_Y.shape)
    #print(torch.exp(1j*results).shape)
    pulse_1_torch_Y = torch.mul(pulse_1_torch_Y, torch.exp(1j*results))
    pulse_1_torch_Y = torch.fft.ifft(pulse_1_torch_Y)
    pusle_1_Y_abs_tensor = pulse_1_torch_Y.abs()
    #pulse_1_conc_result_torch= torch.concatenate((pulse_1_torch_Y.real, pulse_1_torch_Y.imag), axis=1)
    loss = criterion(pusle_1_Y_abs_tensor, pulse_2_Y_abs_tensor) # Calculate Loss/criterion
    
    loss.backward() # backward propagation
    optimizer.step() # Updating parameters
    loss_list.append(loss.data) # store loss
    
    # print loss
    if epoch % 500 == 0:
        pulse_1.Y = pulse_1.Y*np.exp(1j*results.clone().detach().numpy().reshape(signal_len,))
        pulse_1.inv_fourier()   
        sa.plot(pulse_1, title=f'reconstructed_{epoch}' , save=True)
        pulse_1.fourier()
        print('epoch {}, loss {}'.format(epoch, loss.data))


def main():
    # load training data
    trainset = Dataset('./data_hermit/')

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    # init netD and netG
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator(nz).to(device)
    netG.apply(weights_init)


    criterion = nn.BCELoss()

    # used for visualzing training process
    fixed_noise = torch.randn(16, nz, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    for epoch in range(epoch_num):
        for step, (data, _) in enumerate(trainloader):

            real_cpu = data.to(device)
            b_size = real_cpu.size(0)

            # train netD
            label = torch.full((b_size,), real_label,
                               dtype=torch.float, device=device)
            netD.zero_grad()
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train netG
            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            netG.zero_grad()

            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epoch_num, step, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # save training process
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            f, a = plt.subplots(4, 4, figsize=(8, 8))
            for i in range(4):
                for j in range(4):
                    a[i][j].plot(fake[i * 4 + j].view(-1))
                    a[i][j].set_xticks(())
                    a[i][j].set_yticks(())
            plt.savefig('./img/dcgan_epoch_%d.png' % epoch)
            plt.close()
    
    # save models
    torch.save(netG, './nets/dcgan_netG.pkl')
    torch.save(netD, './nets/dcgan_netD.pkl')


if __name__ == '__main__':
    main()