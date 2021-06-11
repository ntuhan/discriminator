import os
import torch
from torch import nn
from torch import optim
import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, latent_dim=10, batchnorm=True):
        """A generator for mapping a latent space to a sample space.
        The sample space for this generator is single-channel, 28x28 images
        with pixel intensity ranging from -1 to +1.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.batchnorm = batchnorm
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        # Project the input
        self.linear1 = nn.Linear(self.latent_dim, 272*6*5, bias=False)
        self.bn1d1 = nn.BatchNorm1d(272*6*5) if self.batchnorm else None
        self.leaky_relu = nn.LeakyReLU()

        # Convolutions
        self.conv1 = nn.Conv2d(
                in_channels=272,
                out_channels=136,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)
        #6X5
        self.bn2d1 = nn.BatchNorm2d(136) if self.batchnorm else None

        self.conv2 = nn.ConvTranspose2d(
                in_channels=136,
                out_channels=68,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False)
        #7X6
        self.bn2d2 = nn.BatchNorm2d(68) if self.batchnorm else None

        self.conv3 = nn.ConvTranspose2d(
                in_channels=68,
                out_channels=17,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False)
        #13X11
        self.tanh = nn.Tanh()

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = self.linear1(input_tensor)
        intermediate = self.bn1d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = intermediate.view((-1, 272, 6, 5))

        intermediate = self.conv1(intermediate)
        if self.batchnorm:
            intermediate = self.bn2d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)
        
        #print(intermediate.size())
        if self.batchnorm:
            intermediate = self.bn2d2(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv3(intermediate)
        
        #print(intermediate.size())
        intermediate = intermediate.narrow(3, 0, 10)
        output_tensor = intermediate
        #output_tensor = self.tanh(intermediate)
        

        new_tensor = torch.zeros(output_tensor.size())

        for i, batch in enumerate(output_tensor):
            max_indices = torch.argmax(batch.narrow(0, 0, 15), dim=0)
            for j in range(max_indices.size(dim=0)):
                for k in range(max_indices.size(dim=1)):
                    if batch[max_indices[j][k]][j][k] > 0.5: 
                        new_tensor[i][max_indices[j][k]][j][k] = 1
                        if batch[15][j][k].abs() > batch[16][j][k].abs():
                            new_tensor[i][15][j][k] = 1 if batch[15][j][k] > 0 else -1
                        else:
                            new_tensor[i][16][j][k] = 1 if batch[16][j][k] > 0 else -1

        
        return new_tensor


class Discriminator(nn.Module):
    def __init__(self):
        """A discriminator for discerning real from generated images.
        Images must be single-channel and 28x28 pixels.
        Output activation is Sigmoid.
        """
        super(Discriminator, self).__init__()
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.conv1 = nn.Conv2d(
                in_channels=17,
                out_channels=68,
                kernel_size=4,
                stride=2,
                padding=2,
                bias=False)
        #7X6
        self.leaky_relu = nn.LeakyReLU()
        self.dropout_2d = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(
                in_channels=68,
                out_channels=136,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False)
        #6X5
        self.linear1 = nn.Linear(136*6*5, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(input_tensor)
        
        #print(intermediate.size())
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = self.conv2(intermediate)
        
        #print(intermediate.size())
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = intermediate.view((-1, 136*6*5))
        intermediate = self.linear1(intermediate)
        output_tensor = self.sigmoid(intermediate)
        #print("in dis forward")
        #print(output_tensor.size())
        #print(output_tensor)
        #print("finished")
        return output_tensor

class DCGAN():
    def __init__(self, latent_dim, noise_fn, dataloader,
                 batch_size=32, device='cpu', lr_d=1e-3, lr_g=2e-4):
        """
        Args:
            generator: a Ganerator network
            discriminator: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """
        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.BCELoss()
        self.optim_d = optim.Adam(self.discriminator.parameters(),
                                  lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.999))
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)

    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.
        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.
        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None
        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.cpu()  # move images to cpu
        return samples

    def train_step_generator(self):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()

        latent_vec = self.noise_fn(self.batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(classifications, self.target_ones)
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self, real_samples):
        """Train the discriminator one step and return the losses."""
        self.discriminator.zero_grad()

        # real samples
        pred_real = self.discriminator(real_samples)
        #print("size of pre_real")
        #print(pred_real.size())
        #print("size of ones")
        #print(self.target_ones.size())
        loss_real = self.criterion(pred_real, self.target_ones)

        # generated samples
        latent_vec = self.noise_fn(self.batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        # combine
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()

    def train_epoch(self, print_frequency=10, max_steps=0):
        """Train both networks for one epoch and return the losses.
        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        for batch, (real_samples) in enumerate(self.dataloader):
            #print("in train_epoch")
            #print(batch)
            #print(real_samples.size())
            #print()
            real_samples = real_samples.to(self.device)
            ldr_, ldf_ = self.train_step_discriminator(real_samples)
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_
            loss_g_running += self.train_step_generator()
            if print_frequency and (batch+1) % print_frequency == 0:
                print(f"{batch+1}/{len(self.dataloader)}:"
                      f" G={loss_g_running / (batch+1):.3f},"
                      f" Dr={loss_d_real_running / (batch+1):.3f},"
                      f" Df={loss_d_fake_running / (batch+1):.3f}",
                      end='\r',
                      flush=True)
            if max_steps and batch == max_steps:
                break
        if print_frequency:
            print()
        loss_g_running /= batch
        loss_d_real_running /= batch
        loss_d_fake_running /= batch
        print("loss_g_running " + str(loss_g_running))
        print("loss_d_real_running" + str(loss_d_real_running))
        print("loss_d_fake_running" + str(loss_d_fake_running))
        return (loss_g_running, (loss_d_real_running, loss_d_fake_running))
    
    def trans_samples(self, samples):
        room_list = []
        obj_list = []
        obj_dict = {"label":0,
                     "x":0,
                     "y":0,
                     "orientation":0}
        for num in range(samples.size(dim=0)):
            for i in range(samples.size(dim=1)-2):
                for j in range(samples.size(dim=2)):
                    for k in range(samples.size(dim=3)):
                        if samples[num][i][j][k] == 1:
                            obj_dict["label"] = i
                            obj_dict["y"] = j
                            obj_dict["x"] = k
                            if samples[num][16][j][k] == -1:
                                obj_dict["orientation"] = 0
                            elif samples[num][16][j][k] == 1:
                                obj_dict["orientation"] = 2
                            elif samples[num][15][j][k] == -1:
                                obj_dict["orientation"] = 1
                            else:
                                obj_dict["orientation"] = 3
                            #tmp = obj_dict.copy()
                            obj_list.append(obj_dict.copy())
            room_list.append(obj_list.copy())
            obj_list.clear()
                            
        return room_list
    
    def save_dis(self):
        torch.save(self.discriminator.state_dict(), "save.pt")

def main():
    import matplotlib.pyplot as plt
    from time import time
    import urllib, json
    import torch

    
    batch_size = 16
    epochs = 100
    latent_dim = 16
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    
    url = "https://raw.githubusercontent.com/Chaoyuuu/Gather-Town-Datasets/master/datasets.json"
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    input_data = torch.zeros(len(data)-(len(data)%batch_size), 17, 13, 10)

    for i in range(len(input_data)):
        for j in range(len(data[i]["room"])):
            input_data[i][data[i]["room"][j]["label"]][data[i]["room"][j]["y"]][data[i]["room"][j]["x"]] = 1
            if data[i]["room"][j]["orientation"] == 0:
                input_data[i][16][data[i]["room"][j]["y"]][data[i]["room"][j]["x"]] = -1
            elif data[i]["room"][j]["orientation"] == 1:
                input_data[i][15][data[i]["room"][j]["y"]][data[i]["room"][j]["x"]] = -1
            elif data[i]["room"][j]["orientation"] == 2:
                input_data[i][16][data[i]["room"][j]["y"]][data[i]["room"][j]["x"]] = 1
            else:
                input_data[i][15][data[i]["room"][j]["y"]][data[i]["room"][j]["x"]] = 1 
    
    
    dataloader = DataLoader(input_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1
            )
    noise_fn = lambda x: torch.randn((x, latent_dim), device=device)
    gan = DCGAN(latent_dim, noise_fn, dataloader, device=device, batch_size=batch_size)
    start = time()
    for i in range(10):
        print(f"Epoch {i+1}; Elapsed time = {int(time() - start)}s")
        gan.train_epoch()
        
    #result_tensor = gan.generate_samples()
    #result = gan.trans_samples(result_tensor)
    
    #print(result_tensor.size())
    #pred_result = gan.discriminator(result_tensor)
    #print(pred_result)
    
    #print(gan.discriminator(input_data[0]))
    #gan.save_dis()
    #print(len(result[0]))
    #print(result[0])
    #print(result)
    #images = gan.generate_samples() * -1
    #ims = tv.utils.make_grid(images, normalize=True)
    #plt.imshow(ims.numpy().transpose((1,2,0)))
    #plt.show()


if __name__ == "__main__":
    main()
