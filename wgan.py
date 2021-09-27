import torch
from torch import nn
from parameters import args


class DCGenerator(nn.Module):
    def __init__(self, latent_size: int, num_of_filter: int, img_channel: int):
        super(DCGenerator, self).__init__()
        self.latent_size = latent_size
        self.num_of_filter = num_of_filter
        self.img_channel = img_channel

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, num_of_filter * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_of_filter * 8),
            nn.ReLU(True),
            # size (c, 4, 4)
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(num_of_filter * 8, num_of_filter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_filter * 4),
            nn.ReLU(True),
            # size (c, 8, 8)
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(num_of_filter * 4, num_of_filter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_filter * 2),
            nn.ReLU(True),
            # size (c, 16, 16)
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(num_of_filter * 2, num_of_filter, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_filter),
            nn.ReLU(True),
            # size (c, 32, 32)
        )

        self.convT_final = nn.Sequential(
            nn.ConvTranspose2d(num_of_filter, img_channel, 4, 2, 1, bias=False),
            nn.Tanh(),
            # size (c, 64, 64)
            # Use tanh to force all value in [-1, 1] section
        )

    def forward(self, latent_code):
        latent_code = self._fix_latent_code_size(latent_code)
        result = self.convT1(latent_code)
        result = self.convT2(result)
        result = self.convT3(result)
        result = self.convT4(result)
        return self.convT_final(result)
        # return self.main(latent_code)

    def _fix_latent_code_size(self, latent_code):
        return latent_code.view(-1, args.latent_size, 1, 1)


class DCDiscriminator(nn.Module):
    def __init__(self, num_of_filter: int, img_channel: int):
        super(DCDiscriminator, self).__init__()

        self.conv1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(img_channel, num_of_filter, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_of_filter, num_of_filter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_filter * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_of_filter * 2, num_of_filter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_filter * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(num_of_filter * 4, num_of_filter * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_filter * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(num_of_filter * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        result = self.conv1(img)
        result = self.conv2(result)
        result = self.conv3(result)
        result = self.conv4(result)
        return self.conv_final(result)


class WGAN:
    def __init__(self, latent_size: int, num_of_gen_filter: int, num_of_dis_filter: int, img_channel: int):
        self.latent_size = latent_size

        # Initialize model
        self.generator = DCGenerator(latent_size, num_of_gen_filter, img_channel)
        self.discriminator = DCDiscriminator(num_of_dis_filter, img_channel)

        # Initialize optimizer
        # self.generator_optimizer = ?
        # self.discriminator_optimizer = ?

    def generate_img(self, num_of_img):
        latent_code = torch.randn((num_of_img, self.latent_size))
        return self.generator(latent_code)

    def discriminate_img(self, img):
        return self.discriminator(img)

    def update_generator(self):
        pass

    def update_discriminator(self):
        pass


if __name__ == '__main__':
    generator = DCGenerator(30, 64, 3)
    discriminator = DCDiscriminator(32, 3)

    latent_code = torch.randn(30)
    img = generator(latent_code)
    ans = discriminator(img)
    print(ans.item())
    # print(img)

