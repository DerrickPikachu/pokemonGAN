import torch
from torch import nn
from parameters import args


class DCGenerator(nn.Module):
    def __init__(self, latent_size: int):
        super(DCGenerator, self).__init__()
        self.latent_size = latent_size

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, args.num_of_filter * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.num_of_filter * 8),
            nn.ReLU(True),
            # size (c, 4, 4)
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(args.num_of_filter * 8, args.num_of_filter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.num_of_filter * 4),
            nn.ReLU(True),
            # size (c, 8, 8)
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(args.num_of_filter * 4, args.num_of_filter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.num_of_filter * 2),
            nn.ReLU(True),
            # size (c, 16, 16)
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(args.num_of_filter * 2, args.num_of_filter, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.num_of_filter),
            nn.ReLU(True),
            # size (c, 32, 32)
        )

        self.convT_final = nn.Sequential(
            nn.ConvTranspose2d(args.num_of_filter, args.img_channel, 4, 2, 1, bias=False),
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


if __name__ == '__main__':
    generator = DCGenerator(30)
    latent_code = torch.randn(30)
    img = generator(latent_code)
    print(img)
