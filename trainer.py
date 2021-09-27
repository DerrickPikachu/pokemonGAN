import torch
from wgan import WGAN


class Trainer:
    def __init__(self, args):
        self.wgan = WGAN(args)

    def train(self):
        pass

    def getModel(self):
        return self.wgan


if __name__ == '__main__':
    pass