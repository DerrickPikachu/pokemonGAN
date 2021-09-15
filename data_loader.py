import PIL.ImageFile
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils import data
from torch.utils.data.dataset import T_co
from parameters import args


class PokemonDataset(data.Dataset):
    def __init__(self, root: str, csv_filename: str, img_dir: str, trans=None):
        self.root = root
        self.csv_filename = csv_filename
        self.img_dir = img_dir
        self.img_filenames = self._read_csv(self.root)
        self.trans = trans
        print(f'>> Find {len(self.img_filenames)} images')

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index) -> torch.Tensor:
        img = self._load_img(self.img_filenames[index])
        img = img.convert('RGB')
        if self.trans is not None:
            img = self.trans(img)
        return img

    def _read_csv(self, root: str) -> list:
        result = list()
        with open(root + self.csv_filename, newline='') as csv_file:
            rows = csv.reader(csv_file)
            for row in rows:
                result.append(row[0] + '.jpg')
        return result[1:]

    def _load_img(self, filename: str) -> PIL.ImageFile.ImageFile:
        return Image.open(self.root + self.img_dir + filename)

    def _img_preprocess(self, img: PIL.ImageFile.ImageFile) -> np.ndarray:
        img = img.convert('RGB')
        return np.array(img)


def show_grid_img(batch: torch.Tensor):
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Training Images')
    plt.imshow(np.transpose(vutils.make_grid(batch[:args.img_size], padding=2, normalize=True), (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    dataset = PokemonDataset(args.file_root, args.csv_filename, args.img_dir, trans=transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    batch = next(iter(loader))
    show_grid_img(batch)

