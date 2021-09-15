import csv
import numpy as np
from PIL import Image

filename = 'pokemon.csv'


def read_csv(root: str):
    result = list()
    with open(root + filename, newline='') as csv_file:
        rows = csv.reader(csv_file)
        for row in rows:
            result.append(row[0])
    return result[1:]


def turn_jpg_to_png(rows):
    for name in rows:

        img = Image.open('dataset/images/' + name + '.png')
        img = img.convert('RGB')
        img_np = np.array(img)
        for channel in range(3):
            for row in range(120):
                for col in range(120):
                    if img_np[row, col, channel] == 0:
                        img_np[row, col, channel] = 255
        img = Image.fromarray(np.uint8(img_np))
        img.save('dataset/jpg_image/' + name + '.jpg')


if __name__ == '__main__':
    rows = read_csv('dataset/')
    turn_jpg_to_png(rows)
