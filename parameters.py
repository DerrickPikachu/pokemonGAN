import argparse


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--file_root', default='dataset/')
    parser.add_argument('--csv_filename', default='pokemon.csv')
    parser.add_argument('--img_dir', default='jpg_image/')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--num_of_gen_filter', default=64, type=int)
    parser.add_argument('--num_of_dis_filter', default=64, type=int)
    parser.add_argument('--img_channel', default=3, type=int)
    parser.add_argument('--latent_size', default=30, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)

    return parser.parse_args()


args = build_parser()
