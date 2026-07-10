import os
import shutil
import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)

    opt = parser.parse_args()
    return opt



if __name__ == '__main__':

    opt = parse_args()

    if not os.path.exists(opt.test_dir):
        os.makedirs(opt.test_dir)

    file_list = sorted(os.listdir(opt.train_dir))

    for i, fname in enumerate(file_list):

        if i % 5 > 0:
            continue

        shutil.move(os.path.join(opt.train_dir, fname), os.path.join(opt.test_dir, fname))