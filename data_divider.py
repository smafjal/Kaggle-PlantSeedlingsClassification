# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
This script is useful for data splitting.
Such as, We have a dataset for the image classification task,
where dataset hase different classes(ants,bees,dogs etc)
It's better to split the dataset as train/val and make proper model.
This script will help to do the things.

DATA_DIR: orginal dataset
SAVE_DIR: splitted dataset will be saved here
DEVIDE_PERCENTAGE: train/val ratio
"""

from os import listdir
from os.path import isfile, join
import subprocess
import sys

DEVIDE_PERCENTAGE = 70  # train-test split
DATA_DIR = 'data'  # root data folder
SAVE_DIR = 'mydata'  # train/val divided data will be save on this folder


def get_classes(_dir):
    class_dir_list = [join(_dir, f) for f in listdir(_dir)]
    return class_dir_list


def devide_and_save(class_dir, save_dir, _move):
    files_list = [join(class_dir, f) for f in listdir(class_dir) if isfile(join(class_dir, f))]
    
    limit = int(DEVIDE_PERCENTAGE * len(files_list)) / 100
    
    # for train
    _dir = join(save_dir, 'train')
    print("Data saved on: ", _dir)
    for _file in files_list[:limit]:
        _src = _file
        _dest = join(_dir, '/'.join(_src.split('/')[-2:]))
        _save_dir = "/".join(_dest.split('/')[:-1])
        
        subprocess.call(['mkdir', '-p', _save_dir])
        if _move == 1:
            subprocess.call(['mv', _src, _dest])
        elif _move == 0:
            subprocess.call(['cp', _src, _dest])
    
    # for validation
    _dir = join(save_dir, 'val')
    print("Data saved on: ", _dir)
    for _file in files_list[limit:]:
        _src = _file
        _dest = join(_dir, '/'.join(_src.split('/')[-2:]))
        _save_dir = "/".join(_dest.split('/')[:-1])
        
        subprocess.call(['mkdir', '-p', _save_dir])
        if _move == 1:
            subprocess.call(['mv', _src, _dest])
        elif _move == 0:
            subprocess.call(['cp', _src, _dest])


def main(_move):
    class_dir_list = get_classes(DATA_DIR)
    for idx, class_dir in enumerate(class_dir_list):
        print(idx, class_dir)
        devide_and_save(class_dir, SAVE_DIR, _move)


if __name__ == '__main__':
    """
    Use python data_divider.py 1(move)/0(copy)
    """
    if len(sys.argv) == 2:
        _move = (int)(sys.argv[1])
        main(_move)
    else:
        print("Use python " + sys.argv[0] + " 1(move)/0(copy)")
