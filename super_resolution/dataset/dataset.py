import torch
import torch.utils.data as data
import glob, os
from os import listdir
from os.path import join
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

import scipy.io as sio
class DatasetFromMat(data.Dataset):
    def __init__(self, image_dir, filters='x2', input_label='im_b_y', target_label='im_gt_y'):
        super(DatasetFromMat, self).__init__()
        lists = glob.glob(os.path.join(image_dir, "*.mat"))
        self.image_filenames = []
        for name in lists:
            if filters in name:
                self.image_filenames.append(name)
        #[join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_label = input_label
        self.target_label = target_label

    def __getitem__(self, index):
        name = self.image_filenames[index]
        im_gt_y = sio.loadmat(name)[self.target_label]
        im_b_y = sio.loadmat(name)[self.input_label]
        im_gt_y = im_gt_y.astype(float)
        im_b_y = im_b_y.astype(float)

        im_input = im_b_y
        target = im_gt_y
        return im_input, target

    def __len__(self):
        return len(self.image_filenames)

import h5py
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        lists = glob.glob(os.path.join(file_path, "*.h5"))
        if len(lists) != 1:
            raise RuntimeError("multiple h5 file found")

        self.data = None
        for name in lists:
            hf = h5py.File(name, "r", libver='latest', swmr=True)
            self.data = hf.get('data')
            self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]


