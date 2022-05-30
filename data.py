import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data

import numpy as np

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, transforms

from PIL.Image import NEAREST
import os.path

GN_G_plot_buffer = []
PN_G_plot_buffer = []
DN_G_plot_buffer = []
num_of_batch = 1
num_of_cores = 1
input_resolution = 256
max_epoch_size = 900  # debug with smaller batch
extra_transform_data = False
resize_interpolation = torchvision.transforms.InterpolationMode.NEAREST
new_torchvision_version = int(torchvision.__version__.__str__().split('.')[1]) >= 8
if not new_torchvision_version:
    print('Library torchvision version < 0.7.0.')
    resize_interpolation = NEAREST  # for torchvision 0.7.0


class PATH:
    def __init__(self, input_path=''):
        self.path = ''
        self.output_dir = ''
        self.image_dir = ''
        self.plot_dir = ''
        self.model_dir = ''
        self.data_dir = ''
        self.test_dir = ''
        self.tensorboard_path = ''
        self.update_path(input_path)

    def create_path(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
            print(f'Created Directory: {path}')

    def update_path(self, input_path):
        self.path = str(input_path)
        self.data_dir = self.path + 'new_data/'
        self.test_dir = self.data_dir + 'test/'
        self.update_output_path(self.path)

    def update_output_path(self, path, create=False):
        self.output_dir = path + 'outputs/'
        self.image_dir = self.output_dir + 'images/'
        self.plot_dir = self.output_dir + 'plots/'
        self.model_dir = self.output_dir + 'model/'
        self.tensorboard_path = self.output_dir + 'tensorboard/'

        if create:
            self.create_path(self.output_dir)
            self.create_path(self.image_dir)
            self.create_path(self.plot_dir)
            self.create_path(self.model_dir)
            self.create_path(self.tensorboard_path)

    def print_path(self):
        print(f'{self.output_dir} || {self.data_dir} || {self.test_dir}')


paths = PATH()
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])


class ReSizeAndCrop(object):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, resolution, random_crop_range=1.25):
        # fill colour for center crop
        # https://discuss.pytorch.org/t/torchvision-transforms-set-fillcolor-for-centercrop/98098/2

        self.transform_random_crop = transforms.Compose([
            transforms.CenterCrop(size=int(resolution * random_crop_range)),
            transforms.RandomCrop(size=resolution, fill=(255, 255, 255), pad_if_needed=True),
        ])
        self.transform_center_crop = transforms.CenterCrop(size=int(resolution))
        self.transform_resize = transforms.Resize((resolution, resolution), interpolation=resize_interpolation)

        self.area = resolution * resolution
        self.crop_area = self.area * random_crop_range * random_crop_range
        self.resolution = resolution

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        # h, w = image.shape[:2]
        h, w = sample.size
        # declare random crop on the fly???

        area = h * w
        if area > self.crop_area:
            return self.transform_random_crop(sample)
        elif area > self.area:
            return self.transform_center_crop(sample)
        elif area < self.area:
            return self.transform_resize(sample)

        return sample


def get_model_directory():
    return paths.model_dir


def get_tensorboard_directory():
    return paths.tensorboard_path


def update_path(input_path):
    paths.update_path(input_path)


def update_output_path(input_path):
    paths.update_output_path(input_path, create=True)


def get_dataloader(directory='', mode='resize', apply_train_transform=True):
    # data folder is loaded continuously, don't modify it when running!
    transform_list = []
    if apply_train_transform:
        transform_list.append(transforms.RandomHorizontalFlip())  # may affect performance
        transform_list.append(transforms.RandomVerticalFlip())
        if extra_transform_data:
            transform_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.0, saturation=0.1, hue=0.3))
            transform_list.append(transforms.RandomGrayscale(p=0.02))
    # Resizing image
    if mode == 'mixed':
        resize_crop_transform = ReSizeAndCrop(resolution=input_resolution)
        transform_list.append(resize_crop_transform)
    elif mode == 'crop':
        transform_list.append(transforms.RandomCrop(size=input_resolution, fill=(255, 255, 255), pad_if_needed=True))
    else:
        transform_list.append(transforms.Resize((input_resolution, input_resolution), interpolation=resize_interpolation))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    data_transforms = transforms.Compose(transform_list)

    image_datasets = datasets.ImageFolder(paths.data_dir + directory, transform=data_transforms)
    if len(image_datasets) > max_epoch_size:
        image_datasets, _ = torch.utils.data.random_split(image_datasets, [max_epoch_size, len(image_datasets) - max_epoch_size])

    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    image_dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=num_of_batch, shuffle=True,
                                                   num_workers=num_of_cores)  # Num of GPU # , pin_memory=True
    return image_dataloader


def resize(image, scale):
    # pytorch not use true NN for efficiency, https://github.com/pytorch/pytorch/issues/34808
    # gradient is concentrated to 1 block when shrinking
    # Gives warning if fractional scale factor
    # return nn.functional.interpolate(input, scale_factor=scale, recompute_scale_factor=False)
    return nn.functional.interpolate(image, scale_factor=scale, recompute_scale_factor=True)


def desample(image, scale, distribute_grad=True):
    if scale <= 0.0:
        return image
    if not distribute_grad:
        intermediate = nn.functional.interpolate(image, scale_factor=1.0/scale, recompute_scale_factor=True)
        return nn.functional.interpolate(intermediate, scale_factor=scale, recompute_scale_factor=True)
    forward_value = nn.functional.interpolate(image, scale_factor=1.0/scale, recompute_scale_factor=True)
    forward_value = nn.functional.interpolate(forward_value, scale_factor=scale, recompute_scale_factor=True)
    output = image.clone()
    output.data = forward_value.data
    return output


def save_image(image, filename, normalize=True, directory=paths.image_dir):
    if normalize: image = image / 2 + 0.5  # activation layer tanh in range [-1, 1]
    torchvision.utils.save_image(image, directory + filename + '.png')


def plot_loss(x, y, title='', save=True):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    if save:
        plt.savefig(paths.plot_dir + title + '_plot.png')
        plt.close()
    else: plt.show()


def plot_all(suffix='', save=True):
    plot_loss(range(0, len(GN_G_plot_buffer)), GN_G_plot_buffer, 'GN_' + suffix, save)
    plot_loss(range(0, len(PN_G_plot_buffer)), PN_G_plot_buffer, 'PN_' + suffix, save)
    plot_loss(range(0, len(DN_G_plot_buffer)), DN_G_plot_buffer, 'DN_' + suffix, save)
    print('Finished Plotting')


def imshow(img, normalize=True):
    if normalize: img = img / 2 + 0.5  # unnormalize
    plt.imshow(img.squeeze().permute(1, 2, 0))  # plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()
    # imshow(torchvision.utils.make_grid(inputs))


def choose_best(input_set):
    # Choosing best output from PN
    outputs, image, min_loss = input_set[0]

    for i in range(1, len(input_set)):
        ith_outputs, ith_image, ith_loss = input_set[i]
        if ith_loss < min_loss:
            outputs = ith_outputs
            image = ith_image
            min_loss = ith_loss

    return outputs, image, min_loss


def choose_random(input_set):
    # Choose randomly
    pos = np.random.randint(len(input_set))
    return input_set[pos]


def choose_min(input_set):
    # loss_64, loss_48, loss_32 = input_set
    # pos = np.argmin((loss_64.item(), loss_48.item(), loss_32.item()))
    pos = np.argmin(input_set)
    return input_set[pos]


def normalize(input, min=0, max=1):
    input_min, input_max = torch.min(input.data), torch.max(input.data)
    return (input - input_min) / (input_max - input_min) * (max - min) + min


def clear_cache(require_print=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if require_print:
            print(f'Memory allocated: {torch.cuda.memory_allocated()}')
            print(f'Memory reserved: {torch.cuda.memory_reserved()}')
