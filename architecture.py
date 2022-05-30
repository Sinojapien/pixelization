# import functools

import torch
import torch.nn as nn
import torch.nn.functional as func

import numpy as np
import data as data

# Device Section
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device = torch.device("cuda")

total_weight = 1.0
res_block_depth = 9
new_model = False
new_model_output_block_length = 3
norm_track = True
norm_affine = False
use_label_smoothing = False
use_shortcut = True
init_para = True
use_variation = False
use_upsample_conv = False
apply_res_act = False
normalize_dropout = False
use_lsgan = False
detach_network = True
use_buffer = True
default_buffer_size = 30
new_discriminator = False
new_GN = False

print(f'Res-block Use Shortcut: {use_shortcut}')
print()

# https://www.cnblogs.com/jiangkejie/p/13390377.html
# on inplace


class Weights:
    def __init__(self, l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0, d_gan=1.0, d_target=1.0, idt=1.0):
        self.l1 = l1
        self.grad = grad
        self.image = image
        self.net = net
        self.null = null
        self.gan = gan
        self.d_gan = d_gan
        self.d_target = d_target
        self.idt = idt

    def reset(self):
        self.l1 = 1.0
        self.grad = 1.0
        self.image = 1.0
        self.net = 1.0
        self.null = 1.0
        self.gan = 1.0
        self.d_gan = 1.0
        self.d_target = 1.0
        self.idt = 1.0

    def to_string(self):
        return f'l1: {self.l1}, g: {self.grad}, mm: {self.image}, mn: {self.net}, null: {self.null}, gan: {self.gan}, discriminator = {self.d_gan}'

    def print_info(self):
        print(self.__dict__)


# GN_weights = Weights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
# PN_weights = Weights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
# DN_weights = Weights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
GN_weights = Weights()
PN_weights = Weights()
DN_weights = Weights()


class NetworkOption:
    def __init__(self):
        self.create = True
        self.train = True
        self.load_generator = True
        self.load_discriminator = True

        self.l1 = True
        self.grad = True
        self.gan = True
        self.gan_g_interval = 1
        self.gan_d_interval = 1
        self.mirror_image = True
        self.mirror_network = True
        self.identity = True

        self.separate_weights = []

    def reset(self):
        self.create = True
        self.train = True
        self.load_generator = True
        self.load_discriminator = True

        self.l1 = True
        self.grad = True
        self.gan = True
        self.gan_g_interval = 1
        self.gan_d_interval = 1
        self.mirror_image = True
        self.mirror_network = True
        self.identity = True

        self.separate_weights = []

    def set_baseline(self):
        self.l1 = True
        self.grad = False
        self.gan = False
        self.gan_g_interval = 1
        self.gan_d_interval = 1
        self.mirror_image = False
        self.mirror_network = False
        self.identity = False

        self.separate_weights = []

    def set_options(self, l1, grad, gan, image, network, idt=False):
        self.l1 = l1
        self.grad = grad
        self.gan = gan
        self.mirror_image = image
        self.mirror_network = network
        self.identity = idt

    def set_interval(self, g=1, d=1):
        self.gan_g_interval = int(g)
        self.gan_d_interval = int(d)

    def print_info(self):
        print(self.__dict__)


class CombinedOptions(NetworkOption):
    def __init__(self, tag, weights: Weights = Weights()):
        super().__init__()
        self.tag = tag
        self.title = ''
        self.print = False
        self.save = False

        self.weights = weights

    def print_info(self):
        print(self.__dict__)
        self.weights.print_info()


class LossBuffer:
    def __init__(self):
        self.GN_D_loss = 0.0
        self.PN_D_loss = 0.0
        self.DN_D_loss = 0.0
        self.GN_G_loss = 0.0
        self.PN_G_loss = 0.0
        self.DN_G_loss = 0.0

    def clear(self):
        self.GN_D_loss = 0.0
        self.PN_D_loss = 0.0
        self.DN_D_loss = 0.0
        self.GN_G_loss = 0.0
        self.PN_G_loss = 0.0
        self.DN_G_loss = 0.0

    def get_total_loss(self):
        return self.GN_D_loss + self.PN_D_loss + self.DN_D_loss + self.GN_G_loss + self.PN_G_loss + self.DN_G_loss

    def sum(self, values):
        if isinstance(values, (list, tuple)):
            total = 0.0
            for value in values:
                total = total + value
            return total
        return values

    def backward(self, loss, retain_graph=False):
        if torch.is_tensor(loss):
            loss.backward(retain_graph=retain_graph)
        elif isinstance(loss, (list, tuple)):
            for value in loss:
                value.backward(retain_graph=retain_graph)

    def print_info(self):
        print(f'D:: GN: {self.sum(self.GN_D_loss)}, PN: {self.sum(self.PN_D_loss)}, DN: {self.DN_D_loss}')
        print(f'G:: GN: {self.GN_G_loss}, PN: {self.PN_G_loss}, DN: {self.DN_G_loss}')


def print_weights():
    print(f'Total weights: {total_weight}')
    print('GN loss weights:' + GN_weights.to_string())
    GN_weights.print_info()
    print('PN loss weights:' + PN_weights.to_string())
    PN_weights.print_info()
    print('DN loss weights:' + DN_weights.to_string())
    DN_weights.print_info()
    print(f'Resblock Depth: {res_block_depth}')


def get_device():
    print(f'Cuda available: {torch.cuda.is_available()}')
    return device


def create_noise(variance=0.5):
    # https://jonathan-hui.medium.com/gan-what-is-wrong-with-the-gan-cost-function-6f594162ce01
    # consider adding noise
    noise = torch.rand(1, 3, 256, 256, device=device)  # [0, 1)
    # noise = noise.to(device)
    noise = 2 * noise - 1  # [-1, 1)
    # noise = noise * variance * (1 - current_epochs / 150.0)  # [-0.5, 0.5)
    noise = noise * variance
    return noise


class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return func.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class ConvBlock(nn.Module):
    def __init__(self, in_dimension, out_dimension, kernel_size, stride=1, padding=0, act=True, norm=True):
        super(ConvBlock, self).__init__()
        self.ACTIAVTE = act
        self.NORMALIZE = norm

        self.conv = nn.Conv2d(in_channels=in_dimension, out_channels=out_dimension, kernel_size=kernel_size,
                              stride=stride, padding=padding)  # 3 RGB  # , device=device
        self.ins = nn.InstanceNorm2d(num_features=out_dimension, affine=norm_affine, track_running_stats=norm_track)  # Won't broke, user defined? Use RGB

        if init_para:
            nn.init.normal_(self.conv.weight.data, 0.0, 0.02)
            # print(self.conv.weight.data)

    def forward(self, x):
        x = self.conv(x)
        if self.NORMALIZE: x = self.ins(x)
        if self.ACTIAVTE: x = func.relu(x, inplace=True)  # Activation
        return x


class DeConvBlock(nn.Module):
    # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    def __init__(self, in_dimension, out_dimension, kernel_size, stride=1, padding=0, out_padding=0, act=True):
        super(DeConvBlock, self).__init__()

        self.ACTIAVTE = act
        self.ins = nn.InstanceNorm2d(num_features=out_dimension, affine=norm_affine, track_running_stats=norm_track)
        # https://medium.com/hoskiss-stand/cycle-gan-note-bd166d9ff176
        # https://distill.pub/2016/deconv-checkerboard/
        if not use_upsample_conv:
            self.deconv = nn.ConvTranspose2d(in_channels=in_dimension, out_channels=out_dimension, kernel_size=kernel_size,
                                             stride=stride, padding=padding, output_padding=out_padding)
            if init_para:
                nn.init.normal_(self.deconv.weight.data, 0.0, 0.02)
        else:
            # in, out kernel_size, scale_factor, padding, not used
            upSampleConvSequence = []
            upSampleConvSequence.append(Upsample(scale_factor=stride))
            upSampleConvSequence.append(nn.Conv2d(in_channels=in_dimension, out_channels=out_dimension,
                                                  kernel_size=kernel_size, stride=1, padding=padding))
            self.deconv = nn.Sequential(*upSampleConvSequence)
            if init_para:
                nn.init.normal_(self.deconv[1].weight.data, 0.0, 0.02)

    def forward(self, x):
        x = self.deconv(x)
        x = self.ins(x)
        if self.ACTIAVTE: x = func.relu(x, inplace=True)  # Activation
        return x


# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L25
# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, act=apply_res_act, shortcut_condition=True):
        super(ResBlock, self).__init__()
        # self.in_channels, self.out_channels = in_channels, out_channels
        self.ACTIVATE = act
        self.SHORTCUT = shortcut_condition and use_shortcut

        self.pad = nn.ReflectionPad2d(1)
        self.block1 = ConvBlock(in_channels, out_channels, kernel_size=3)
        self.block2 = ConvBlock(out_channels, out_channels, kernel_size=3, act=False, norm=True)
        self.shortcut = nn.Dropout(0.5)  # nn.Identity()

    def forward(self, x):
        residual = self.pad(x)
        residual = self.block1(residual)  # No resolution change
        if self.SHORTCUT:
            residual = self.shortcut(residual)
            if normalize_dropout:
                residual = residual * self.shortcut.p
        residual = self.pad(residual)
        residual = self.block2(residual)  # No resolution change
        x = x + residual
        if self.ACTIVATE: x = func.relu(x, inplace=True)
        return x


class GridNet(nn.Module):
    def __init__(self, resolution=256):
        super(GridNet, self).__init__()

        self.pad = nn.ReflectionPad2d(3)  # padding up to resolution
        self.tanh = nn.Tanh()  # output activation function
        resolution = int(resolution / 4)

        self.conv1 = ConvBlock(3, resolution, 7)  # 256x256
        self.conv2 = ConvBlock(resolution, resolution*2, 3, 2, 1)  # 128x128
        self.conv3 = ConvBlock(resolution*2, resolution*4, 3, 2, 1)  # 64x64
        resBlockSequence = []
        for i in range(res_block_depth):
            resBlockSequence.append(ResBlock(resolution * 4, resolution * 4))  # Reduce artifact, less retain shape
        self.res = nn.Sequential(*resBlockSequence)  # 64x64

        convBlockSequence4 = []
        if new_GN:
            # for i in range(4): convBlockSequence4.append(ConvBlock(resolution*4, resolution*4, 3, 1, 0))
            # for i in range(2): convBlockSequence4.append(ResBlock(resolution * 4, resolution * 4))
            # for i in range(4): convBlockSequence4.append(ConvBlock(resolution*4, resolution*4, 3, 1, 0))
            # for i in range(4): convBlockSequence4.append(ConvBlock(resolution*4, resolution*4, 5, 1, 0))
            # for i in range(2): convBlockSequence4.append(ConvBlock(resolution*4, resolution*4, 9, 1, 0))
            for i in range(1): convBlockSequence4.append(ConvBlock(resolution*4, resolution*4, 17, 1, 0))
        else:
            for i in range(8): convBlockSequence4.append(ConvBlock(resolution*4, resolution*4, 3, 1, 0))
            # convBlockSequence1.append(ConvBlock(resolution*4, resolution*4, 3, 1, 0))
        self.conv4 = nn.Sequential(*convBlockSequence4)  # 48x48

        convBlockSequence5 = []
        if new_GN:
            # for i in range(4): convBlockSequence5.append(ConvBlock(resolution*4, resolution*4, 3, 1, 0))
            # for i in range(2): convBlockSequence5.append(ResBlock(resolution * 4, resolution * 4))
            # for i in range(4): convBlockSequence5.append(ConvBlock(resolution*4, resolution*4, 3, 1, 0))
            # for i in range(4): convBlockSequence5.append(ConvBlock(resolution*4, resolution*4, 5, 1, 0))
            # for i in range(2): convBlockSequence5.append(ConvBlock(resolution*4, resolution*4, 9, 1, 0))
            for i in range(1): convBlockSequence5.append(ConvBlock(resolution*4, resolution*4, 17, 1, 0))
        else:
            for i in range(8): convBlockSequence5.append(ConvBlock(resolution*4, resolution*4, 3, 1, 0))
            # convBlockSequence2.append(ConvBlock(resolution*4, resolution*4, 3, 1, 0))
        self.conv5 = nn.Sequential(*convBlockSequence5)  # 32x32

        outputConvDepth = int(resolution * 4)  # 64
        OutputBlockSequence64 = []
        if new_model:
            for i in range(new_model_output_block_length): OutputBlockSequence64.append(ConvBlock(resolution * 4, outputConvDepth, 3, 1, 1))
        OutputBlockSequence64.append(nn.ReflectionPad2d(3))
        OutputBlockSequence64.append(ConvBlock(outputConvDepth, 3, 7, 1, 0, act=False, norm=False))
        OutputBlockSequence64.append(nn.Tanh())
        self.convOutput64 = nn.Sequential(*OutputBlockSequence64)
        OutputBlockSequence48 = []
        if new_model:
            for i in range(new_model_output_block_length): OutputBlockSequence48.append(ConvBlock(resolution * 4, outputConvDepth, 3, 1, 1))
        OutputBlockSequence48.append(nn.ReflectionPad2d(3))
        OutputBlockSequence48.append(ConvBlock(outputConvDepth, 3, 7, 1, 0, act=False, norm=False))
        OutputBlockSequence48.append(nn.Tanh())
        self.convOutput48 = nn.Sequential(*OutputBlockSequence48)
        OutputBlockSequence32 = []
        if new_model:
            for i in range(new_model_output_block_length): OutputBlockSequence32.append(ConvBlock(resolution * 4, outputConvDepth, 3, 1, 1))
        OutputBlockSequence32.append(nn.ReflectionPad2d(3))
        OutputBlockSequence32.append(ConvBlock(outputConvDepth, 3, 7, 1, 0, act=False, norm=False))
        OutputBlockSequence32.append(nn.Tanh())
        self.convOutput32 = nn.Sequential(*OutputBlockSequence32)
        # OutputBlockSequence64 = []
        # for i in range(1): OutputBlockSequence64.append(ConvBlock(resolution * 4, resolution * 4, 3, 1, 1))
        # OutputBlockSequence64.append(nn.ReflectionPad2d(3))
        # OutputBlockSequence64.append(ConvBlock(resolution * 4, 3, 7, 1, 0, False, False))
        # OutputBlockSequence64.append(nn.Tanh())
        # self.convOutput64 = nn.Sequential(*OutputBlockSequence64)
        # self.convOutput64 = ConvBlock(resolution*4, 3, 7, 1, 0, False, False)
        # self.convOutput48 = ConvBlock(resolution*4, 3, 7, 1, 0, False, False)
        # self.convOutput32 = ConvBlock(resolution*4, 3, 7, 1, 0, False, False)  # Don't normalize before output!!!

    def forward(self, x):
        if detach_network:
            x = x.detach()
        # Due to mirror loss, conv1, conv2, conv3 should be the same as DN (F), idk from last
        x_256 = self.conv1(self.pad(x))
        x_128 = self.conv2(x_256)
        x_64 = self.conv3(x_128)

        x_64_res = self.res(x_64)
        x_48_res = self.conv4(x_64_res)
        x_32_res = self.conv5(x_48_res)

        o_64 = self.convOutput64(x_64_res)
        o_48 = self.convOutput48(x_48_res)
        o_32 = self.convOutput32(x_32_res)
        # o_64 = self.tanh(self.convOutput64(self.pad(x_64_res)))
        # o_48 = self.tanh(self.convOutput48(self.pad(x_48_res)))
        # o_32 = self.tanh(self.convOutput32(self.pad(x_32_res)))   # 7x7 Conv-Ins-Relu layer with ref padding

        return (x_256, x_128, x_64), o_64, o_48, o_32


class PixelNet(nn.Module):
    def __init__(self, resolution=256, input_channel=3, output_channel=3):
        super(PixelNet, self).__init__()

        self.pad = nn.ReflectionPad2d(3)
        self.tanh = nn.Tanh()
        resolution = int(resolution / 4)

        self.conv1 = ConvBlock(input_channel, resolution, 7)  # 256x256
        self.conv2 = ConvBlock(resolution, resolution*2, 3, 2, 1)  # 128x128
        self.conv3 = ConvBlock(resolution*2, resolution*4, 3, 2, 1)  # 64x64
        resBlockSequence = []
        for i in range(res_block_depth): resBlockSequence.append(ResBlock(resolution*4, resolution*4))  # Reduce artifact, less retain shape
        self.res = nn.Sequential(*resBlockSequence)  # 64x64
        self.deconv1 = DeConvBlock(resolution*4, resolution*2, 3, 2, 1, 1)  # 128x128, tune kernel size!!
        self.deconv2 = DeConvBlock(resolution*2, resolution, 3, 2, 1, 1)  # 256x256

        if new_model:
            OutputBlockSequence = []
            for i in range(new_model_output_block_length): OutputBlockSequence.append(ConvBlock(resolution, resolution, 3, 1, 1))
            OutputBlockSequence.append(nn.ReflectionPad2d(3))
            OutputBlockSequence.append(ConvBlock(resolution, output_channel, 7, 1, 0, act=False, norm=False))
            OutputBlockSequence.append(nn.Tanh())
            self.convOutput = nn.Sequential(*OutputBlockSequence)
        else:
            self.convOutput = ConvBlock(resolution, 3, 7, 1, 0, False, False)

    def forward(self, x):
        if detach_network:
            x = x.detach()
        x_256 = self.conv1(self.pad(x))
        x_128 = self.conv2(x_256)
        x_64 = self.conv3(x_128)
        x_64_res = self.res(x_64)
        x_128_res = self.deconv1(x_64_res)
        x_256_res = self.deconv2(x_128_res)
        if new_model:
            o_256 = self.convOutput(x_256_res)
        else:
            o_256 = self.tanh(self.convOutput(self.pad(x_256_res)))
        return (x_256, x_128, x_64, x_64_res, x_128_res, x_256_res), o_256


class DePixelNet(PixelNet):
    def __init__(self, resolution=256):
        super(DePixelNet, self).__init__(resolution)


class ImageBuffer():
    def __init__(self, buffer_size=default_buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def size(self):
        return len(self.buffer)

    def _get_image_from_buffer(self, image):
        # https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
        # https://cloud.tencent.com/developer/article/1750730
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(image)
            return image
        else:
            if np.random.rand() < 0.5:
                return image
            else:
                index = np.random.randint(len(self.buffer) - 1)
                new_image = self.buffer.pop(index).clone()
                self.buffer.append(image)
                return new_image

    def get_image(self, images):
        if not isinstance(images, list):
            images = [images]
        outputs = []

        if not use_buffer:
            return torch.cat(images)

        for image in images:
            output = self._get_image_from_buffer(image)
            outputs.append(output)

        return torch.cat(outputs)


# https://clay-atlas.com/blog/2020/01/09/pytorch-chinese-tutorial-mnist-generator-discriminator-mnist/
class Discriminator(nn.Module):
    def __init__(self, resolution, in_channels=3):
        super(Discriminator, self).__init__()
        features = int(resolution / 4)

        kernel_size = 4
        out_stride = 1

        self.layers = nn.Sequential(
            # 3 * 256 * 256
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, stride=2, padding=1),  # 128x128
            # nn.InstanceNorm2d(resolution, affine=norm_affine, track_running_stats=norm_track),  # Why no need relu??? Why use 4x4 kernel???
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features*2, kernel_size, 2, 1),  # 64x64
            nn.InstanceNorm2d(features*2, affine=norm_affine, track_running_stats=norm_track),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*2, features*4, kernel_size, 2, 1),  # 32x32
            nn.InstanceNorm2d(features*4, affine=norm_affine, track_running_stats=norm_track),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*4, features*8, kernel_size, out_stride, 1),  # 16x16
            nn.InstanceNorm2d(features*8, affine=norm_affine, track_running_stats=norm_track),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*8, 1, kernel_size, out_stride, 1),  # 8x8
            nn.Sigmoid(),
        )

        for module in self.layers:
            if module.__class__.__name__.find('Conv2d') != -1:
                nn.init.normal_(module.weight.data, 0.0, 0.2)

    def forward(self, x):
        # noise = 0.0
        # if from_data or add_noise: noise = create_noise()
        label = self.layers(x)
        if use_label_smoothing:
            torch.clamp(label, 0.01, 0.99)

        return label


def create_lr_scheduler(optimizer, start_step, end_step):  #  -> torch.optim.lr_scheduler.LambdaLR
    def lr_lambda(epoch):
        if epoch <= start_step:
            return 1
        else:
            return 1 - (epoch - start_step) / (end_step - start_step)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# Loss Section
class ImageGradientNet(nn.Module):
    # https://medium.com/@anumolcs1996/image-gradient-for-edge-detection-in-pytorch-a9498a7827d6
    def __init__(self):
        super(ImageGradientNet, self).__init__()
        block_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        cluster_x = [block_x, block_x, block_x]
        kernel_x = torch.FloatTensor([cluster_x, cluster_x, cluster_x]).to(device)
        block_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        cluster_y = [block_y, block_y, block_y]
        kernel_y = torch.FloatTensor([cluster_y, cluster_y, cluster_y]).to(device)

        self.conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(data=kernel_x, requires_grad=False)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        return grad_x, grad_y


image_gradient_net = ImageGradientNet().to(device)
l1Loss = nn.L1Loss(reduction='mean')  # or reduction='sum'
if use_lsgan:
    GANLoss = nn.MSELoss()
else:
    GANLoss = nn.BCELoss()


def l1_criterion(image_generated, image_target, weight=1.0):
    return l1Loss(image_generated, image_target) * weight * total_weight


def adversarial_criterion(label, not_generated=True, weight=1.0):
    # Separate for real and fake
    if not_generated: target_label = torch.ones_like(label)
    else: target_label = torch.zeros_like(label)

    output = GANLoss(label, target_label)

    # output = 0.0
    # for label in labels:
    #     output = output + loss(label, target_label)
    # return output / len(labels) * weight
    return output * weight


def get_variation(image):
    # batch_size, num_of_channels, width, height = image.size()
    variation_x = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    variation_y = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])

    return variation_x, variation_y


# add loss for variation / very different stripes???


def gradient_criterion(image_generated, image_target, weight=1.0, variation=use_variation):

    if not variation:
        image_generated_x, image_generated_y = image_gradient_net(image_generated)
        image_target_x, image_target_y = image_gradient_net(image_target)
    else:
        image_generated_x, image_generated_y = get_variation(image_generated)
        image_target_x, image_target_y = get_variation(image_target)

    output = l1Loss(image_generated_x, image_target_x) + l1Loss(image_generated_y, image_target_y) * 0.5

    return output * weight


def image_mirror_criterion(image_generated, image_target, weight=1.0):
    return l1Loss(image_generated, image_target) * weight * total_weight


def network_mirror_criterion(net_input_output, net_target_output, weight=1.0):
    output = 0.0
    length = len(net_target_output)

    # calculate loss on each layer of networks between GN and DN / DN and PN in forward in reverse order
    for i in range(length):
        # match resolution and output feature dimension!!!
        output = output + l1Loss(net_input_output[i], net_target_output[i])
    output = output / length

    return output * weight * total_weight


def null_pixel_loss(image_generated, image_target, weight=1.0):
    threshold = -1 + 5/255
    loss = nn.L1Loss(reduction='mean')

    index_target = (image_target.detach() < threshold).float()

    null_generated = image_generated * index_target
    null_target = image_target * index_target

    return loss(null_generated, null_target) * weight


def get_image_gradient(input_image, normalize=True):
    x, y = image_gradient_net(input_image)
    if normalize:
        x = data.normalize(x, -1, 1)
        y = data.normalize(y, -1, 1)
    return x, y


def combine_sobel_image(gradient_x, gradient_y):
    gradient = torch.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    gradient_normalized = data.normalize(gradient)
    return gradient_normalized


def get_sobel_image(input_image):
    x, y = image_gradient_net(input_image)
    return combine_sobel_image(x, y)
