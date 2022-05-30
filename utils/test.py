# https://clay-atlas.com/blog/2020/01/09/pytorch-chinese-tutorial-mnist-generator-discriminator-mnist/
import functools

import torch
from torch import nn
import torch.nn.functional as functional
import torchvision
import matplotlib.pyplot as plt
from os.path import exists
import sys

import numpy as np

import architecture as arch
import data

gan_d_interval = 32
for i in range(900):
    print((((i + 1) % int(gan_d_interval)) == 0) or i == 0)
exit()


# print(data.choose_random(([1, 1], [1, 2], [3, 1])))
print(arch.adversarial_criterion(torch.ones(1, 3, 64, 64)*0.5, True, 1.0))
# print(arch.adversarial_criterion(torch.ones(1, 3, 256, 256)*0.25, True, 0.5))
# print(arch.adversarial_criterion(torch.ones(1, 3, 256, 256)*0.5, False))
# print(arch.adversarial_criterion(torch.ones(1, 3, 64, 64)*0.5, True))
# print(arch.adversarial_criterion(torch.ones(1, 3, 48, 48)*0.5, False))
# print(arch.adversarial_criterion(torch.ones(1, 3, 32, 32)*0.5, False))
exit()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


convBlockSequence4 = []
for i in range(8): convBlockSequence4.append(arch.ConvBlock(64 * 4, 64 * 4, 3, 1, 0))
# for i in range(4): convBlockSequence4.append(arch.ConvBlock(64 * 4, 64 * 4, 5, 1, 0))
# for i in range(2): convBlockSequence4.append(arch.ConvBlock(64 * 4, 64 * 4, 9, 1, 0))
# for i in range(1): convBlockSequence4.append(arch.ConvBlock(64 * 4, 64 * 4, 17, 1, 0))
conv4 = nn.Sequential(*convBlockSequence4)  # 48x48
print(count_parameters(conv4))
exit()
model = arch.GridNet()
print(conv4(torch.randn(1, 64*4, 64, 64)).size())
print(conv4(torch.randn(1, 64*4, 48, 48)).size())
exit()

# sample = torch.randn(1, 3, 32, 32)
# # model = torch.nn.Conv2d(3, 16, 3, bias=False)
# model = torch.nn.ConvTranspose2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
# output = model(sample)
# print(output.size())

# a = [256, 128, 64]
# b = [256*4, 128*4, 64*4, 64, 128, 256]
#
# print((a[0], a[1], a[2]))
# print(b[-1], b[-2], b[-3])
# exit()

test = torch.randn(1, 3, 4, 4, requires_grad=True)
test_processed = data.desample(test, scale=2.0, distribute_grad=True)
arch.l1_criterion(test_processed, torch.ones_like(test_processed)).backward()
print(test.grad)
test_small = torch.randn(1, 3, 4, 4, requires_grad=True)
test_small_resize = data.resize(test_small, scale=16.0/3.0)
arch.l1_criterion(test_small_resize, torch.ones_like(test_small_resize)).backward()
print(test_small_resize.size())
print(test_small.grad)
exit()

a = (torch.ones(1, 1, 1, 1, requires_grad=True)*0.0, torch.ones(1, 1, 1, 1, requires_grad=True)*0.0, torch.ones(1, 1, 1, 1, requires_grad=True)*0.0)
print(data.choose_min(a))
exit()

print(torch.cat((torch.ones(1, 1, 1, 1, requires_grad=True), torch.ones(1, 1, 1, 1, requires_grad=True)), dim=1).detach())
exit()

# checkpoint = torch.load('backup/self-learning/new/test i1d2 more gan effect/checkpoint22_GN.pth', map_location=torch.device('cpu'))
# print(checkpoint.keys())
# exit()

num_epochs = 100
def lr_lambda(epoch):
    if epoch <= num_epochs:
        return 1
    else:
        return 1 - (epoch - 100) / (num_epochs - 100)

for i in range(num_epochs):
    print(f'{i+1}: {lr_lambda(i+1)}')
exit()


# buffer = arch.ImageBuffer()
# for i in range(60):
#     print(buffer.get_image(torch.ones((1, 1, 1, 1)) * i))
#     print(buffer.size())
# exit()


# Initial Variables
GN_option: arch.CombinedOptions = arch.CombinedOptions('GN')
PN_option: arch.CombinedOptions = arch.CombinedOptions('PN')
DN_option: arch.CombinedOptions = arch.CombinedOptions('DN')

current_epochs = 0
current_iter = 0
num_epochs = 0

learning_rate = 0.0002
learning_rate_discriminator = learning_rate
weight_decay = 0

version = ''
baseline_flag = ''
use_buffer = True
save_plot = False
save_image = False
use_tensorboard = True
use_lr_scheduler = True
save_model = True
test_model = True
override_state_dict = False
clear_buffer = False
use_floodfill = False
cartoon_resize_mode = 'resize'
pixel_resize_mode = 'resize'

detach_network = True
train_GAN_mirror = False

if len(version) > 0:
    if int(version) >= 5:
        data.num_of_batch = 1
        print(f'Batch Size: {data.num_of_batch}')
        arch.new_model = False
        print(f'New Model: {arch.new_model}')
        arch.norm_track = True
        print(f'Normalisation Track: {arch.norm_track}')
        # weight_decay = 0.0001  # 1e-3
        # print(f'Weight Decay: {weight_decay}')
        override_state_dict = True
        print(f'Override State Dictionary: {override_state_dict}')
        return_downsample_GN = True
        print(f'GN Return DownSample: {return_downsample_GN}')
        arch.use_label_smoothing = False
        print(f'Label Smoothing: {arch.use_label_smoothing}')
        data.extra_transform_data = False
        print(f'Data use extra transforms: {data.extra_transform_data}')
        arch.norm_affine = False
        print(f'Normalisation Affine: {arch.norm_affine}')
        combine_backward = False
        print(f'Combine Backward: {combine_backward}')
        use_floodfill = True
        PN_backward_mode = 'random'
        print(f'PN Backward Mode: {PN_backward_mode}')
        train_backward = True
        print(f'Train Backward: {train_backward}')
        arch.use_lsgan = False
        print(f'LSGAN: {arch.use_lsgan}')
        arch.apply_res_act = False
        print(f'Res Act: {arch.apply_res_act}')
        arch.normalize_dropout = False
        print(f'Dropout Normalize: {arch.normalize_dropout}')
        arch.use_shortcut = True
        return_downsample_GN = False
        print(f'GN Return DownSample: {return_downsample_GN}')
        arch.use_variation = False
        print(f'Use Variation for gradient loss: {arch.use_variation}')

        cartoon_resize_mode = 'mixed'
        pixel_resize_mode = 'resize'

        data.max_epoch_size = 50

        GN_option.train = True
        PN_option.train = True
        DN_option.train = False

        GN_option.weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
        PN_option.weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
        DN_option.weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0*20, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)

        GN_option.set_options(l1=True, grad=True, gan=False, image=False, network=False, idt=False)
        PN_option.set_options(l1=True, grad=True, gan=False, image=False, network=False, idt=False)
        DN_option.set_options(l1=True, grad=True, gan=False, image=False, network=False, idt=False)
        GN_option.set_interval(1, 1)
        PN_option.set_interval(1, 1)
        DN_option.set_interval(1, 1)

        arch.use_lsgan = True
        # detach_network = False
        # train_GAN_mirror = True
        use_lr_scheduler = num_epochs >= 75 or True
        save_model = data.max_epoch_size >= 900

arch.print_weights()

# Buffer Section
loss_buffer = arch.LossBuffer()

print()
print('Current Epoch = ' + str(current_epochs))
print(f'Current Iterations = {current_iter}')
print(f'Epochs = {num_epochs}')
print(f'lr = {learning_rate}')
print(f'Weight Decay: {weight_decay}')
print(f'Version = \'{version}\'')
print(f'Baseline Flag = \'{baseline_flag}\'')
print(f'Use Buffer = {use_buffer}')
print(f'Load GN = {GN_option.load_generator}, {GN_option.load_discriminator}')
print(f'Load PN = {PN_option.load_generator}, {PN_option.load_discriminator}')
print(f'Load DN = {DN_option.load_generator}, {DN_option.load_discriminator}')
print(f'Use lr_scheduler: {use_lr_scheduler}')
print(f'Save Model = {save_model}')
print(f'Test Model = {test_model}')
print(f'Use Flood Fill: {use_floodfill}')
print(f'Cartoon Resize Mode: {cartoon_resize_mode}')
print(f'Pixel Resize Mode: {pixel_resize_mode}')
print(f'Detach Network: {detach_network}')
GN_option.print_info()
PN_option.print_info()
DN_option.print_info()

exit()

def function_A(gg, x):
    return gg(x) * 2
def abc(x):
    return x * 3
print(function_A(abc, 3))
exit()

# test = torch.randn(1, 3, 256, 256)
# conv = torch.nn.Conv2d(3, 3, 3, 1, 1)
# abc = arch.Network(model=conv, tag='PN')
# print(abc(test))
# print(conv(test))
#
# exit()

test = torch.randn(1, 3, 4, 4, requires_grad=True)
test_b = torch.randn(1, 3, 4, 4, requires_grad=False)
print(torch.cat((test, test_b), dim=1).size())

exit()


test = torch.randn(1, 3, 4, 4, requires_grad=True)
print(test)
print(test.detach())
print(test.data)
exit()

test = torch.randn(1, 3, 256, 256)
conv = torch.nn.Conv2d(3, 3, 3, 1, 1)

target = torch.randn(1, 3, 256, 256)
output = conv(test)
label = conv(torch.ones_like(target) * 0.5)
label = data.normalize(label, 0, 1)

# arch.l1_criterion(output, target).backward()
# arch.gradient_criterion(output, target).backward()
# arch.image_mirror_criterion(output, target).backward()
# arch.network_mirror_criterion((output, output, output), (target, target, target)).backward()
# arch.adversarial_criterion(label).backward()
print(conv.weight.grad)

output_not_detach = (output, output, output)
output = output.detach()
output_detach = (output, output, output)
print(output_not_detach)
print(output_detach)

exit()


class NetworkOption:
    def __init__(self):
        self.create = True
        self.load_generator = True
        self.load_discriminator = True
        self.save = True

        self.l1 = True
        self.grad = True
        self.gan = True
        self.gan_interval = 1
        self.mirror_image = True
        self.mirror_network = True

        self.separate_weights = []

    def reset(self):
        self.create = True
        self.load_generator = True
        self.load_discriminator = True
        self.save = True

        self.l1 = True
        self.grad = True
        self.gan = True
        self.gan_interval = 1
        self.mirror_image = True
        self.mirror_network = True

        self.separate_weights = []

    def print(self):
        print(self.__dict__)

NetworkOption().print()

exit()

NETWORK_TAG = 'GN_'
a = NETWORK_TAG + 'generator'
b = 'GN_generator'
print(a)
print(b)
print(a == b)
exit()

x = 1
y = None
z = {}
print(x is not None)
print(y is not None)
if z is not None:
    z['gg'] = y
print(z)

exit()


for i in range(10):
    print(np.random.rand() < 0.5)
exit()


class ResnetBlock(nn.Module):
    def __init__(self, channel, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(channel, use_dropout)

    def build_conv_block(self, channel, use_dropout):
        conv_block = []
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(channel, channel, kernel_size=3, padding=0, bias=True),
                       norm_layer(channel, track_running_stats=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(channel, channel, kernel_size=3, padding=0, bias=True),
                       norm_layer(channel, track_running_stats=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class GridNet_Network(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, gpu_ids=[]):
        super(GridNet_Network, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        # model for downsampling
        model_0 = [nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                             bias=True),
                   norm_layer(ngf, track_running_stats=True),
                   nn.ReLU(True)]

        mult = 1
        model_1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                             stride=2, padding=1, bias=True),
                   norm_layer(ngf * mult * 2, track_running_stats=True),
                   nn.ReLU(True)]

        mult = 2
        model_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                             stride=2, padding=1, bias=True),
                   norm_layer(ngf * mult * 2, track_running_stats=True),
                   nn.ReLU(True)]

        mult = 4
        model_3 = []
        for i in range(9):
            model_3 += [ResnetBlock(ngf * mult, ngf * mult)]

        model_4 = []
        for i in range(8):
            model_4 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                  stride=1, padding=0, bias=True),
                        norm_layer(ngf * mult, track_running_stats=True),
                        nn.ReLU(True)]

        model_5 = []
        for i in range(8):
            model_5 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                  stride=1, padding=0, bias=True),
                        norm_layer(ngf * mult, track_running_stats=True),
                        nn.ReLU(True)]

        model_output_64 = [nn.ReflectionPad2d(3),
                           nn.Conv2d(ngf * mult, output_nc, kernel_size=7, padding=0),
                           nn.Tanh()]

        model_output_48 = [nn.ReflectionPad2d(3),
                           nn.Conv2d(ngf * mult, output_nc, kernel_size=7, padding=0),
                           nn.Tanh()]

        model_output_32 = [nn.ReflectionPad2d(3),
                           nn.Conv2d(ngf * mult, output_nc, kernel_size=7, padding=0),
                           nn.Tanh()]

        self.model_0 = nn.Sequential(*model_0)
        self.model_1 = nn.Sequential(*model_1)
        self.model_2 = nn.Sequential(*model_2)
        self.model_3 = nn.Sequential(*model_3)
        self.model_4 = nn.Sequential(*model_4)
        self.model_5 = nn.Sequential(*model_5)
        self.model_output_32 = nn.Sequential(*model_output_32)
        self.model_output_48 = nn.Sequential(*model_output_48)
        self.model_output_64 = nn.Sequential(*model_output_64)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            feature_256 = nn.parallel.data_parallel(self.model_0, input, self.gpu_ids)
            feature_128 = nn.parallel.data_parallel(self.model_1, feature_256, self.gpu_ids)
            feature_64 = nn.parallel.data_parallel(self.model_2, feature_128, self.gpu_ids)
            output_for_64 = nn.parallel.data_parallel(self.model_3, feature_64, self.gpu_ids)
            output_for_48 = nn.parallel.data_parallel(self.model_4, output_for_64, self.gpu_ids)
            output_for_32 = nn.parallel.data_parallel(self.model_5, output_for_48, self.gpu_ids)

            output_64 = nn.parallel.data_parallel(self.model_output_64, output_for_64, self.gpu_ids)
            output_48 = nn.parallel.data_parallel(self.model_output_48, output_for_48, self.gpu_ids)
            output_32 = nn.parallel.data_parallel(self.model_output_32, output_for_32, self.gpu_ids)

        else:
            feature_256 = self.model_0(input)
            feature_128 = self.model_1(feature_256)
            feature_64 = self.model_2(feature_128)
            output_for_64 = self.model_3(feature_64)
            output_for_48 = self.model_4(output_for_64)
            output_for_32 = self.model_5(output_for_48)

            output_64 = self.model_output_64(output_for_64)
            output_48 = self.model_output_48(output_for_48)
            output_32 = self.model_output_32(output_for_32)

        return feature_256, feature_128, feature_64, output_32, output_48, output_64

_, _, _, a, b, c = GridNet_Network(3, 3)(torch.randn(1, 3, 256, 256))
print(a.size())
print(b.size())
print(c.size())
exit()


GANLoss = nn.BCELoss(reduction='mean')

def adversarial_criterion(label, not_generated=True, weight=1.0):
    # Separate for real and fake
    if not_generated: target_label = torch.ones_like(label)
    else: target_label = torch.zeros_like(label)
    output = GANLoss(label, target_label)
    return output * weight

print(arch.adversarial_criterion(torch.ones(1, 1, 8, 8) * 0.5, not_generated=False) / 20)
exit()

test_t = torch.randn(1, 3, 256, 256)
test_n = data.normalize(test_t)
print((test_t.max(), test_t.min()))
print((test_n.max(), test_n.min()))
exit()

d = arch.FullSizeDiscriminator(64)
print(d(torch.rand(1, 3, 256, 256)).size())
a, b = arch.image_gradient_net(torch.randn(1, 3, 256, 256))
print(a.size())
print(b.size())
exit()


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

d = Discriminator()
print(d(torch.randn(8, 3, 64, 64)).size())
exit()


print(torch.ones(1, 1, 4, 4))
print(torch.nn.Dropout(0.5)(torch.ones(1, 1, 4, 4)))
print(torch.nn.Dropout(0.5).p)
test = torch.randn(1, 1, 4, 4)
print(test)
print(torch.nn.Identity()(test))
exit()

deconv_input = torch.ones(1, 3, 16, 16)
upSampleConvSequence = []
upSampleConvSequence.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
upSampleConvSequence.append(torch.nn.Conv2d(in_channels=3, out_channels=8,
                                      kernel_size=3, stride=1, padding=1))
deconv = torch.nn.Sequential(*upSampleConvSequence)

print(torch.nn.init.normal_(deconv[1].weight.data, 0.0, 0.02))
print(deconv[1].weight)

print(deconv(deconv_input).size())
print(torch.nn.Dropout(0.5)(torch.ones(1, 1, 4, 4)))

exit()

def rand_int():
    return np.random.randint(10) - 5

print(rand_int())

exit()

i = 10
n_total_steps = 900
epoch = 0
print((i + 1) % int(n_total_steps) and (epoch % 2 == 0))

def train_GN(input_image, forward=True, require_print=False, require_save=False, train_D=True, train_G=True):
    print(f'{input_image}, {forward}, {require_print}, {require_save}, {train_D}, {train_G}')

B_DN_G_image = False
addScalarCondition = True
saveCondition = True
train_GN(B_DN_G_image, False, addScalarCondition, saveCondition, train_D=False)

exit()

buffer = arch.ImageBuffer(2)
image = torch.ones(1, 1, 4, 4, requires_grad=True)
buffered = buffer.get_image([image.detach(), image * 1.1, image * 1.2])
print(image)
print(buffered)
print(buffer.buffer)
print(buffer.size())

labels = [torch.ones(1, 1, 4, 4) * 0.8, torch.ones(1, 1, 4, 4) * 0.9]
# print(torch.cat(labels).size())
# print(torch.cat(labels).mean())
# print(torch.nn.L1Loss()(torch.stack(labels), torch.stack(labels)))
# print(torch.nn.BCELoss()(torch.stack(labels), torch.stack(labels)).data)
# torch.nn.L1Loss()(buffered, labels[1]).backward()
# print(image.grad)
exit()

count = [0, 0, 0]
total = (1, 2, 3)
for i in range(100):
    r = data.choose_random(total)
    if r == total[0]:
        count[0] += 1
    elif r == total[1]:
        count[1] += 1
    elif  r == total[2]:
        count[2] += 1
print(count)
exit()

test_model = torch.nn.Conv2d(3, 3, 3)
test_optimizer = torch.optim.Adam(test_model.parameters(), lr=1, betas=(0.5, 0.999), weight_decay=2)
print(test_model)
print(test_optimizer)
print(test_optimizer.param_groups[0]['lr'])
print(test_optimizer.param_groups[0]['betas'])
print(test_optimizer.param_groups[0]['weight_decay'])
test_optimizer.param_groups[0]['betas'] = (0.2, 0.99)
test_optimizer.param_groups[0]['weight_decay'] = 0.2
print(test_optimizer.state_dict())
exit()


ins = torch.nn.InstanceNorm2d(num_features=2, affine=True, track_running_stats=True)
ins_no_grad = torch.nn.InstanceNorm2d(num_features=2, affine=False, track_running_stats=False)

g = torch.optim.Adam(ins.parameters(), lr=1.0, betas=(0.5, 0.999), weight_decay=0.0)
# gg = torch.optim.Adam(ins_no_grad.parameters(), lr=1.0, betas=(0.5, 0.999), weight_decay=0.0)

print(ins.affine)
print(ins.track_running_stats)
print(ins.bias)
print(ins.parameters())
print(ins)
print(ins.weight)
print(ins.eps)
print(ins.momentum)
print(ins.T_destination)
print(ins.__dict__)
print()
print(ins_no_grad.affine)
print(ins_no_grad.track_running_stats)
print(ins_no_grad.bias)
print(ins_no_grad.parameters())
print(ins_no_grad)
print(ins_no_grad.weight)
print(ins_no_grad.eps)
print(ins_no_grad.momentum)
print(ins_no_grad.T_destination)
print(ins_no_grad.__dict__)
print(ins_no_grad.buffers())
print()
ins.affine = False
ins.bias = None
ins.weight = None
ins.reset_parameters()
ins.reset_running_stats()
print(ins.__dict__)
print(ins.affine)
print(ins.track_running_stats)
print(ins.bias)
print(ins.weight)

print(g.param_groups)
# print(gg.param_groups)

exit()

# data.num_of_batch = 1
# data.num_of_cores = 1
# print(f'Batch Size: {data.num_of_batch}')
# arch.new_model = True
# print(f'New Model: {arch.new_model}')
# print(f'New Discriminator: {arch.new_discriminator}')
# arch.use_spectral = True
# print(f'Spectral Normalization: {arch.use_spectral}')
# weight_decay = 0.0001
# arch.GN_weights.reset()
# arch.PN_weights.reset()
# arch.DN_weights.reset()

# arch.total_weight = 100.0
# arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0)
# arch.PN_weights = arch.Weights(l1=1.0 / 20, grad=1.0, image=1.0 / 10, net=1.0 / 10, null=1.0, gan=1.0)
# arch.DN_weights = arch.Weights(l1=1.0 / 10, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0)
# train_baseline = False
# train_discriminator = True
# train_generator = True
# sync_discriminator = True
# train_frequency = [1, 1, 1]
# train_grad = True
# train_mirror_layer = True
# train_null = False
# arch.use_label_smoothing = False
# data.extra_transform_data = True
# print(f'Data use extra transforms: {data.extra_transform_data}')
# weight_decay = 0.0
# arch.new_discriminator = True
# print(f'New Model: {arch.new_discriminator}')

# Directory
prefix = 'demo/1/'
image_path = prefix + 'test.png'
# model_path = data.get_model_directory()
model_path = 'demo/' + ''
version = '4'  # 1==l1, 2==l1+GAN (3==l1+GRAD) 4==l1+GRAD+MNL 6==ALL
clamp_flag = 'False'
forward = True

if len(sys.argv) > 1: image_path = sys.argv[1]
if len(sys.argv) > 2: model_path = sys.argv[2]
if len(sys.argv) > 3: version = sys.argv[3]
if len(sys.argv) > 4: clamp_flag = sys.argv[4]

clamp = clamp_flag == 'True'
model_dir = model_path + 'checkpoint' + version + '.pth'
print(f'Model Directory: {model_dir}')

# Device Section
device = arch.get_device()

arch.new_model = True
print(f'New Model: {arch.new_model}')
arch.norm_track = True
print(f'Normalisation Track: {arch.norm_track}')
# arch.new_discriminator = True
print(f'New Discriminator: {arch.new_discriminator}')

# Model Section
GN_generator = arch.GridNet()
PN_generator = arch.PixelNet()
DN_generator = arch.PixelNet()

GN_generator = torch.nn.DataParallel(GN_generator).to(device)
PN_generator = torch.nn.DataParallel(PN_generator).to(device)
DN_generator = torch.nn.DataParallel(DN_generator).to(device)

if exists(model_dir):
    checkpoint = torch.load(model_dir)

    current_epochs = checkpoint['epoch']
    current_iter = checkpoint['iteration']

    GN_generator.load_state_dict(checkpoint['GN_generator'])
    PN_generator.load_state_dict(checkpoint['PN_generator'])
    DN_generator.load_state_dict(checkpoint['DN_generator'])

    checkpoint = {}  # save memory!!!
    print("Loaded model successfully.")

GN_generator.train()
PN_generator.train()
DN_generator.train()
# GN_generator.eval()
# PN_generator.eval()
# DN_generator.eval()

# Generator
image = plt.imread(image_path)
size_x, size_y = image.shape[0], image.shape[1]
image = image[:, :, :3]
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
image = functional.interpolate(image, size=(256, 256))

if forward:
    # Forward
    _, GN_64, GN_48, GN_32 = GN_generator(image)
    GN_64 = data.reSize(GN_64, 4.0)
    GN_48 = data.reSize(GN_48, 16.0/3.0)
    GN_32 = data.reSize(GN_32, 8.0)
    _, PN_64 = PN_generator(GN_64)
    _, PN_48 = PN_generator(GN_48)
    _, PN_32 = PN_generator(GN_32)
    _, DN_64 = DN_generator(PN_64)
    _, DN_48 = DN_generator(PN_48)
    _, DN_32 = DN_generator(PN_32)

    # if clamp:
    #     GN_64 = torch.clamp(GN_64, 0.0, 1.0)
    #     GN_48 = torch.clamp(GN_48, 0.0, 1.0)
    #     GN_32 = torch.clamp(GN_32, 0.0, 1.0)
    #     PN_64 = torch.clamp(PN_64, 0.0, 1.0)
    #     PN_48 = torch.clamp(PN_48, 0.0, 1.0)
    #     PN_32 = torch.clamp(PN_32, 0.0, 1.0)
    #     DN_64 = torch.clamp(DN_64, 0.0, 1.0)
    #     DN_48 = torch.clamp(DN_48, 0.0, 1.0)
    #     DN_32 = torch.clamp(DN_32, 0.0, 1.0)

    data.save_image(GN_64, 'GN_64', normalize=not clamp, directory=prefix + '')
    data.save_image(GN_48, 'GN_48', normalize=not clamp, directory=prefix + '')
    data.save_image(GN_32, 'GN_32', normalize=not clamp, directory=prefix + '')

    PN_64_resized = functional.interpolate(PN_64, size=(size_x, size_y))
    PN_48_resized = functional.interpolate(PN_48, size=(size_x, size_y))
    PN_32_resized = functional.interpolate(PN_32, size=(size_x, size_y))
    DN_64_resized = functional.interpolate(DN_64, size=(size_x, size_y))
    DN_48_resized = functional.interpolate(DN_48, size=(size_x, size_y))
    DN_32_resized = functional.interpolate(DN_32, size=(size_x, size_y))
    data.save_image(PN_64_resized, 'PN_64_resized', normalize=not clamp, directory=prefix + '')
    data.save_image(PN_48_resized, 'PN_48_resized', normalize=not clamp, directory=prefix + '')
    data.save_image(PN_32_resized, 'PN_32_resized', normalize=not clamp, directory=prefix + '')
    data.save_image(DN_64_resized, 'DN_64_resized', normalize=not clamp, directory=prefix + '')
    data.save_image(DN_48_resized, 'DN_48_resized', normalize=not clamp, directory=prefix + '')
    data.save_image(DN_32_resized, 'DN_32_resized', normalize=not clamp, directory=prefix + '')

else:
    # Backward
    _, B_DN = DN_generator(image)
    _, B_GN_64, B_GN_48, B_GN_32 = GN_generator(B_DN)
    B_GN_64 = data.reSize(B_GN_64, 4.0)
    B_GN_48 = data.reSize(B_GN_48, 16.0/3.0)
    B_GN_32 = data.reSize(B_GN_32, 8.0)
    _, B_PN_64 = PN_generator(B_GN_64)
    _, B_PN_48 = PN_generator(B_GN_48)
    _, B_PN_32 = PN_generator(B_GN_32)

    data.save_image(B_GN_64, 'B_GN_64', normalize=not clamp, directory=prefix + '')
    data.save_image(B_GN_48, 'B_GN_48', normalize=not clamp, directory=prefix + '')
    data.save_image(B_GN_32, 'B_GN_32', normalize=not clamp, directory=prefix + '')
    B_DN_resized = functional.interpolate(B_DN, size=(size_x, size_y))
    data.save_image(B_DN_resized, 'B_DN_resized', normalize=not clamp, directory=prefix + '')
    B_PN_64_resized = functional.interpolate(B_PN_64, size=(size_x, size_y))
    B_PN_48_resized = functional.interpolate(B_PN_48, size=(size_x, size_y))
    B_PN_32_resized = functional.interpolate(B_PN_32, size=(size_x, size_y))
    data.save_image(B_PN_64_resized, 'B_PN_64_resized', normalize=not clamp, directory=prefix + '')
    data.save_image(B_PN_48_resized, 'B_PN_48_resized', normalize=not clamp, directory=prefix + '')
    data.save_image(B_PN_32_resized, 'B_PN_32_resized', normalize=not clamp, directory=prefix + '')

print('Saved image successfully.')
