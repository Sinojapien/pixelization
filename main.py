import os.path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from os.path import exists

import architecture as arch
from architecture import data  # import data as data

import sys
import datetime

# Initial Variables
GN_option = arch.NetworkOption()
PN_option = arch.NetworkOption()
DN_option = arch.NetworkOption()

current_epochs = 0
current_iter = 0
num_epochs = 0

learning_rate = 0.0002
learning_rate_discriminator = learning_rate
weight_decay = 0

version = ''
baseline_flag = ''
save_plot = False
save_image = False
use_tensorboard = True
use_lr_scheduler = True
lr_scheduler_initial_step = 100
save_model = True
test_model = True
override_state_dict = False
clear_buffer = False
use_floodfill = False
cartoon_resize_mode = 'resize'
pixel_resize_mode = 'resize'

PN_backward_mode = 'min'
return_downsample_GN = False
downsample_as_PN_input_target = False
train_GAN_mirror = False
PN_target_scale = False
combine_loss = False
small_batch = False

# Handling input arguments
print(f'Arguments: {sys.argv}')
if len(sys.argv) > 1: data.update_path(sys.argv[1])  # path = '/data/d0/y19/chyeung9/fyp1/submit/'
if len(sys.argv) > 2: data.update_output_path(sys.argv[2])  # '/research/dept8/fyp21/ttw2104/pixelization/'
if len(sys.argv) > 3: num_epochs = int(sys.argv[3])
if len(sys.argv) > 4: version = sys.argv[4]

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
        arch.use_variation = False
        print(f'Use Variation for gradient loss: {arch.use_variation}')

        arch.GN_weights.reset()
        arch.PN_weights.reset()
        arch.DN_weights.reset()

        cartoon_resize_mode = 'mixed'
        pixel_resize_mode = 'mixed'

        if int(version) >= 10:
            GN_option.train = False
            PN_option.train = False
            DN_option.train = False
            use_floodfill = False

            arch.total_weight = 1.0
            # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
            # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
            # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)

            if int(version) == 10:
                arch.GN_weights.grad = 1.0 / 6
                PN_option.separate_weights = [1.0, 1.0, 0.75]
                GN_option.gan_d_interval = 4
                PN_option.gan_d_interval = 4
                DN_option.gan_d_interval = 1
                arch.use_upsample_conv = True
                print(f'Upsample Convolution: {arch.use_upsample_conv}')

            if int(version) >= 11:
                # arch.GN_weights = arch.Weights(l1=1.0*5, grad=1.0*5, image=1.0, net=1.0, null=1.0, gan=1.0/1.5, d_gan=1.0/1.5)
                # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*5, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
                # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*5, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
                # arch.GN_weights = arch.Weights(l1=1.0*2, grad=1.0*2, image=1.0, net=1.0, null=1.0, gan=1.0/1.25, d_gan=1.0/1.25)
                # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*20, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
                # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*20, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0, d_gan=1.0)
                # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*20, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*20, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
                # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0/4, image=1.0, net=1.0, null=1.0, gan=1.0, d_gan=1.0)
                # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*5, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
                # arch.DN_weights = arch.Weights(l1=1.0*1.1, grad=1.0, image=1.0*5, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0/4, image=1.0, net=1.0, null=1.0, gan=1.0, d_gan=1.0)
                # arch.PN_weights = arch.Weights(l1=1.0/2, grad=1.0/2, image=1.0*5/5, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
                # arch.DN_weights = arch.Weights(l1=1.0*1.5, grad=1.0, image=1.0*5, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)

                # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)  # set
                # arch.PN_weights = arch.Weights(l1=1.0*0.9/0.9, grad=1.0*0.9, image=1.0*3, net=1.0/10, null=1.0, gan=1.0/1.5, d_gan=1.0/1.5)  # more l1, less mirror
                # # arch.DN_weights = arch.Weights(l1=1.0/1.5, grad=1.0/1.5, image=1.0*20, net=1.0*10, null=1.0, gan=1.0/3, d_gan=1.0/3)
                # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0/10, null=1.0, gan=1.0/2, d_gan=1.0/2)  # set, not trained

                # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0/5, image=1.0, net=1.0, null=1.0, gan=1.0/10, d_gan=1.0/10)
                # # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0/3, null=1.0, gan=1.0/1.5, d_gan=1.0/1.5)
                # arch.PN_weights = arch.Weights(l1=1.0*1.5, grad=1.0*0.9, image=1.0, net=1.0, null=1.0, gan=1.0/1.25, d_gan=1.0/1.25)
                # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0/3, null=1.0, gan=1.0/2, d_gan=1.0/2)

                # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0, net=1.0, null=1.0, gan=1.0/10, d_gan=1.0/10)
                # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0/10, null=1.0, gan=1.0/5, d_gan=1.0/5)
                # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0/10, null=1.0, gan=1.0/15, d_gan=1.0/15)
                arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/10, d_gan=1.0/10/10, d_target=1.0/10/10)
                arch.PN_weights = arch.Weights(l1=1.0*4*2, grad=1.0*2/5, image=1.0, net=1.0/10, null=1.0, gan=1.0/4, d_gan=1.0/4/15, d_target=1.0/4/15)
                # arch.DN_weights = arch.Weights(l1=1.0*3, grad=1.0*4, image=1.0, net=1.0/10, null=1.0, gan=1.0/10, d_gan=1.0/10/10, d_target=1.0/10/10)
                arch.DN_weights = arch.Weights(l1=1.0*3, grad=1.0*4, image=1.0, net=1.0/10, null=1.0, gan=1.0/8, d_gan=1.0/8/10, d_target=1.0/8/10)
                GN_option.train = False
                PN_option.train = True
                DN_option.train = False
                GN_option.set_interval(g=1, d=1)  # 15
                PN_option.set_interval(g=1, d=1)  # 10
                DN_option.set_interval(g=1, d=1)

                # GN_option.set_options(l1=True, grad=True, gan=True, image=False, network=False, idt=False)
                # PN_option.set_options(l1=True, grad=True, gan=True, image=True, network=True, idt=False)
                # DN_option.set_options(l1=True, grad=True, gan=True, image=True, network=True, idt=False)
                GN_option.set_options(l1=True, grad=True, gan=True, image=False, network=False, idt=False)
                PN_option.set_options(l1=True, grad=True, gan=True, image=True, network=True, idt=False)
                DN_option.set_options(l1=True, grad=True, gan=True, image=True, network=True, idt=False)
                GN_option.separate_weights = [0.1, 0.25, 1.0]
                # GN_option.separate_weights = [1.0, 1.0, 1.0]

                data.max_epoch_size = 50
                PN_backward_mode = 'min'
                cartoon_resize_mode = 'resize'
                pixel_resize_mode = 'resize'
                lr_scheduler_initial_step = 25
                arch.use_lsgan = False
                downsample_as_PN_input_target = False
                print(f'PN target: {downsample_as_PN_input_target}')
                small_batch = False
                print(f'Small batch: {small_batch}')
                PN_target_scale = False
                return_downsample_GN = False

                if int(version) == 12:
                    # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/4, d_gan=1.0/4*2)
                    # arch.PN_weights = arch.Weights(l1=1.0*3.5, grad=1.0/2, image=1.0, net=1.0/10, null=1.0, gan=1.0/1.25, d_gan=1.0/1.25*2)
                    # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0*1.25, image=1.0, net=1.0/10, null=1.0, gan=1.0/2, d_gan=1.0/2)

                    # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0, net=1.0, null=1.0, gan=1.0/10, d_gan=1.0/10/20)
                    # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/5, d_gan=1.0/5/15)
                    # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/15, d_gan=1.0/15/2)
                    # arch.GN_weights = arch.Weights(l1=1.0*3, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/25, d_gan=1.0/25/25, d_target=1.0/2)
                    # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/15, d_gan=1.0/15/15, d_target=1.0/2)
                    # arch.DN_weights = arch.Weights(l1=1.0*4, grad=1.0*4, image=1.0, net=1.0, null=1.0, gan=1.0/5, d_gan=1.0/5/2, d_target=1.0/2)
                    arch.GN_weights = arch.Weights(l1=1.0, grad=1.0/15, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2/6, d_target=1.0/2)
                    arch.PN_weights = arch.Weights(l1=1.0, grad=1.0/1.5*1.5, image=1.0, net=1.0/5*3, null=1.0, gan=1.0/15/2, d_gan=1.0/15/2/50, d_target=1.0/2)
                    arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0/5, null=1.0, gan=1.0/10/2, d_gan=1.0/10/2/15, d_target=1.0/2)

                    GN_option.train = False
                    PN_option.train = True
                    DN_option.train = True
                    GN_option.set_interval(g=1, d=1)
                    PN_option.set_interval(g=1, d=1)
                    DN_option.set_interval(g=1, d=1)
                    GN_option.set_options(l1=True, grad=True, gan=True and False, image=False, network=False, idt=False)
                    PN_option.set_options(l1=True, grad=True, gan=True, image=True, network=True, idt=False)
                    DN_option.set_options(l1=True, grad=True, gan=True, image=True, network=True, idt=False)

            elif int(version) == 12:
                data.max_epoch_size = 900
                # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0, d_gan=1.0)
                # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*20, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*20, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
                arch.GN_weights = arch.Weights(l1=1.0, grad=1.0/4, image=1.0, net=1.0, null=1.0, gan=1.0, d_gan=1.0)
                arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*5, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*5, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                GN_option.gan_d_interval = 1
                PN_option.gan_d_interval = 1
                DN_option.gan_d_interval = 1
                GN_option.gan_g_interval = 1
                PN_option.gan_g_interval = 1
                DN_option.gan_g_interval = 1
                GN_option.train = True
                PN_option.train = True
                DN_option.train = True

                GN_option.gan = False
                GN_option.l1 = True
                GN_option.grad = True

                PN_option.gan = True
                PN_option.l1 = True
                PN_option.grad = True
                PN_option.mirror_image = True # False
                PN_option.mirror_network = True

                DN_option.gan = True
                DN_option.l1 = True
                DN_option.grad = True  # False, forward of DN is not detached
                DN_option.mirror_image = True
                DN_option.mirror_network = True

                # cartoon_resize_mode = 'resize'
                # pixel_resize_mode = 'resize'

            elif int(version) == 13:
                data.max_epoch_size = 900
                use_floodfill = True
                # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/1.5, d_gan=1.0/1.5)
                # arch.PN_weights = arch.Weights(l1=1.0*1.5, grad=1.0*1.5, image=1.0*5, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0*5, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)

                # arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/1.5, d_gan=1.0/1.5)
                # arch.PN_weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0*20, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                # arch.DN_weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0*20, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                arch.GN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0, d_gan=1.0)
                arch.PN_weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/2.5, d_gan=1.0/2.5)
                arch.DN_weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0*20, net=1.0, null=1.0, gan=1.0/3, d_gan=1.0/3)
                GN_option.gan_d_interval = 1
                PN_option.gan_d_interval = 1
                DN_option.gan_d_interval = 1
                GN_option.gan_g_interval = 1
                PN_option.gan_g_interval = 1
                DN_option.gan_g_interval = 1

                GN_option.train = False or True
                PN_option.train = True
                DN_option.train = False

                PN_option.gan = False or True
                PN_option.l1 = True
                PN_option.grad = True
                PN_option.mirror_image = False  # False
                DN_option.gan = True
                DN_option.l1 = True
                DN_option.grad = True  # False, forward of DN is not detached
                DN_option.mirror_image = True

            # arch.detach_network = False
            # train_GAN_mirror = True
            use_lr_scheduler = num_epochs > lr_scheduler_initial_step
            save_model = data.max_epoch_size >= 900 and num_epochs > 5 and not small_batch

arch.print_weights()

# Directory
model_dir = data.get_model_directory() + 'checkpoint' + version + '.pth'
print(f'Model Directory: {model_dir}')

# Data section
cartoon_path = 'cartoon/'
pixel_path = 'pixel/'
if use_floodfill:
    cartoon_path = 'floodfill/' + cartoon_path
    pixel_path = 'floodfill/' + pixel_path
if small_batch:
    cartoon_path = 'small/' + 'cartoon/'
    pixel_path = 'small/' + 'pixel/'
cartoon_dataloader = data.get_dataloader(cartoon_path, mode=cartoon_resize_mode)
pixel_dataloader = data.get_dataloader(pixel_path, mode=pixel_resize_mode)
n_total_steps = min(len(cartoon_dataloader), len(pixel_dataloader))
print('Number of iterations: ' + str(n_total_steps))

# Device Section
device = arch.get_device()  # torch.device("cuda")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensorboard_log_dir = data.get_tensorboard_directory() + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_v-" + version
writer = SummaryWriter(log_dir=tensorboard_log_dir)

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
print(f'Use Buffer = {arch.use_buffer}')
print(f'Load GN = {GN_option.load_generator}, {GN_option.load_discriminator}')
print(f'Load PN = {PN_option.load_generator}, {PN_option.load_discriminator}')
print(f'Load DN = {DN_option.load_generator}, {DN_option.load_discriminator}')
print(f'Use lr_scheduler: {use_lr_scheduler}')
print(f'lr_scheduler initial step: {lr_scheduler_initial_step}')
print(f'Save Model = {save_model}')
print(f'Test Model = {test_model}')
print(f'Use Flood Fill: {use_floodfill}')
print(f'Cartoon Resize Mode: {cartoon_resize_mode}')
print(f'Pixel Resize Mode: {pixel_resize_mode}')
print(f'Detach Network: {arch.detach_network}')
GN_option.print_info()
PN_option.print_info()
DN_option.print_info()
print(f'Current Time: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')


# Model Section
if GN_option.create:
    GN_generator = arch.GridNet()
    GN_generator = nn.DataParallel(GN_generator).to(device)
    GN_generator.train()
    GN_discriminator = arch.Discriminator(data.input_resolution)
    GN_discriminator = nn.DataParallel(GN_discriminator).to(device)
    GN_discriminator.train()
    GN_G_optimizer = torch.optim.Adam(GN_generator.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=weight_decay)
    GN_D_optimizer = torch.optim.Adam(GN_discriminator.parameters(), lr=learning_rate_discriminator, betas=(0.5, 0.999), weight_decay=weight_decay)
    # print(GN_generator)
    # print(GN_discriminator)
else:
    GN_generator = None
    GN_discriminator = None
    GN_G_optimizer = None
    GN_D_optimizer = None

if PN_option.create:
    PN_generator = arch.PixelNet()
    PN_generator = nn.DataParallel(PN_generator).to(device)
    PN_generator.train()
    PN_discriminator = arch.Discriminator(data.input_resolution)
    PN_discriminator = nn.DataParallel(PN_discriminator).to(device)
    PN_discriminator.train()
    PN_G_optimizer = torch.optim.Adam(PN_generator.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=weight_decay)
    PN_D_optimizer = torch.optim.Adam(PN_discriminator.parameters(), lr=learning_rate_discriminator, betas=(0.5, 0.999), weight_decay=weight_decay)
    # print(PN_generator)
    # print(PN_discriminator)
else:
    PN_generator = None
    PN_discriminator = None
    PN_G_optimizer = None
    PN_D_optimizer = None

if DN_option.create:
    DN_generator = arch.PixelNet()
    DN_generator = nn.DataParallel(DN_generator).to(device)
    DN_generator.train()
    DN_discriminator = arch.Discriminator(data.input_resolution)
    DN_discriminator = nn.DataParallel(DN_discriminator).to(device)
    DN_discriminator.train()
    DN_G_optimizer = torch.optim.Adam(DN_generator.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=weight_decay)
    DN_D_optimizer = torch.optim.Adam(DN_discriminator.parameters(), lr=learning_rate_discriminator, betas=(0.5, 0.999), weight_decay=weight_decay)
    # print(DN_generator)
    # print(DN_discriminator)
else:
    DN_generator = None
    DN_discriminator = None
    DN_G_optimizer = None
    DN_D_optimizer = None

# Buffer Section
GN_buffer_64 = arch.ImageBuffer()
GN_buffer_48 = arch.ImageBuffer()
GN_buffer_32 = arch.ImageBuffer()
PN_buffer_64 = arch.ImageBuffer()
PN_buffer_48 = arch.ImageBuffer()
PN_buffer_32 = arch.ImageBuffer()
DN_buffer = arch.ImageBuffer()


def lr_lambda(epoch):
    if epoch <= lr_scheduler_initial_step:
        return 1
    else:
        return 1 - (epoch - lr_scheduler_initial_step) / (num_epochs - lr_scheduler_initial_step)


schedulers = []
if GN_G_optimizer is not None:
    schedulers.append(lr_scheduler.LambdaLR(GN_G_optimizer, lr_lambda=lr_lambda))
    schedulers.append(lr_scheduler.LambdaLR(GN_D_optimizer, lr_lambda=lr_lambda))
if PN_G_optimizer is not None:
    schedulers.append(lr_scheduler.LambdaLR(PN_G_optimizer, lr_lambda=lr_lambda))
    schedulers.append(lr_scheduler.LambdaLR(PN_D_optimizer, lr_lambda=lr_lambda))
if DN_G_optimizer is not None:
    schedulers.append(lr_scheduler.LambdaLR(DN_G_optimizer, lr_lambda=lr_lambda))
    schedulers.append(lr_scheduler.LambdaLR(DN_D_optimizer, lr_lambda=lr_lambda))

# Load Section
if exists(model_dir):
    checkpoint = torch.load(model_dir)
    # checkpoint = torch.load(data.get_model_directory() + 'checkpoint' + '_baseline' + '.pth')

    current_epochs = checkpoint['epoch']
    current_iter = checkpoint['iteration']

    if GN_generator is not None:
        # if int(version) == 12:
        #     GN_checkpoint = torch.load(data.get_model_directory() + 'checkpoint' + '_' + '11' + '_GN' + '.pth')
        #     print(GN_checkpoint.keys())
        #     print(data.get_model_directory() + 'checkpoint' + '_' + '11' + '_GN' + '.pth')
        #     if GN_option.load_generator:
        #         GN_generator.load_state_dict(GN_checkpoint['GN_generator'])
        #         GN_G_optimizer.load_state_dict(GN_checkpoint['GN_G_optimizer'])
        #     if GN_option.load_discriminator:
        #         GN_discriminator.load_state_dict(GN_checkpoint['GN_discriminator'])
        #         GN_D_optimizer.load_state_dict(GN_checkpoint['GN_D_optimizer'])
        # else:
        if GN_option.load_generator:
            GN_generator.load_state_dict(checkpoint['GN_generator'])
            GN_G_optimizer.load_state_dict(checkpoint['GN_G_optimizer'])
        if GN_option.load_discriminator:
            GN_discriminator.load_state_dict(checkpoint['GN_discriminator'])
            GN_D_optimizer.load_state_dict(checkpoint['GN_D_optimizer'])
        if override_state_dict:
            GN_G_optimizer.param_groups[0]['lr'] = learning_rate
            GN_D_optimizer.param_groups[0]['lr'] = learning_rate_discriminator
            GN_G_optimizer.param_groups[0]['betas'] = (0.5, 0.999)
            GN_D_optimizer.param_groups[0]['betas'] = (0.5, 0.999)
            GN_G_optimizer.param_groups[0]['weight_decay'] = weight_decay
            GN_D_optimizer.param_groups[0]['weight_decay'] = weight_decay

    if PN_generator is not None:
        # if int(version) == 11:
        #     # PN_model_dir = data.get_model_directory() + 'checkpoint' + '_' + '11' + '_PN' + '.pth'
        #     PN_model_dir = data.get_model_directory() + 'checkpoint' + '22' + '.pth'
        #     print(PN_model_dir)
        #     PN_checkpoint = torch.load(PN_model_dir)
        #     print(PN_checkpoint.keys())
        #     if PN_option.load_generator:
        #         PN_generator.load_state_dict(PN_checkpoint['GN_generator'])
        #         PN_G_optimizer.load_state_dict(PN_checkpoint['GN_G_optimizer'])
        #     if PN_option.load_discriminator:
        #         PN_discriminator.load_state_dict(PN_checkpoint['GN_discriminator'])
        #         PN_D_optimizer.load_state_dict(PN_checkpoint['GN_D_optimizer'])
        # else:
        if PN_option.load_generator:
            PN_generator.load_state_dict(checkpoint['PN_generator'])
            PN_G_optimizer.load_state_dict(checkpoint['PN_G_optimizer'])
        if PN_option.load_discriminator:
            PN_discriminator.load_state_dict(checkpoint['PN_discriminator'])
            PN_D_optimizer.load_state_dict(checkpoint['PN_D_optimizer'])
        if override_state_dict:
            PN_G_optimizer.param_groups[0]['lr'] = learning_rate
            PN_D_optimizer.param_groups[0]['lr'] = learning_rate_discriminator
            PN_G_optimizer.param_groups[0]['betas'] = (0.5, 0.999)
            PN_D_optimizer.param_groups[0]['betas'] = (0.5, 0.999)
            PN_G_optimizer.param_groups[0]['weight_decay'] = weight_decay
            PN_D_optimizer.param_groups[0]['weight_decay'] = weight_decay

    if DN_generator is not None:
        # if int(version) == 12:
        #     DN_checkpoint = torch.load(data.get_model_directory() + 'checkpoint' + '_' + '11' + '_DN' + '.pth')
        #     print(DN_checkpoint.keys())
        #     print(data.get_model_directory() + 'checkpoint' + '_' + '11' + '_DN' + '.pth')
        #     if DN_option.load_generator:
        #         DN_generator.load_state_dict(DN_checkpoint['DN_generator'])
        #         DN_G_optimizer.load_state_dict(DN_checkpoint['DN_G_optimizer'])
        #     if DN_option.load_discriminator:
        #         DN_discriminator.load_state_dict(DN_checkpoint['DN_discriminator'])
        #         DN_D_optimizer.load_state_dict(DN_checkpoint['DN_D_optimizer'])
        # else:
        if DN_option.load_generator:
            DN_generator.load_state_dict(checkpoint['DN_generator'])
            DN_G_optimizer.load_state_dict(checkpoint['DN_G_optimizer'])
        if DN_option.load_discriminator:
            DN_discriminator.load_state_dict(checkpoint['DN_discriminator'])
            DN_D_optimizer.load_state_dict(checkpoint['DN_D_optimizer'])
        if override_state_dict:
            DN_G_optimizer.param_groups[0]['lr'] = learning_rate
            DN_D_optimizer.param_groups[0]['lr'] = learning_rate_discriminator
            DN_G_optimizer.param_groups[0]['betas'] = (0.5, 0.999)
            DN_D_optimizer.param_groups[0]['betas'] = (0.5, 0.999)
            DN_G_optimizer.param_groups[0]['weight_decay'] = weight_decay
            DN_D_optimizer.param_groups[0]['weight_decay'] = weight_decay

    checkpoint = None  # save memory!!!

    print("Loaded model checkpoint successfully.")
else:
    print('Model checkpoint not found.')

version_prefix = 'CUSTOM/'


def train_GN_discriminator(generated_images, target_image, printCondition):
    if not GN_option.gan:
        return 0.0
    generated_image_64, generated_image_48, generated_image_32 = generated_images
    target_image_64 = data.resize(target_image, 1.0 / 4.0)
    target_image_48 = data.resize(target_image, 3.0 / 16.0)
    target_image_32 = data.resize(target_image, 1.0 / 8.0)

    # On target image
    GN_D_labels_input_64 = GN_discriminator(target_image_64)
    GN_D_labels_input_48 = GN_discriminator(target_image_48)
    GN_D_labels_input_32 = GN_discriminator(target_image_32)
    GN_D_loss_input_64 = arch.adversarial_criterion(GN_D_labels_input_64, True, weight=arch.GN_weights.d_target)
    GN_D_loss_input_48 = arch.adversarial_criterion(GN_D_labels_input_48, True, weight=arch.GN_weights.d_target)
    GN_D_loss_input_32 = arch.adversarial_criterion(GN_D_labels_input_32, True, weight=arch.GN_weights.d_target)
    GN_D_loss_input = GN_D_loss_input_64 + GN_D_loss_input_48 + GN_D_loss_input_32

    # On generated image
    GN_D_labels_generated_64 = GN_discriminator(GN_buffer_64.get_image(generated_image_64.detach()))
    GN_D_labels_generated_48 = GN_discriminator(GN_buffer_48.get_image(generated_image_48.detach()))
    GN_D_labels_generated_32 = GN_discriminator(GN_buffer_32.get_image(generated_image_32.detach()))
    GN_D_loss_generated_64 = arch.adversarial_criterion(GN_D_labels_generated_64, False, weight=arch.GN_weights.d_gan)
    GN_D_loss_generated_48 = arch.adversarial_criterion(GN_D_labels_generated_48, False, weight=arch.GN_weights.d_gan)
    GN_D_loss_generated_32 = arch.adversarial_criterion(GN_D_labels_generated_32, False, weight=arch.GN_weights.d_gan)
    GN_D_loss_generated = GN_D_loss_generated_64 + GN_D_loss_generated_48 + GN_D_loss_generated_32

    if use_tensorboard and printCondition:
        prefix = version_prefix + 'GN/'
        writer.add_scalar(prefix + "D/LOSS/64/TARGET", GN_D_loss_input_64 / arch.GN_weights.d_target, current_iter)
        writer.add_scalar(prefix + "D/LOSS/64/GENERATED", GN_D_loss_generated_64 / arch.GN_weights.d_gan, current_iter)
        writer.add_scalar(prefix + "D/LOSS/48/TARGET", GN_D_loss_input_48 / arch.GN_weights.d_target, current_iter)
        writer.add_scalar(prefix + "D/LOSS/48/GENERATED", GN_D_loss_generated_48 / arch.GN_weights.d_gan, current_iter)
        writer.add_scalar(prefix + "D/LOSS/32/TARGET", GN_D_loss_input_32 / arch.GN_weights.d_target, current_iter)
        writer.add_scalar(prefix + "D/LOSS/32/GENERATED", GN_D_loss_generated_32 / arch.GN_weights.d_gan, current_iter)

    GN_D_loss = (GN_D_loss_input + GN_D_loss_generated) * 0.5
    # loss_buffer.GN_D_loss = GN_D_loss
    return GN_D_loss


def train_GN_generator(generated_images, input_image, printCondition):
    input_image_64 = data.resize(input_image, 1.0 / 4.0)
    input_image_48 = data.resize(input_image, 3.0 / 16.0)
    input_image_32 = data.resize(input_image, 1.0 / 8.0)
    generated_image_64, generated_image_48, generated_image_32 = generated_images
    if GN_option.gan:
        GN_G_labels_generated_64 = GN_discriminator(generated_image_64)
        GN_G_labels_generated_48 = GN_discriminator(generated_image_48)
        GN_G_labels_generated_32 = GN_discriminator(generated_image_32)
        GN_G_gan_loss_64 = arch.adversarial_criterion(GN_G_labels_generated_64, True, weight=arch.GN_weights.gan)
        GN_G_gan_loss_48 = arch.adversarial_criterion(GN_G_labels_generated_48, True, weight=arch.GN_weights.gan)
        GN_G_gan_loss_32 = arch.adversarial_criterion(GN_G_labels_generated_32, True, weight=arch.GN_weights.gan)
        GN_G_gan_loss = GN_G_gan_loss_64 + GN_G_gan_loss_48 + GN_G_gan_loss_32
    else:
        GN_G_gan_loss_64 = 0
        GN_G_gan_loss_48 = 0
        GN_G_gan_loss_32 = 0
        GN_G_gan_loss = 0.0

    if GN_option.grad:
        GN_G_grad_loss_64 = arch.gradient_criterion(generated_image_64, input_image_64, arch.GN_weights.grad, variation=False)
        GN_G_grad_loss_48 = arch.gradient_criterion(generated_image_48, input_image_48, arch.GN_weights.grad, variation=False)
        GN_G_grad_loss_32 = arch.gradient_criterion(generated_image_32, input_image_32, arch.GN_weights.grad, variation=False)
        GN_G_grad_loss = GN_G_grad_loss_64 + GN_G_grad_loss_48 + GN_G_grad_loss_32
    else:
        GN_G_grad_loss_64 = 0
        GN_G_grad_loss_48 = 0
        GN_G_grad_loss_32 = 0
        GN_G_grad_loss = 0

    if GN_option.l1:
        # GN_G_l1_loss_64 = arch.l1_criterion(generated_image_64, input_image_64, arch.GN_weights.l1)
        # GN_G_l1_loss_48 = arch.l1_criterion(generated_image_48, input_image_48, arch.GN_weights.l1)
        # GN_G_l1_loss_32 = arch.l1_criterion(generated_image_32, input_image_32, arch.GN_weights.l1)
        GN_G_l1_loss_64 = arch.l1_criterion(data.resize(generated_image_64, scale=4.0), input_image, arch.GN_weights.l1)
        GN_G_l1_loss_48 = arch.l1_criterion(data.resize(generated_image_48, scale=16.0/3.0), input_image, arch.GN_weights.l1)
        GN_G_l1_loss_32 = arch.l1_criterion(data.resize(generated_image_32, scale=8.0), input_image, arch.GN_weights.l1)
        if len(GN_option.separate_weights) > 2:
            weights = [GN_option.separate_weights[0], GN_option.separate_weights[1], GN_option.separate_weights[2]]
            GN_G_l1_loss = GN_G_l1_loss_64 * weights[0] + GN_G_l1_loss_48 * weights[1] + GN_G_l1_loss_32 * weights[2]
        else:
            GN_G_l1_loss = GN_G_l1_loss_64 + GN_G_l1_loss_48 + GN_G_l1_loss_32
    else:
        GN_G_l1_loss_64 = 0
        GN_G_l1_loss_48 = 0
        GN_G_l1_loss_32 = 0
        GN_G_l1_loss = 0

    if use_tensorboard and printCondition:
        prefix = version_prefix + 'GN/'
        writer.add_scalar(prefix + "G/LOSS/GAN/64", GN_G_gan_loss_64 / arch.GN_weights.gan, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GAN/48", GN_G_gan_loss_48 / arch.GN_weights.gan, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GAN/32", GN_G_gan_loss_32 / arch.GN_weights.gan, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GRAD/64", GN_G_grad_loss_64, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GRAD/48", GN_G_grad_loss_48, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GRAD/32", GN_G_grad_loss_32, current_iter)
        writer.add_scalar(prefix + "G/LOSS/L1/64", GN_G_l1_loss_64, current_iter)
        writer.add_scalar(prefix + "G/LOSS/L1/48", GN_G_l1_loss_48, current_iter)
        writer.add_scalar(prefix + "G/LOSS/L1/32", GN_G_l1_loss_32, current_iter)

    GN_G_loss = GN_G_l1_loss + GN_G_grad_loss + GN_G_gan_loss
    # loss_buffer.GN_G_loss = GN_G_loss
    return GN_G_loss


def train_PN_discriminator(generated_images, target_image, printCondition):
    if not PN_option.gan:
        return 0.0
    generated_image_64, generated_image_48, generated_image_32 = generated_images
    # On input image
    if PN_target_scale:
        target_image_64 = data.desample(target_image, 4.0)
        target_image_48 = data.desample(target_image, 16.0 / 3.0)
        target_image_32 = data.desample(target_image, 8.0)
        PN_D_labels_input_64 = PN_discriminator(target_image_64)
        PN_D_labels_input_48 = PN_discriminator(target_image_48)
        PN_D_labels_input_32 = PN_discriminator(target_image_32)
    else:
        PN_D_labels_input_64 = PN_discriminator(target_image)
        PN_D_labels_input_48 = PN_discriminator(target_image)
        PN_D_labels_input_32 = PN_discriminator(target_image)
    PN_D_loss_input_64 = arch.adversarial_criterion(PN_D_labels_input_64, True, weight=arch.PN_weights.d_target)
    PN_D_loss_input_48 = arch.adversarial_criterion(PN_D_labels_input_48, True, weight=arch.PN_weights.d_target)
    PN_D_loss_input_32 = arch.adversarial_criterion(PN_D_labels_input_32, True, weight=arch.PN_weights.d_target)
    PN_D_loss_input = PN_D_loss_input_64 + PN_D_loss_input_48 + PN_D_loss_input_32

    # On generated image
    PN_D_labels_generated_64 = PN_discriminator(PN_buffer_64.get_image(generated_image_64.detach()))
    PN_D_labels_generated_48 = PN_discriminator(PN_buffer_48.get_image(generated_image_48.detach()))
    PN_D_labels_generated_32 = PN_discriminator(PN_buffer_32.get_image(generated_image_32.detach()))
    PN_D_loss_generated_64 = arch.adversarial_criterion(PN_D_labels_generated_64, False, weight=arch.PN_weights.d_gan)
    PN_D_loss_generated_48 = arch.adversarial_criterion(PN_D_labels_generated_48, False, weight=arch.PN_weights.d_gan)
    PN_D_loss_generated_32 = arch.adversarial_criterion(PN_D_labels_generated_32, False, weight=arch.PN_weights.d_gan)
    PN_D_loss_generated = PN_D_loss_generated_64 + PN_D_loss_generated_48 + PN_D_loss_generated_32

    if use_tensorboard and printCondition:
        prefix = version_prefix + 'PN/'
        writer.add_scalar(prefix + "D/LOSS/64/TARGET", PN_D_loss_input_64 / arch.PN_weights.d_target, current_iter)
        writer.add_scalar(prefix + "D/LOSS/64/GENERATED", PN_D_loss_generated_64 / arch.PN_weights.d_gan, current_iter)
        writer.add_scalar(prefix + "D/LOSS/48/TARGET", PN_D_loss_input_48 / arch.PN_weights.d_target, current_iter)
        writer.add_scalar(prefix + "D/LOSS/48/GENERATED", PN_D_loss_generated_48 / arch.PN_weights.d_gan, current_iter)
        writer.add_scalar(prefix + "D/LOSS/32/TARGET", PN_D_loss_input_32 / arch.PN_weights.d_target, current_iter)
        writer.add_scalar(prefix + "D/LOSS/32/GENERATED", PN_D_loss_generated_32 / arch.PN_weights.d_gan, current_iter)

    PN_D_loss = (PN_D_loss_input + PN_D_loss_generated) * 0.5
    # loss_buffer.PN_D_loss = PN_D_loss
    return PN_D_loss


def train_PN_generator(generated_images, input_images, printCondition):
    input_image_64, input_image_48, input_image_32 = input_images
    generated_image_64, generated_image_48, generated_image_32 = generated_images
    if PN_option.gan:
        PN_G_labels_generated_64 = PN_discriminator(generated_image_64)
        PN_G_labels_generated_48 = PN_discriminator(generated_image_48)
        PN_G_labels_generated_32 = PN_discriminator(generated_image_32)
        PN_G_gan_loss_64 = arch.adversarial_criterion(PN_G_labels_generated_64, True, weight=arch.PN_weights.gan)
        PN_G_gan_loss_48 = arch.adversarial_criterion(PN_G_labels_generated_48, True, weight=arch.PN_weights.gan)
        PN_G_gan_loss_32 = arch.adversarial_criterion(PN_G_labels_generated_32, True, weight=arch.PN_weights.gan)
        PN_G_gan_loss = PN_G_gan_loss_64 + PN_G_gan_loss_48 + PN_G_gan_loss_32
    else:
        PN_G_gan_loss_64 = 0
        PN_G_gan_loss_48 = 0
        PN_G_gan_loss_32 = 0
        PN_G_gan_loss = 0.0

    if PN_option.grad:
        PN_G_grad_loss_64 = arch.gradient_criterion(generated_image_64, input_image_64, arch.PN_weights.grad, variation=False)
        PN_G_grad_loss_48 = arch.gradient_criterion(generated_image_48, input_image_48, arch.PN_weights.grad, variation=False)
        PN_G_grad_loss_32 = arch.gradient_criterion(generated_image_32, input_image_32, arch.PN_weights.grad, variation=False)
        PN_G_grad_loss = PN_G_grad_loss_64 + PN_G_grad_loss_48 + PN_G_grad_loss_32
    else:
        PN_G_grad_loss_64 = 0
        PN_G_grad_loss_48 = 0
        PN_G_grad_loss_32 = 0
        PN_G_grad_loss = 0

    if PN_option.l1:
        PN_G_l1_loss_64 = arch.l1_criterion(generated_image_64, input_image_64, arch.PN_weights.l1)
        PN_G_l1_loss_48 = arch.l1_criterion(generated_image_48, input_image_48, arch.PN_weights.l1)
        PN_G_l1_loss_32 = arch.l1_criterion(generated_image_32, input_image_32, arch.PN_weights.l1)
        PN_G_l1_loss = PN_G_l1_loss_64 + PN_G_l1_loss_48 + PN_G_l1_loss_32
    else:
        PN_G_l1_loss_64 = 0
        PN_G_l1_loss_48 = 0
        PN_G_l1_loss_32 = 0
        PN_G_l1_loss = 0

    if use_tensorboard and printCondition:
        prefix = version_prefix + 'PN/'
        writer.add_scalar(prefix + "G/LOSS/GAN/64", PN_G_gan_loss_64 / arch.PN_weights.gan, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GAN/48", PN_G_gan_loss_48 / arch.PN_weights.gan, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GAN/32", PN_G_gan_loss_32 / arch.PN_weights.gan, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GRAD/64", PN_G_grad_loss_64, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GRAD/48", PN_G_grad_loss_48, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GRAD/32", PN_G_grad_loss_32, current_iter)
        writer.add_scalar(prefix + "G/LOSS/L1/64", PN_G_l1_loss_64, current_iter)
        writer.add_scalar(prefix + "G/LOSS/L1/48", PN_G_l1_loss_48, current_iter)
        writer.add_scalar(prefix + "G/LOSS/L1/32", PN_G_l1_loss_32, current_iter)

    PN_G_loss = PN_G_l1_loss + PN_G_grad_loss + PN_G_gan_loss
    # loss_buffer.PN_G_loss = PN_G_loss
    return PN_G_loss


def train_PN_generator_mirror(generated_images, mirror_image, printCondition):
    # https://ssnl.github.io/better_cycles/report.pdf
    # https://github.com/shijx12/DeepSim/blob/master/deepSimGAN/deepSimNet.py
    # https://arxiv.org/pdf/1602.02644.pdf
    generated_image_64, generated_image_48, generated_image_32 = generated_images
    if PN_option.mirror_image:
        if PN_target_scale:
            mirror_image = data.desample(mirror_image, 8.0)
        PN_G_mirror_image_loss_64 = arch.image_mirror_criterion(generated_image_64, mirror_image, arch.PN_weights.image)
        PN_G_mirror_image_loss_48 = arch.image_mirror_criterion(generated_image_48, mirror_image, arch.PN_weights.image)
        PN_G_mirror_image_loss_32 = arch.image_mirror_criterion(generated_image_32, mirror_image, arch.PN_weights.image)
        if PN_backward_mode == 'all':
            PN_G_mirror_image_loss = PN_G_mirror_image_loss_64 + PN_G_mirror_image_loss_48 + PN_G_mirror_image_loss_32
        elif PN_backward_mode == 'min':
            PN_G_mirror_image_loss = data.choose_min((PN_G_mirror_image_loss_64, PN_G_mirror_image_loss_48, PN_G_mirror_image_loss_32))
        else:
            PN_G_mirror_image_loss = data.choose_random((PN_G_mirror_image_loss_64, PN_G_mirror_image_loss_48, PN_G_mirror_image_loss_32))
    else:
        PN_G_mirror_image_loss = 0.0

    if use_tensorboard and printCondition:
        prefix = version_prefix + 'PN/'
        writer.add_scalar(prefix + "G/LOSS/MIRROR/IMAGE", PN_G_mirror_image_loss, current_iter)

    PN_G_mirror_loss = PN_G_mirror_image_loss

    # if train_GAN_mirror and PN_option.gan:
    #     PN_G_labels_generated = PN_discriminator(generated_image_32)
    #     PN_G_mirror_adversarial_loss = arch.adversarial_criterion(PN_G_labels_generated, True, weight=arch.PN_weights.gan)
    #     PN_G_mirror_loss = PN_G_mirror_loss + PN_G_mirror_adversarial_loss
    #
    #     if use_tensorboard and printCondition:
    #         prefix = version_prefix + 'PN/'
    #         writer.add_scalar(prefix + "G/LOSS/MIRROR/GAN", PN_G_mirror_adversarial_loss, current_iter)

    return PN_G_mirror_loss


def train_DN_discriminator(generated_image, target_image, printCondition):
    if not DN_option.gan:
        return 0.0
    # On input image
    DN_D_labels_input = DN_discriminator(target_image)
    DN_D_loss_input = arch.adversarial_criterion(DN_D_labels_input, True, weight=arch.DN_weights.d_target)

    # On generated image
    DN_D_labels_generated = DN_discriminator(DN_buffer.get_image(generated_image.detach()))
    DN_D_loss_generated = arch.adversarial_criterion(DN_D_labels_generated, False, weight=arch.DN_weights.d_gan)

    if use_tensorboard and printCondition:
        prefix = version_prefix + 'DN/'
        writer.add_scalar(prefix + "D/LOSS/TARGET", DN_D_loss_input / arch.DN_weights.d_target, current_iter)
        writer.add_scalar(prefix + "D/LOSS/GENERATED", DN_D_loss_generated / arch.DN_weights.d_gan, current_iter)

    DN_D_loss = (DN_D_loss_input + DN_D_loss_generated) * 0.5
    # loss_buffer.DN_D_loss = DN_D_loss
    return DN_D_loss


def train_DN_generator(generated_image, input_image, printCondition):
    if DN_option.gan:
        DN_G_labels_generated = DN_discriminator(generated_image)
        DN_G_gan_loss = arch.adversarial_criterion(DN_G_labels_generated, True, weight=arch.DN_weights.gan)
    else:
        DN_G_gan_loss = 0.0

    if DN_option.grad:
        DN_G_grad_loss = arch.gradient_criterion(generated_image, input_image, arch.DN_weights.grad,
                                                 variation=True)
    else:
        DN_G_grad_loss = 0

    if DN_option.l1:
        DN_G_l1_loss = arch.l1_criterion(generated_image, input_image, arch.DN_weights.l1)
    else:
        DN_G_l1_loss = 0


    if use_tensorboard and printCondition:
        prefix = version_prefix + 'DN/'
        writer.add_scalar(prefix + "G/LOSS/GAN", DN_G_gan_loss / arch.DN_weights.gan, current_iter)
        writer.add_scalar(prefix + "G/LOSS/GRAD", DN_G_grad_loss, current_iter)
        writer.add_scalar(prefix + "G/LOSS/L1", DN_G_l1_loss, current_iter)

    DN_G_loss = DN_G_l1_loss + DN_G_grad_loss + DN_G_gan_loss
    # loss_buffer.DN_G_loss = DN_G_loss
    return DN_G_loss


def train_DN_generator_mirror(generated_image, mirror_image, printCondition):
    if DN_option.mirror_image:
        DN_G_mirror_image_loss = arch.image_mirror_criterion(generated_image, mirror_image, arch.DN_weights.image)
    else:
        DN_G_mirror_image_loss = 0.0

    if use_tensorboard and printCondition:
        prefix = version_prefix + 'DN/'
        writer.add_scalar(prefix + "G/LOSS/MIRROR/IMAGE", DN_G_mirror_image_loss, current_iter)

    DN_G_mirror_loss = DN_G_mirror_image_loss

    # if train_GAN_mirror and DN_option.gan:
    #     DN_G_labels_generated = DN_discriminator(generated_image)
    #     DN_G_mirror_adversarial_loss = arch.adversarial_criterion(DN_G_labels_generated, True, weight=arch.DN_weights.gan)
    #     DN_G_mirror_loss = DN_G_mirror_loss + DN_G_mirror_adversarial_loss
    #
    #     if use_tensorboard and printCondition:
    #         prefix = version_prefix + 'DN/'
    #         writer.add_scalar(prefix + "G/LOSS/MIRROR/GAN", DN_G_mirror_adversarial_loss, current_iter)

    return DN_G_mirror_loss


def train_identity_loss(model, identity_image, tag, writeCondition, drawCondition):
    _, generated_image = model(identity_image)
    identity_loss = arch.l1_criterion(generated_image, identity_image, 0.0)

    if use_tensorboard:
        prefix = version_prefix + tag + '/'
        if writeCondition:
            writer.add_scalar(prefix + "G/LOSS/IDENTITY", identity_loss, current_iter)
        if drawCondition:
            writer.add_images(prefix + 'OUTPUT/IDENTITY', generated_image / 2 + 0.5, current_iter)

    return identity_loss


# Training Section
for epoch in range(num_epochs):
    cartoon_iter = iter(cartoon_dataloader)
    pixel_iter = iter(pixel_dataloader)

    for i in range(n_total_steps):

        loss_buffer.clear()

        cartoon_image, cartoon_label = next(cartoon_iter)
        pixel_image, pixel_label = next(pixel_iter)
        cartoon_image = cartoon_image.to(device)
        pixel_image = pixel_image.to(device)

        cartoon_image_64 = data.desample(cartoon_image, 4.0)
        cartoon_image_48 = data.desample(cartoon_image, 16.0 / 3.0)
        cartoon_image_32 = data.desample(cartoon_image, 8.0)

        saveCondition = (i + 1) % int(n_total_steps) == 0 and (epoch % 2 == 0)
        printCondition = (i + 1) % int(n_total_steps) == 0 or i == 0
        addScalarCondition = (i + 1) % int(n_total_steps / 4) == 0 or i == 0
        # GN_train_G_condition = (i + 1) % int(GN_option.gan_g_interval) == 0 or i == 0
        # GN_train_D_condition = (i + 1) % int(GN_option.gan_d_interval) == 0 or i == 0
        # PN_train_G_condition = (i + 1) % int(PN_option.gan_g_interval) == 0 or i == 0
        # PN_train_D_condition = (i + 1) % int(PN_option.gan_d_interval) == 0 or i == 0
        # DN_train_G_condition = (i + 1) % int(DN_option.gan_g_interval) == 0 or i == 0
        # DN_train_D_condition = (i + 1) % int(DN_option.gan_d_interval) == 0 or i == 0
        GN_train_G_condition = (current_iter) % int(GN_option.gan_g_interval) == 0
        GN_train_D_condition = (current_iter) % int(GN_option.gan_d_interval) == 0
        PN_train_G_condition = (current_iter) % int(PN_option.gan_g_interval) == 0
        PN_train_D_condition = (current_iter) % int(PN_option.gan_d_interval) == 0
        DN_train_G_condition = (current_iter) % int(DN_option.gan_g_interval) == 0
        DN_train_D_condition = (current_iter) % int(DN_option.gan_d_interval) == 0

        if printCondition:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}]')
            writer.add_scalar("LR", schedulers[0].get_last_lr()[0], current_iter)

        # Forward
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
        GN_features, F_GN_generated_image_64, F_GN_generated_image_48, F_GN_generated_image_32 = GN_generator(cartoon_image)
        F_GN_outputs = (F_GN_generated_image_64, F_GN_generated_image_48, F_GN_generated_image_32)
        GN_features = (GN_features[0].detach(), GN_features[1].detach(), GN_features[2].detach())

        F_GN_generated_image_64 = data.resize(F_GN_generated_image_64, 4.0)
        F_GN_generated_image_48 = data.resize(F_GN_generated_image_48, 16.0 / 3.0)
        F_GN_generated_image_32 = data.resize(F_GN_generated_image_32, 8.0)

        if return_downsample_GN:
            _, F_PN_generated_image_64 = PN_generator(cartoon_image_64)
            _, F_PN_generated_image_48 = PN_generator(cartoon_image_48)
            _, F_PN_generated_image_32 = PN_generator(cartoon_image_32)
        else:
            _, F_PN_generated_image_64 = PN_generator(F_GN_generated_image_64)
            _, F_PN_generated_image_48 = PN_generator(F_GN_generated_image_48)
            _, F_PN_generated_image_32 = PN_generator(F_GN_generated_image_32)
        F_PN_outputs = (F_PN_generated_image_64, F_PN_generated_image_48, F_PN_generated_image_32)

        F_PN_output = data.choose_random(F_PN_outputs)
        F_DN_features, F_DN_generated_image = DN_generator(F_PN_output)
        F_DN_features = (F_DN_features[-1], F_DN_features[-2], F_DN_features[-3])

        # Backward
        B_DN_features, B_DN_generated_image = DN_generator(pixel_image)
        B_DN_features = (B_DN_features[0].detach(), B_DN_features[1].detach(), B_DN_features[2].detach())

        _, B_GN_generated_image_64, B_GN_generated_image_48, B_GN_generated_image_32 = GN_generator(B_DN_generated_image)
        B_GN_generated_image_64 = data.resize(B_GN_generated_image_64, 4.0)
        B_GN_generated_image_48 = data.resize(B_GN_generated_image_48, 16.0 / 3.0)
        B_GN_generated_image_32 = data.resize(B_GN_generated_image_32, 8.0)
        # B_GN_output = data.choose_random((B_GN_generated_image_64, B_GN_generated_image_48, B_GN_generated_image_32))

        if return_downsample_GN:
            PN_features_64, B_PN_generated_image_64 = PN_generator(data.desample(B_DN_generated_image.detach(), 4.0))
            PN_features_48, B_PN_generated_image_48 = PN_generator(data.desample(B_DN_generated_image.detach(), 16.0 / 3.0))
            PN_features_32, B_PN_generated_image_32 = PN_generator(data.desample(B_DN_generated_image.detach(), 8.0))
        else:
            PN_features_64, B_PN_generated_image_64 = PN_generator(B_GN_generated_image_64)
            PN_features_48, B_PN_generated_image_48 = PN_generator(B_GN_generated_image_48)
            PN_features_32, B_PN_generated_image_32 = PN_generator(B_GN_generated_image_32)
        PN_features = data.choose_random((PN_features_64, PN_features_48, PN_features_32))
        PN_features = (PN_features[-1], PN_features[-2], PN_features[-3])

        # Train GN
        if GN_option.train:
            # Training Discriminator
            if GN_train_D_condition:
                loss_buffer.GN_D_loss = train_GN_discriminator(F_GN_outputs, pixel_image, addScalarCondition)

            # Training Generator
            if GN_train_G_condition:
                loss_buffer.GN_G_loss = train_GN_generator(F_GN_outputs, cartoon_image, addScalarCondition)

        # Train PN
        if PN_option.train:
            # Training Discriminator
            if PN_train_D_condition:
                loss_buffer.PN_D_loss = train_PN_discriminator(F_PN_outputs, pixel_image, addScalarCondition)

            # Training Generator
            if PN_train_G_condition:
                if return_downsample_GN or downsample_as_PN_input_target:
                    F_GN_outputs = (cartoon_image_64, cartoon_image_48, cartoon_image_32)
                else:
                    F_GN_outputs = (F_GN_generated_image_64.detach(), F_GN_generated_image_48.detach(), F_GN_generated_image_32.detach())
                loss_buffer.PN_G_loss = train_PN_generator(F_PN_outputs, F_GN_outputs, addScalarCondition)

                B_PN_outputs = (B_PN_generated_image_64, B_PN_generated_image_48, B_PN_generated_image_32)
                loss_buffer.PN_G_loss = loss_buffer.PN_G_loss + train_PN_generator_mirror(B_PN_outputs,
                                                                                          pixel_image,
                                                                                          addScalarCondition)
                if PN_option.mirror_network:
                    PN_mirror_network_loss = arch.network_mirror_criterion(PN_features, B_DN_features, arch.PN_weights.net)
                    loss_buffer.PN_G_loss = loss_buffer.PN_G_loss + PN_mirror_network_loss
                    if use_tensorboard and addScalarCondition:
                        prefix = version_prefix + 'PN/'
                        writer.add_scalar(prefix + "G/LOSS/MIRROR_NETWORK", PN_mirror_network_loss, current_iter)

                if PN_option.identity:
                    loss_buffer.PN_G_loss = loss_buffer.PN_G_loss + train_identity_loss(PN_generator, pixel_image, 'PN',
                                                                                        addScalarCondition, saveCondition) * arch.PN_weights.idt

        if DN_option.train:
            # Training Discriminator
            if DN_train_D_condition:
                loss_buffer.DN_D_loss = train_DN_discriminator(B_DN_generated_image, cartoon_image, addScalarCondition)

            # Training Generator
            if DN_train_G_condition:
                loss_buffer.DN_G_loss = train_DN_generator(B_DN_generated_image, pixel_image, addScalarCondition)
                loss_buffer.DN_G_loss = loss_buffer.DN_G_loss + train_DN_generator_mirror(F_DN_generated_image,
                                                                                          cartoon_image,
                                                                                          addScalarCondition)
                if DN_option.mirror_network:
                    DN_mirror_network_loss = arch.network_mirror_criterion(F_DN_features, GN_features, arch.DN_weights.net)
                    loss_buffer.DN_G_loss = loss_buffer.DN_G_loss + DN_mirror_network_loss
                    if use_tensorboard and addScalarCondition:
                        prefix = version_prefix + 'DN/'
                        writer.add_scalar(prefix + "G/LOSS/MIRROR_NETWORK", DN_mirror_network_loss, current_iter)

                if DN_option.identity:
                    loss_buffer.DN_G_loss = loss_buffer.DN_G_loss + train_identity_loss(DN_generator, cartoon_image,
                                                                                        'DN',
                                                                                        addScalarCondition,
                                                                                        saveCondition) * arch.DN_weights.idt

        if use_tensorboard and saveCondition:
            prefix = version_prefix + 'A/OVERALL/'
            writer.add_images(prefix + 'INPUT/CARTOON', cartoon_image / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'INPUT/PIXEL', pixel_image / 2 + 0.5, current_iter)
            prefix = version_prefix + 'B/GN/'
            writer.add_images(prefix + 'OUTPUT/64', F_GN_generated_image_64 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'OUTPUT/48', F_GN_generated_image_48 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'OUTPUT/32', F_GN_generated_image_32 / 2 + 0.5, current_iter)
            prefix = version_prefix + 'C/PN/'
            writer.add_images(prefix + 'OUTPUT/64', F_PN_generated_image_64 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'OUTPUT/48', F_PN_generated_image_48 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'OUTPUT/32', F_PN_generated_image_32 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'MIRROR/64', B_PN_generated_image_64 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'MIRROR/48', B_PN_generated_image_48 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'MIRROR/32', B_PN_generated_image_32 / 2 + 0.5, current_iter)
            prefix = version_prefix + 'D/DN/'
            writer.add_images(prefix + 'OUTPUT', B_DN_generated_image / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'MiRROR', F_DN_generated_image / 2 + 0.5, current_iter)

        if combine_loss:
            GN_D_optimizer.zero_grad()
            GN_G_optimizer.zero_grad()
            PN_G_optimizer.zero_grad()
            PN_D_optimizer.zero_grad()
            DN_G_optimizer.zero_grad()
            DN_D_optimizer.zero_grad()

            total_loss = loss_buffer.GN_D_loss + loss_buffer.GN_G_loss + loss_buffer.PN_D_loss + loss_buffer.PN_G_loss + loss_buffer.DN_D_loss + loss_buffer.DN_G_loss
            total_loss.backward()
            if GN_option.train:
                GN_D_optimizer.step()
                GN_G_optimizer.step()
            if PN_option.train:
                PN_G_optimizer.step()
                PN_D_optimizer.step()
            if DN_option.train:
                DN_G_optimizer.step()
                DN_D_optimizer.step()
        else:
            if GN_option.train and GN_train_G_condition:
                GN_G_optimizer.zero_grad()
                loss_buffer.GN_G_loss.backward(retain_graph=True)
                GN_G_optimizer.step()
            if PN_option.train and PN_train_G_condition:
                PN_G_optimizer.zero_grad()
                loss_buffer.PN_G_loss.backward(retain_graph=True)
                PN_G_optimizer.step()
            if DN_option.train and DN_train_G_condition:
                DN_G_optimizer.zero_grad()
                loss_buffer.DN_G_loss.backward()
                DN_G_optimizer.step()
            if GN_option.train and GN_option.gan and GN_train_D_condition:
                GN_D_optimizer.zero_grad()
                loss_buffer.GN_D_loss.backward()
                GN_D_optimizer.step()
            if PN_option.train and PN_option.gan and PN_train_D_condition:
                PN_D_optimizer.zero_grad()
                loss_buffer.PN_D_loss.backward()
                PN_D_optimizer.step()
            if DN_option.train and DN_option.gan and DN_train_D_condition:
                DN_D_optimizer.zero_grad()
                loss_buffer.DN_D_loss.backward()
                DN_D_optimizer.step()

        if printCondition:
            loss_buffer.print_info()
        current_iter += 1

    # lr_scheduler
    if use_lr_scheduler:
        for scheduler in schedulers: scheduler.step()

    current_epochs += 1
    print(f'Current Time: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

print('Finished Training')

# Save Section
if save_model:
    save_checkpoint = {
        'epoch': current_epochs,
        'iteration': current_iter,
    }

    if GN_generator is not None:
        save_checkpoint['GN_generator'] = GN_generator.state_dict()
        save_checkpoint['GN_discriminator'] = GN_discriminator.state_dict()
        save_checkpoint['GN_G_optimizer'] = GN_G_optimizer.state_dict()
        save_checkpoint['GN_D_optimizer'] = GN_D_optimizer.state_dict()
    if PN_generator is not None:
        save_checkpoint['PN_generator'] = PN_generator.state_dict()
        save_checkpoint['PN_discriminator'] = PN_discriminator.state_dict()
        save_checkpoint['PN_G_optimizer'] = PN_G_optimizer.state_dict()
        save_checkpoint['PN_D_optimizer'] = PN_D_optimizer.state_dict()
    if DN_generator is not None:
        save_checkpoint['DN_generator'] = DN_generator.state_dict()
        save_checkpoint['DN_discriminator'] = DN_discriminator.state_dict()
        save_checkpoint['DN_G_optimizer'] = DN_G_optimizer.state_dict()
        save_checkpoint['DN_D_optimizer'] = DN_D_optimizer.state_dict()

    torch.save(save_checkpoint, model_dir)

    save_checkpoint = None
    print(f'Saved models as \"{model_dir}\"')

if save_plot:
    data.plot_all(str(current_epochs - num_epochs+1) + '_' + str(current_epochs))

if use_tensorboard:
    writer.flush()
    writer.close()
    print('Flushed to Tensorboard')

# Test Section
if test_model:
    pixel_dataloader = None
    cartoon_dataloader = None
    pixel_iter = None
    cartoon_iter = None

    GN_buffer_32 = None
    GN_buffer_48 = None
    GN_buffer_64 = None
    PN_buffer_32 = None
    PN_buffer_48 = None
    PN_buffer_64 = None
    DN_buffer = None

    GN_discriminator = None
    PN_discriminator = None
    DN_discriminator = None

    GN_G_optimizer = None
    GN_D_optimizer = None
    PN_G_optimizer = None
    PN_D_optimizer = None
    DN_G_optimizer = None
    DN_D_optimizer = None
    schedulers = []

    if use_tensorboard:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=tensorboard_log_dir + '_test')

    GN_generator.eval()
    PN_generator.eval()
    DN_generator.eval()

    if os.path.exists(data.paths.test_dir):
        # Test with cartoon
        test_dataloader = data.get_dataloader('test/', apply_train_transform=False)
        test_iter = iter(test_dataloader)

        for i in range(len(test_dataloader)):
            test_image, test_label = next(test_iter)
            test_image = test_image.to(device)

            if test_label != 2:
                _, GN_64, GN_48, GN_32 = GN_generator(test_image)

                _, PN_64_test = PN_generator(data.desample(test_image, 4.0))
                _, PN_48_test = PN_generator(data.desample(test_image, 16.0 / 3.0))
                _, PN_32_test = PN_generator(data.desample(test_image, 8.0))

                GN_64 = data.resize(GN_64, 4.0)
                GN_48 = data.resize(GN_48, 16.0 / 3.0)
                GN_32 = data.resize(GN_32, 8.0)
                _, PN_64 = PN_generator(GN_64)
                _, PN_48 = PN_generator(GN_48)
                _, PN_32 = PN_generator(GN_32)

                _, DN_64 = DN_generator(PN_64)
                _, DN_48 = DN_generator(PN_48)
                _, DN_32 = DN_generator(PN_32)

                if use_tensorboard:
                    writer.add_images('TEST/FORWARD/GN/G/64/', GN_64 / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/GN/G/48/', GN_48 / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/GN/G/32/', GN_32 / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/PN/G/64/TEST/', PN_64_test / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/PN/G/48/TEST/', PN_48_test / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/PN/G/32/TEST/', PN_32_test / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/PN/G/64/', PN_64 / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/PN/G/48/', PN_48 / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/PN/G/32/', PN_32 / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/DN/G/64/', DN_64 / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/DN/G/48/', DN_48 / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/DN/G/32/', DN_32 / 2 + 0.5, i)
                    writer.add_images('TEST/FORWARD/INPUT/', test_image / 2 + 0.5, i)
            else:
                _, DN = DN_generator(test_image)

                if use_tensorboard:
                    writer.add_images('TEST/BACKWARD/DN/G/', DN / 2 + 0.5, i)
                    writer.add_images('TEST/BACKWARD/INPUT/', test_image / 2 + 0.5, i)
    else:
        print('Testing directory does not exist.')

    print('Finished Testing')

# tensorboard --logdir=/data/d0/y19/chyeung9/fyp1/submit/outputs/tensorboard/ --host=137.189.88.87 --port=6006
# tensorboard --logdir=C:\Users\User\PycharmProjects\pixelization\tensorboard --port=6006
# tensorboard --logdir=C:\Users\Lenovo\PycharmProjects\pixelization\tensorboard --port=6006
# tensorboard --logdir=C:\Users\Sinojapien\PycharmProjects\pixelization\tensorboard --port=6006

# https://jiaya.me/projects/rollguidance/index.html#Results
# https://www.kyprianidis.com/p/tpcg2008/
# https://gfx.cs.princeton.edu/pubs/Gerstner_2012_PIA/index.php
# https://gfx.cs.princeton.edu/pubs/Gerstner_2012_PIA/Gerstner_2012_PIA_full.pdf
# https://stackoverflow.com/questions/62174141/how-to-balance-the-generator-and-the-discriminator-performances-in-a-gan

# Quantization done
# jpeg: artifact, worse on uncompressed, weight? use loss? Random chance?
# floodfill result, fixed D Image buffer and GN D bugs, GN artifact

# invertibility
# quality factor 70, find new test cases, debug, find fork in https://github.com/zh217/torch-dct
# done, worse at normal, better at jpeg


# Try large mirror_image loss with no detach?
# similar effect as detach

# Disable mirror loss and see if grid effects over image (apply to small mirror loss)
# Bad converge, need large dataset and luck, grid is caused by poor GAN loss

# GAN in mirror
# some use? Need test on large dataset

# Large GAN and mirror loss: similar to no mirror

# very small gan: similar to very large L1/GRAD

# try lsgan? similar effect atm

# identity loss? not useful, bad if image is similar to cartoon

# d > g weight: similar to increased GAN loss

# no detach with GN train? idk

# Super PN: only colour gradient change (L1 GN_input, grad output)

# Uneven L1/GRAD: not sure, a bit better

# very small GAN/L1/GAN/MIRROR: PN not much effect, DN requires large mirror loss

# try GN again: trying
# fix bug that GAN has no effect: fixing, should have effect

# try resize again: seems good???

# disable mirror loss: definite take part in stablizing network
# very high gan for very long time:

# useless for detach GN

# not useful for 100 loss weight
# set target images to pixel_image_32: same for mirror?
# train gan on mirror rather than forward
# add white in floodfill

# cartoon target for GN, skipping strategy in PN/DN
