import os.path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from os.path import exists

import architecture as arch
from architecture import data  # import data as data

import sys
import datetime

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
drop_PN = False
PN_input_channel = 6
PN_backward_mode = 'random'
combine_loss = False
small_batch = False

lr_scheduler_initial_step = 100

# Handling input arguments
print(f'Arguments: {sys.argv}')
if len(sys.argv) > 1: data.update_path(sys.argv[1])  # path = '/data/d0/y19/chyeung9/fyp1/submit/'
if len(sys.argv) > 2: data.update_output_path(sys.argv[2])  # '/research/dept8/fyp21/ttw2104/pixelization/'
if len(sys.argv) > 3: num_epochs = int(sys.argv[3])
if len(sys.argv) > 4: version = sys.argv[4]

if len(version) > 0:
    if int(version) >= 0:
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
        pixel_resize_mode = 'mixed'

        data.max_epoch_size = 900

        GN_option.train = True
        PN_option.train = True
        DN_option.train = True

        # Try to acheive 0.5 vs 0.0
        # GN_option.weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0, net=1.0, null=1.0, gan=1.0/1.25, d_gan=1.0/1.25)
        # PN_option.weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0, net=1.0, null=1.0, gan=1.0/10, d_gan=1.0/10)
        # DN_option.weights = arch.Weights(l1=1.0/1.5, grad=1.0/2, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)
        # GN_option.weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/1.25, d_gan=1.0/1.25)
        # PN_option.weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/15, d_gan=1.0/15)
        # DN_option.weights = arch.Weights(l1=1.0, grad=1.0/2, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2)

        # GN_option.weights = arch.Weights(l1=1.0*1.5, grad=1.0*0.9, image=1.0*4/4/5, net=1.0, null=1.0, gan=1.0/1.25, d_gan=1.0/1.25)  # set, grad/d/d_gan 1 to 1.25 to 1.5
        # PN_option.weights = arch.Weights(l1=1.0, grad=1.0*2, image=1.0, net=1.0, null=1.0, gan=1.0/1.5, d_gan=1.0/1.5)  # set, mirror loss?
        # DN_option.weights = arch.Weights(l1=1.0, grad=1.0/1.5, image=1.0, net=1.0, null=1.0, gan=1.0/2*2/1.5, d_gan=1.0/2*2/1.5)

        # GN_option.set_options(l1=True, grad=True, gan=True, image=False, network=False, idt=False)
        # PN_option.set_options(l1=True, grad=True, gan=True, image=True, network=False, idt=False)
        # DN_option.set_options(l1=True, grad=True, gan=True, image=True, network=False, idt=False)
        # GN_option.set_interval(g=2, d=1)  # set
        # PN_option.set_interval(g=3, d=1)
        # DN_option.set_interval(g=2, d=1)

        # GN_option.weights = arch.Weights(l1=1.0*1.5, grad=1.0*0.9, image=1.0, net=1.0, null=1.0, gan=1.0/1.25/5, d_gan=1.0/1.25/5)
        # PN_option.weights = arch.Weights(l1=1.0, grad=1.0*2, image=1.0, net=1.0, null=1.0, gan=1.0/1.5/5, d_gan=1.0/1.5/5)
        # DN_option.weights = arch.Weights(l1=1.0, grad=1.0/1.5, image=1.0, net=1.0, null=1.0, gan=1.0/1.5/5, d_gan=1.0/1.5/5)
        # GN_option.weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2/48)
        # PN_option.weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/8, d_gan=1.0/8/4)
        # DN_option.weights = arch.Weights(l1=1.0*2, grad=1.0*2, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2/15)

        # GN_option.weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/10, d_gan=1.0/10/25)
        # PN_option.weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0/20, d_gan=1.0/20/15)
        # DN_option.weights = arch.Weights(l1=1.0, grad=1.0, image=1.0, net=1.0, null=1.0, gan=1.0, d_gan=1.0/10)
        # GN_option.weights = arch.Weights(l1=1.0, grad=1.0/10*2, image=1.0, net=1.0, null=1.0, gan=1.0/6, d_gan=1.0/6/25)
        # PN_option.weights = arch.Weights(l1=1.0, grad=1.0/10*2, image=1.0, net=1.0, null=1.0, gan=1.0/8, d_gan=1.0/8/25)
        # DN_option.weights = arch.Weights(l1=1.0, grad=1.0/10*2, image=1.0, net=1.0, null=1.0, gan=1.0/2, d_gan=1.0/2/25)
        GN_option.weights = arch.Weights(l1=1.0, grad=1.0/1.5*1.5, image=1.0, net=1.0, null=1.0, gan=1.0/10, d_gan=1.0/10/25)
        PN_option.weights = arch.Weights(l1=1.0, grad=1.0/1.5*1.5, image=1.0, net=1.0, null=1.0, gan=1.0/15, d_gan=1.0/15/25)
        DN_option.weights = arch.Weights(l1=1.0, grad=1.0/1.5*1.5, image=1.0, net=1.0, null=1.0, gan=1.0/4/2, d_gan=1.0/4/2/25)
        GN_option.set_options(l1=True, grad=True, gan=True, image=False, network=False, idt=False)
        PN_option.set_options(l1=True, grad=True, gan=True, image=True, network=False, idt=False)
        DN_option.set_options(l1=True, grad=True, gan=True, image=True, network=False, idt=False)
        GN_option.set_interval(g=1, d=1)
        PN_option.set_interval(g=1, d=1)
        DN_option.set_interval(g=1, d=1)

        PN_input_channel = 3
        # PN_backward_mode = 'all'
        arch.use_lsgan = True
        arch.detach_network = True
        small_batch = False

        drop_PN = False
        lr_scheduler_initial_step = 0
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
    GN_generator = arch.PixelNet(input_channel=3, output_channel=3)
    GN_generator = nn.DataParallel(GN_generator)
    GN_generator = GN_generator.to(device)
    GN_generator.train()
    GN_discriminator = arch.Discriminator(data.input_resolution)
    GN_discriminator = nn.DataParallel(GN_discriminator)
    GN_discriminator = GN_discriminator.to(device)
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
    PN_generator = arch.PixelNet(input_channel=PN_input_channel, output_channel=3)
    PN_generator = nn.DataParallel(PN_generator)
    PN_generator = PN_generator.to(device)
    PN_generator.train()
    PN_discriminator = arch.Discriminator(data.input_resolution)
    PN_discriminator = nn.DataParallel(PN_discriminator)
    PN_discriminator = PN_discriminator.to(device)
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
    DN_generator = nn.DataParallel(DN_generator)
    DN_generator = DN_generator.to(device)
    DN_generator.train()
    DN_discriminator = arch.Discriminator(data.input_resolution)
    DN_discriminator = nn.DataParallel(DN_discriminator)
    DN_discriminator = DN_discriminator.to(device)
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
GN_buffer_256_x = arch.ImageBuffer()
GN_buffer_256_y = arch.ImageBuffer()
GN_buffer_256 = arch.ImageBuffer()
GN_buffer_64 = arch.ImageBuffer()
GN_buffer_48 = arch.ImageBuffer()
GN_buffer_32 = arch.ImageBuffer()
PN_buffer_256 = arch.ImageBuffer()
PN_buffer_64 = arch.ImageBuffer()
PN_buffer_48 = arch.ImageBuffer()
PN_buffer_32 = arch.ImageBuffer()
DN_buffer = arch.ImageBuffer()

schedulers = []
if GN_G_optimizer is not None:
    schedulers.append(arch.create_lr_scheduler(GN_G_optimizer, lr_scheduler_initial_step, num_epochs))
    schedulers.append(arch.create_lr_scheduler(GN_D_optimizer, lr_scheduler_initial_step, num_epochs))
if PN_G_optimizer is not None:
    schedulers.append(arch.create_lr_scheduler(PN_G_optimizer, lr_scheduler_initial_step, num_epochs))
    schedulers.append(arch.create_lr_scheduler(PN_D_optimizer, lr_scheduler_initial_step, num_epochs))
if DN_G_optimizer is not None:
    schedulers.append(arch.create_lr_scheduler(DN_G_optimizer, lr_scheduler_initial_step, num_epochs))
    schedulers.append(arch.create_lr_scheduler(DN_D_optimizer, lr_scheduler_initial_step, num_epochs))

# Load Section
if exists(model_dir):
    checkpoint = torch.load(model_dir)
    print(checkpoint.keys())

    current_epochs = checkpoint['epoch']
    current_iter = checkpoint['iteration']

    if GN_generator is not None:
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
        # if PN_input_channel == 3:
        #     PN_model_dir = data.get_model_directory() + 'checkpoint' + '_baseline' + '.pth'
        #     print(PN_model_dir)
        #     PN_checkpoint = torch.load(PN_model_dir)
        #     print(PN_checkpoint.keys())
        #     if PN_option.load_generator:
        #         PN_generator.load_state_dict(PN_checkpoint['PN_generator'])
        #         PN_G_optimizer.load_state_dict(PN_checkpoint['PN_G_optimizer'])
        #     if PN_option.load_discriminator and False:
        #         PN_discriminator.load_state_dict(PN_checkpoint['PN_discriminator'])
        #         PN_D_optimizer.load_state_dict(PN_checkpoint['PN_D_optimizer'])
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
        # if int(version) == 22:
        #     DN_checkpoint = torch.load(data.get_model_directory() + 'checkpoint' + '11' + '.pth')
        #     print(DN_checkpoint.keys())
        #     print(data.get_model_directory() + 'checkpoint' + '11' + '.pth')
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

    checkpoint = None

    print("Loaded model checkpoint successfully.")
else:
    print('Model checkpoint not found.')

version_prefix = 'CUSTOM/'


def train_discriminator(discriminator: nn.Module, buffer: arch.ImageBuffer, generated_image: torch.Tensor, target_image: torch.Tensor, option: arch.CombinedOptions):
    if not option.gan: return 0.0
    # On input image
    D_labels_target = discriminator(target_image)
    D_loss_target = arch.adversarial_criterion(D_labels_target, True, weight=option.weights.d_gan)

    # On generated image
    D_labels_generated = discriminator(buffer.get_image(generated_image.detach()))
    D_loss_generated = arch.adversarial_criterion(D_labels_generated, False, weight=option.weights.d_gan)

    if use_tensorboard and option.print:
        prefix = version_prefix + option.tag + '/'
        writer.add_scalar(prefix + 'D/LOSS/' + option.title + '/TARGET', D_loss_target / option.weights.d_gan, current_iter)
        writer.add_scalar(prefix + 'D/LOSS/' + option.title + '/GENERATED', D_loss_generated / option.weights.d_gan, current_iter)

    D_loss = (D_loss_target + D_loss_generated) * 0.5
    return D_loss


def train_adversarial_loss(discriminator: nn.Module, generated_image: torch.Tensor, option: arch.CombinedOptions):
    if not option.gan: return 0.0
    G_labels_generated = discriminator(generated_image)
    G_gan_loss = arch.adversarial_criterion(G_labels_generated, True, weight=option.weights.gan)

    if use_tensorboard and option.print:
        prefix = version_prefix + option.tag + '/'
        writer.add_scalar(prefix + 'G/LOSS/GAN/' + option.title, G_gan_loss / option.weights.gan, current_iter)

    return G_gan_loss


def train_grad_loss(generated_image: torch.Tensor, target_image: torch.Tensor, option: arch.CombinedOptions):
    if not option.grad: return 0.0
    G_grad_loss = arch.gradient_criterion(generated_image, target_image, option.weights.grad, variation=False)

    if use_tensorboard and option.print:
        prefix = version_prefix + option.tag + '/'
        writer.add_scalar(prefix + "G/LOSS/GRAD/" + option.title, G_grad_loss, current_iter)

    return G_grad_loss


def train_grad_loss_tuple(generated_image: torch.Tensor, target_gradient_x: torch.Tensor, target_gradient_y: torch.Tensor, option: arch.CombinedOptions):
    if not option.grad: return 0.0
    generated_image_x, generated_image_y = arch.image_gradient_net(generated_image)
    G_grad_loss_x = arch.l1_criterion(generated_image_x, target_gradient_x, option.weights.grad)
    G_grad_loss_y = arch.l1_criterion(generated_image_y, target_gradient_y, option.weights.grad)
    G_grad_loss = (G_grad_loss_x + G_grad_loss_y) * 0.5

    if use_tensorboard and option.print:
        prefix = version_prefix + option.tag + '/'
        writer.add_scalar(prefix + "G/LOSS/GRAD/" + option.title, G_grad_loss, current_iter)

    return G_grad_loss


def train_l1_loss(generated_image: torch.Tensor, target_image: torch.Tensor, option: arch.CombinedOptions):
    if not option.l1: return 0.0
    G_l1_loss = arch.l1_criterion(generated_image, target_image, option.weights.l1)

    if use_tensorboard and option.print:
        prefix = version_prefix + option.tag + '/'
        writer.add_scalar(prefix + "G/LOSS/L1/" + option.title, G_l1_loss, current_iter)

    return G_l1_loss


def train_mirror_image_loss(generated_image: torch.Tensor, target_image: torch.Tensor, option: arch.CombinedOptions):
    if not option.mirror_image: return 0.0
    G_mirror_loss = arch.l1_criterion(generated_image, target_image, option.weights.image)

    if use_tensorboard and option.print:
        prefix = version_prefix + option.tag + '/'
        writer.add_scalar(prefix + "G/LOSS/MIRROR/" + option.title, G_mirror_loss, current_iter)

    return G_mirror_loss


def train_identity_loss(model, identity_image: torch.Tensor, option: arch.CombinedOptions):
    _, generated_image = model(identity_image)
    identity_loss = arch.l1_criterion(generated_image, identity_image, option.weights.idt)

    if use_tensorboard:
        prefix = version_prefix + option.tag + '/'
        if option.save:
            writer.add_scalar(prefix + "G/LOSS/IDENTITY", identity_loss, current_iter)
        if option.print:
            writer.add_images(prefix + 'OUTPUT/IDENTITY', generated_image / 2 + 0.5, current_iter)

    return identity_loss


def train_GN_discriminator():
    # pixel_gradient_x, pixel_gradient_y = arch.image_gradient_net(pixel_image)
    # pixel_gradient_x = data.normalize(pixel_gradient_x, -1, 1)
    # pixel_gradient_y = data.normalize(pixel_gradient_y, -1, 1)
    # GN_option.title = 'X'
    # GN_D_loss_x = train_discriminator(GN_discriminator, GN_buffer_256_x, F_GN_generated_image_x, pixel_gradient_x, GN_option)
    # GN_option.title = 'Y'
    # GN_D_loss_y = train_discriminator(GN_discriminator, GN_buffer_256_y, F_GN_generated_image_y, pixel_gradient_y, GN_option)
    # loss_buffer.GN_D_loss = GN_D_loss_x + GN_D_loss_y
    # use pixel_image
    GN_option.title = '256'
    GN_D_loss_256 = train_discriminator(GN_discriminator, GN_buffer_256, F_GN_generated_image_256, pixel_image, GN_option)
    GN_option.title = '64'
    GN_D_loss_64 = train_discriminator(GN_discriminator, GN_buffer_64, F_GN_generated_image_64, pixel_image_64, GN_option)
    GN_option.title = '48'
    GN_D_loss_48 = train_discriminator(GN_discriminator, GN_buffer_48, F_GN_generated_image_48, pixel_image_48, GN_option)
    GN_option.title = '32'
    GN_D_loss_32 = train_discriminator(GN_discriminator, GN_buffer_32, F_GN_generated_image_32, pixel_image_32, GN_option)
    # loss_buffer.GN_D_loss = GN_D_loss_64 + GN_D_loss_48 + GN_D_loss_32
    # return GN_D_loss_64 + GN_D_loss_48 + GN_D_loss_32 + GN_D_loss_256
    return GN_D_loss_256 + GN_D_loss_64 + GN_D_loss_48 + GN_D_loss_32


def train_GN_generator():
    GN_option.title = '256/'
    GN_G_loss_gan_256 = train_adversarial_loss(GN_discriminator, F_GN_generated_image_256, GN_option)
    GN_G_loss_l1_256 = train_l1_loss(F_GN_generated_image_256, cartoon_image, GN_option)
    GN_G_loss_grad_256 = train_grad_loss(F_GN_generated_image_256, cartoon_image, GN_option)
    GN_G_loss_256 = GN_G_loss_gan_256 + GN_G_loss_l1_256 + GN_G_loss_grad_256

    GN_option.title = '64/'
    GN_G_loss_gan_64 = train_adversarial_loss(GN_discriminator, F_GN_generated_image_64, GN_option)
    GN_G_loss_l1_64 = train_l1_loss(F_GN_generated_image_64, cartoon_image_64, GN_option)
    GN_G_loss_grad_64 = train_grad_loss(F_GN_generated_image_64, cartoon_image_64, GN_option)
    GN_G_loss_64 = GN_G_loss_gan_64 + GN_G_loss_l1_64 + GN_G_loss_grad_64

    GN_option.title = '48/'
    GN_G_loss_gan_48 = train_adversarial_loss(GN_discriminator, F_GN_generated_image_48, GN_option)
    GN_G_loss_l1_48 = train_l1_loss(F_GN_generated_image_48, cartoon_image_48, GN_option)
    GN_G_loss_grad_48 = train_grad_loss(F_GN_generated_image_48, cartoon_image_48, GN_option)
    GN_G_loss_48 = GN_G_loss_gan_48 + GN_G_loss_l1_48 + GN_G_loss_grad_48

    GN_option.title = '32/'
    GN_G_loss_gan_32 = train_adversarial_loss(GN_discriminator, F_GN_generated_image_32, GN_option)
    GN_G_loss_l1_32 = train_l1_loss(F_GN_generated_image_32, cartoon_image_32, GN_option)
    GN_G_loss_grad_32 = train_grad_loss(F_GN_generated_image_32, cartoon_image_32, GN_option)
    GN_G_loss_32 = GN_G_loss_gan_32 + GN_G_loss_l1_32 + GN_G_loss_grad_32

    # GN_option.title = '256/'
    # GN_G_loss_mirror_image_256 = train_mirror_image_loss(B_GN_generated_image_256, pixel_image, GN_option)
    # GN_G_loss_mirror_image_256 = 0
    GN_option.title = '64/'
    GN_G_loss_mirror_image_64 = train_mirror_image_loss(B_GN_generated_image_64, pixel_image_64, GN_option)
    GN_option.title = '48/'
    GN_G_loss_mirror_image_48 = train_mirror_image_loss(B_GN_generated_image_48, pixel_image_48, GN_option)
    GN_option.title = '32/'
    GN_G_loss_mirror_image_32 = train_mirror_image_loss(B_GN_generated_image_32, pixel_image_32, GN_option)
    GN_G_loss_mirror_image = data.choose_random((GN_G_loss_mirror_image_64, GN_G_loss_mirror_image_48, GN_G_loss_mirror_image_32))

    # loss_buffer.GN_G_loss = GN_G_loss_64 + GN_G_loss_48 + GN_G_loss_32
    return GN_G_loss_64 + GN_G_loss_48 + GN_G_loss_32 + GN_G_loss_256 + GN_G_loss_mirror_image


def train_PN_discriminator():
    target_images = [pixel_image, pixel_image, pixel_image, pixel_image]
    # target_images = [pixel_image, pixel_image_64, pixel_image_48, pixel_image_32]
    # target_images = [pixel_image_64, pixel_image_48, pixel_image_32, data.desample(pixel_image, scale=16.0)]
    PN_option.title = '256'
    PN_D_loss_256 = train_discriminator(PN_discriminator, PN_buffer_256, F_PN_generated_image_256, target_images[0], PN_option)
    PN_option.title = '64'
    PN_D_loss_64 = train_discriminator(PN_discriminator, PN_buffer_64, F_PN_generated_image_64, target_images[1], PN_option)
    PN_option.title = '48'
    PN_D_loss_48 = train_discriminator(PN_discriminator, PN_buffer_48, F_PN_generated_image_48, target_images[2], PN_option)
    PN_option.title = '32'
    PN_D_loss_32 = train_discriminator(PN_discriminator, PN_buffer_32, F_PN_generated_image_32, target_images[3], PN_option)

    # return PN_D_loss_64 + PN_D_loss_48 + PN_D_loss_32 + PN_D_loss_256
    return PN_D_loss_256 + PN_D_loss_64 + PN_D_loss_48 + PN_D_loss_32


def train_PN_generator():
    PN_option.title = '256/'
    PN_G_loss_gan_256 = train_adversarial_loss(PN_discriminator, F_PN_generated_image_256, PN_option)
    if PN_input_channel == 3:
        PN_G_loss_l1_256 = train_l1_loss(F_PN_generated_image_256, F_GN_generated_image_256.detach(), PN_option)
    else:
        PN_G_loss_l1_256 = train_l1_loss(F_PN_generated_image_256, cartoon_image, PN_option)
    PN_G_loss_grad_256 = train_grad_loss(F_PN_generated_image_256, F_GN_generated_image_256.detach(), PN_option)
    # PN_G_loss_grad_256 = train_grad_loss_tuple(F_PN_generated_image, F_GN_generated_image_x.detach(), F_GN_generated_image_y.detach(), PN_option)

    PN_option.title = '64/'
    PN_G_loss_gan_64 = train_adversarial_loss(PN_discriminator, F_PN_generated_image_64, PN_option)
    if PN_input_channel == 3:
        PN_G_loss_l1_64 = train_l1_loss(F_PN_generated_image_64, F_GN_generated_image_64.detach(), PN_option)
    else:
        PN_G_loss_l1_64 = train_l1_loss(F_PN_generated_image_64, cartoon_image, PN_option)
    PN_G_loss_grad_64 = train_grad_loss(F_PN_generated_image_64, F_GN_generated_image_64.detach(), PN_option)
    # PN_G_loss_grad_64 = train_grad_loss_tuple(F_PN_generated_image_64, cartoon_image_64_x, cartoon_image_64_y, PN_option)

    PN_option.title = '48/'
    PN_G_loss_gan_48 = train_adversarial_loss(PN_discriminator, F_PN_generated_image_48, PN_option)
    if PN_input_channel == 3:
        PN_G_loss_l1_48 = train_l1_loss(F_PN_generated_image_48, F_GN_generated_image_48.detach(), PN_option)
    else:
        PN_G_loss_l1_48 = train_l1_loss(F_PN_generated_image_48, cartoon_image, PN_option)
    PN_G_loss_grad_48 = train_grad_loss(F_PN_generated_image_48, F_GN_generated_image_48.detach(), PN_option)
    # PN_G_loss_grad_48 = train_grad_loss_tuple(F_PN_generated_image_48, cartoon_image_48_x, cartoon_image_48_y, PN_option)

    PN_option.title = '32/'
    PN_G_loss_gan_32 = train_adversarial_loss(PN_discriminator, F_PN_generated_image_32, PN_option)
    if PN_input_channel == 3:
        PN_G_loss_l1_32 = train_l1_loss(F_PN_generated_image_32, F_GN_generated_image_32.detach(), PN_option)
    else:
        PN_G_loss_l1_32 = train_l1_loss(F_PN_generated_image_32, cartoon_image, PN_option)
    PN_G_loss_grad_32 = train_grad_loss(F_PN_generated_image_32, F_GN_generated_image_32.detach(), PN_option)
    # PN_G_loss_grad_32 = train_grad_loss_tuple(F_PN_generated_image_32, cartoon_image_32_x, cartoon_image_32_y, PN_option)

    mirror_images = [pixel_image, pixel_image, pixel_image, pixel_image]
    # PN_option.title = '256/'
    # PN_G_loss_mirror_image_256 = train_mirror_image_loss(B_PN_generated_image_256, mirror_images[0], PN_option)
    PN_option.title = '64/'
    PN_G_loss_mirror_image_64 = train_mirror_image_loss(B_PN_generated_image_64, mirror_images[1], PN_option)
    PN_option.title = '48/'
    PN_G_loss_mirror_image_48 = train_mirror_image_loss(B_PN_generated_image_48, mirror_images[2], PN_option)
    PN_option.title = '32/'
    PN_G_loss_mirror_image_32 = train_mirror_image_loss(B_PN_generated_image_32, mirror_images[3], PN_option)
    if PN_backward_mode == 'all':
        PN_G_loss_mirror_image = PN_G_loss_mirror_image_64 + PN_G_loss_mirror_image_48 + PN_G_loss_mirror_image_32
    else:
        PN_G_loss_mirror_image = data.choose_random((PN_G_loss_mirror_image_64, PN_G_loss_mirror_image_48, PN_G_loss_mirror_image_32))

    PN_G_loss_gan = PN_G_loss_gan_64 + PN_G_loss_gan_48 + PN_G_loss_gan_32
    PN_G_loss_l1 = PN_G_loss_l1_64 + PN_G_loss_l1_48 + PN_G_loss_l1_32
    PN_G_loss_grad = PN_G_loss_grad_64 + PN_G_loss_grad_48 + PN_G_loss_grad_32
    PN_G_loss_256 = PN_G_loss_gan_256 + PN_G_loss_l1_256 + PN_G_loss_grad_256
    return PN_G_loss_gan + PN_G_loss_l1 + PN_G_loss_grad + PN_G_loss_256 + PN_G_loss_mirror_image


def train_DN_discriminator():
    DN_option.title = ''
    return train_discriminator(DN_discriminator, DN_buffer, B_DN_generated_image, cartoon_image, DN_option)
    # DN_option.title = 'MIRROR'
    # loss_buffer.DN_D_loss = train_discriminator(DN_discriminator, DN_buffer, F_DN_generated_image, cartoon_image, DN_option)


def train_DN_generator():
    DN_option.title = ''
    DN_G_loss_gan = train_adversarial_loss(DN_discriminator, B_DN_generated_image, DN_option)
    DN_G_loss_l1 = train_l1_loss(B_DN_generated_image, pixel_image, DN_option)
    DN_G_loss_grad = train_grad_loss(B_DN_generated_image, pixel_image, DN_option)
    DN_option.title = ''
    DN_G_loss_mirror_image = train_mirror_image_loss(F_DN_generated_image, cartoon_image, DN_option)
    return DN_G_loss_gan + DN_G_loss_l1 + DN_G_loss_grad + DN_G_loss_mirror_image


# Training Section
for epoch in range(num_epochs):
    cartoon_iter = iter(cartoon_dataloader)
    pixel_iter = iter(pixel_dataloader)

    for i in range(n_total_steps):
        loss_buffer.clear()

        cartoon_image, _ = next(cartoon_iter)
        pixel_image, _ = next(pixel_iter)
        cartoon_image = cartoon_image.to(device)
        pixel_image = pixel_image.to(device)

        saveCondition = (i + 1) % int(n_total_steps) == 0 and (epoch % 3 == 0 or epoch == num_epochs)
        printCondition = (i + 1) % int(n_total_steps) == 0 or i == 0
        addScalarCondition = (i + 1) % int(n_total_steps / 4) == 0 or i == 0
        GN_option.print = addScalarCondition
        GN_option.save = saveCondition
        PN_option.print = addScalarCondition
        PN_option.save = saveCondition
        DN_option.print = addScalarCondition
        DN_option.save = saveCondition
        GN_train_G_condition = (i + 1) % int(GN_option.gan_g_interval) == 0 or i == 0
        GN_train_D_condition = (i + 1) % int(GN_option.gan_d_interval) == 0 or i == 0
        PN_train_G_condition = (i + 1) % int(PN_option.gan_g_interval) == 0 or i == 0
        PN_train_D_condition = (i + 1) % int(PN_option.gan_d_interval) == 0 or i == 0
        DN_train_G_condition = (i + 1) % int(DN_option.gan_g_interval) == 0 or i == 0
        DN_train_D_condition = (i + 1) % int(DN_option.gan_d_interval) == 0 or i == 0

        if printCondition:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}]')
            writer.add_scalar("LR", schedulers[0].get_last_lr()[0], current_iter)

        # Forward
        pixel_image_64 = data.desample(pixel_image, 4.0)
        pixel_image_48 = data.desample(pixel_image, 16.0 / 3.0)
        pixel_image_32 = data.desample(pixel_image, 8.0)

        cartoon_image_64 = data.desample(cartoon_image, 4.0)
        cartoon_image_48 = data.desample(cartoon_image, 16.0 / 3.0)
        cartoon_image_32 = data.desample(cartoon_image, 8.0)
        _, F_GN_generated_image_256 = GN_generator(cartoon_image)
        _, F_GN_generated_image_64 = GN_generator(cartoon_image_64)
        _, F_GN_generated_image_48 = GN_generator(cartoon_image_48)
        _, F_GN_generated_image_32 = GN_generator(cartoon_image_32)
        # cartoon_gradient_x, cartoon_gradient_y = arch.get_image_gradient(cartoon_image, normalize=True)
        # _, F_GN_generated_image_x = GN_generator(cartoon_gradient_x)
        # _, F_GN_generated_image_y = GN_generator(cartoon_gradient_y)

        # cartoon_image_256 = data.desample(cartoon_image, 8.0 / 3.0)
        # cartoon_image_64 = data.desample(cartoon_image, 4.0)
        # cartoon_image_48 = data.desample(cartoon_image, 16.0 / 3.0)
        # cartoon_image_32 = data.desample(cartoon_image, 8.0)
        # _, F_PN_generated_image = PN_generator(cartoon_image_256)
        # _, F_PN_generated_image_64 = PN_generator(cartoon_image_64)
        # _, F_PN_generated_image_48 = PN_generator(cartoon_image_48)
        # _, F_PN_generated_image_32 = PN_generator(cartoon_image_32)

        if not drop_PN:
            if PN_input_channel == 3:
                _, F_PN_generated_image_256 = PN_generator(F_GN_generated_image_256)
                _, F_PN_generated_image_64 = PN_generator(F_GN_generated_image_64)
                _, F_PN_generated_image_48 = PN_generator(F_GN_generated_image_48)
                _, F_PN_generated_image_32 = PN_generator(F_GN_generated_image_32)
            else:
                _, F_PN_generated_image_256 = PN_generator(torch.cat((cartoon_image, F_GN_generated_image_256), dim=1))
                _, F_PN_generated_image_64 = PN_generator(torch.cat((cartoon_image, F_GN_generated_image_64), dim=1))
                _, F_PN_generated_image_48 = PN_generator(torch.cat((cartoon_image, F_GN_generated_image_48), dim=1))
                _, F_PN_generated_image_32 = PN_generator(torch.cat((cartoon_image, F_GN_generated_image_32), dim=1))
            # cartoon_image_64_x, cartoon_image_64_y = arch.get_image_gradient(data.desample(cartoon_image, 4.0), normalize=True)
            # cartoon_image_48_x, cartoon_image_48_y = arch.get_image_gradient(data.desample(cartoon_image, 16.0 / 3.0), normalize=True)
            # cartoon_image_32_x, cartoon_image_32_y = arch.get_image_gradient(data.desample(cartoon_image, 8.0), normalize=True)
            # _, F_PN_generated_image = PN_generator(torch.cat((cartoon_image, F_GN_generated_image_x, F_GN_generated_image_y), dim=1))
            # _, F_PN_generated_image_64 = PN_generator(torch.cat((cartoon_image, cartoon_image_64_x, cartoon_image_64_y), dim=1))
            # _, F_PN_generated_image_48 = PN_generator(torch.cat((cartoon_image, cartoon_image_48_x, cartoon_image_48_y), dim=1))
            # _, F_PN_generated_image_32 = PN_generator(torch.cat((cartoon_image, cartoon_image_32_x, cartoon_image_32_y), dim=1))

            F_PN_output = data.choose_random((F_PN_generated_image_64, F_PN_generated_image_48, F_PN_generated_image_32))
        else:
            F_PN_output = data.choose_random((F_GN_generated_image_64, F_GN_generated_image_48, F_GN_generated_image_32))
        _, F_DN_generated_image = DN_generator(F_PN_output)

        # Backward
        _, B_DN_generated_image = DN_generator(pixel_image)
        B_DN_generated_image_64 = data.desample(B_DN_generated_image, 4.0)
        B_DN_generated_image_48 = data.desample(B_DN_generated_image, 16.0 / 3.0)
        B_DN_generated_image_32 = data.desample(B_DN_generated_image, 8.0)

        # _, B_GN_generated_image_256 = GN_generator(B_DN_generated_image)
        _, B_GN_generated_image_64 = GN_generator(B_DN_generated_image_64)
        _, B_GN_generated_image_48 = GN_generator(B_DN_generated_image_48)
        _, B_GN_generated_image_32 = GN_generator(B_DN_generated_image_32)

        if not drop_PN:
            if PN_input_channel == 3:
                # _, B_PN_generated_image_256 = PN_generator(B_GN_generated_image_256)
                _, B_PN_generated_image_64 = PN_generator(B_GN_generated_image_64)
                _, B_PN_generated_image_48 = PN_generator(B_GN_generated_image_48)
                _, B_PN_generated_image_32 = PN_generator(B_GN_generated_image_32)
            else:
                # B_GN_output = data.choose_random((B_GN_generated_image_64, B_GN_generated_image_48, B_GN_generated_image_32))
                # _, B_PN_generated_image_256 = PN_generator(torch.cat((B_DN_generated_image.detach(), B_GN_generated_image_256), dim=1))
                _, B_PN_generated_image_64 = PN_generator(torch.cat((B_DN_generated_image.detach(), B_GN_generated_image_64), dim=1))
                _, B_PN_generated_image_48 = PN_generator(torch.cat((B_DN_generated_image.detach(), B_GN_generated_image_48), dim=1))
                _, B_PN_generated_image_32 = PN_generator(torch.cat((B_DN_generated_image.detach(), B_GN_generated_image_32), dim=1))

        if GN_option.train:
            # Training Discriminator
            if GN_train_D_condition:
                loss_buffer.GN_D_loss = train_GN_discriminator()

            # Training Generator
            if GN_train_G_condition:
                loss_buffer.GN_G_loss = train_GN_generator()

        if PN_option.train and not drop_PN:
            # Training Discriminator
            if PN_train_D_condition:
                loss_buffer.PN_D_loss = train_PN_discriminator()

            # Training Generator
            if PN_train_G_condition:
                loss_buffer.PN_G_loss = train_PN_generator()

        if DN_option.train:
            # Training Discriminator
            if DN_train_D_condition:
                loss_buffer.DN_D_loss = train_DN_discriminator()

            # Training Generator
            if DN_train_G_condition:
                loss_buffer.DN_G_loss = train_DN_generator()

        if use_tensorboard and saveCondition:
            prefix = version_prefix + 'A/OVERALL/'
            writer.add_images(prefix + 'INPUT/CARTOON', cartoon_image / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'INPUT/PIXEL', pixel_image / 2 + 0.5, current_iter)
            prefix = version_prefix + 'B/GN/'
            # writer.add_images(prefix + 'OUTPUT/256/X', F_GN_generated_image_x / 2 + 0.5, current_iter)
            # writer.add_images(prefix + 'OUTPUT/256/Y', F_GN_generated_image_y / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'OUTPUT/256', F_GN_generated_image_256 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'OUTPUT/64', F_GN_generated_image_64 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'OUTPUT/48', F_GN_generated_image_48 / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'OUTPUT/32', F_GN_generated_image_32 / 2 + 0.5, current_iter)
            if not drop_PN:
                prefix = version_prefix + 'C/PN/'
                writer.add_images(prefix + 'OUTPUT/256', F_PN_generated_image_256 / 2 + 0.5, current_iter)
                writer.add_images(prefix + 'OUTPUT/64', F_PN_generated_image_64 / 2 + 0.5, current_iter)
                writer.add_images(prefix + 'OUTPUT/48', F_PN_generated_image_48 / 2 + 0.5, current_iter)
                writer.add_images(prefix + 'OUTPUT/32', F_PN_generated_image_32 / 2 + 0.5, current_iter)
                # writer.add_images(prefix + 'MIRROR/256', B_PN_generated_image_256 / 2 + 0.5, current_iter)
                writer.add_images(prefix + 'MIRROR/64', B_PN_generated_image_64 / 2 + 0.5, current_iter)
                writer.add_images(prefix + 'MIRROR/48', B_PN_generated_image_48 / 2 + 0.5, current_iter)
                writer.add_images(prefix + 'MIRROR/32', B_PN_generated_image_32 / 2 + 0.5, current_iter)
            prefix = version_prefix + 'D/DN/'
            writer.add_images(prefix + 'OUTPUT', B_DN_generated_image / 2 + 0.5, current_iter)
            writer.add_images(prefix + 'MiRROR', F_DN_generated_image / 2 + 0.5, current_iter)

        cartoon_image = None
        pixel_image = None
        F_GN_generated_image_256 = None
        F_GN_generated_image_64 = None
        F_GN_generated_image_48 = None
        F_GN_generated_image_32 = None
        if not drop_PN:
            F_PN_generated_image_256 = None
            F_PN_generated_image_64 = None
            F_PN_generated_image_48 = None
            F_PN_generated_image_32 = None
            B_PN_generated_image_256 = None
            B_PN_generated_image_64 = None
            B_PN_generated_image_48 = None
            B_PN_generated_image_32 = None
        B_DN_generated_image = None
        F_DN_generated_image = None

        if combine_loss:
            GN_G_optimizer.zero_grad()
            GN_D_optimizer.zero_grad()
            PN_G_optimizer.zero_grad()
            PN_D_optimizer.zero_grad()
            DN_G_optimizer.zero_grad()
            DN_D_optimizer.zero_grad()
            loss_buffer.backward(loss_buffer.get_total_loss())
            if GN_option.train:
                GN_G_optimizer.step()
                GN_D_optimizer.step()
            if PN_option.train:
                PN_G_optimizer.step()
                PN_D_optimizer.step()
            if DN_option.train:
                DN_G_optimizer.step()
                DN_D_optimizer.step()
        else:
            if GN_option.train and GN_train_G_condition:
                GN_G_optimizer.zero_grad()
                loss_buffer.backward(loss_buffer.GN_G_loss, retain_graph=True)
                # loss_buffer.GN_G_loss.backward(retain_graph=True)
                GN_G_optimizer.step()
            if PN_option.train and PN_train_G_condition and not drop_PN:
                PN_G_optimizer.zero_grad()
                loss_buffer.backward(loss_buffer.PN_G_loss, retain_graph=True)
                # loss_buffer.PN_G_loss.backward(retain_graph=True)
                PN_G_optimizer.step()
            if DN_option.train and DN_train_G_condition:
                DN_G_optimizer.zero_grad()
                loss_buffer.backward(loss_buffer.DN_G_loss)
                # loss_buffer.DN_G_loss.backward()
                DN_G_optimizer.step()
            if GN_option.train and GN_option.gan and GN_train_D_condition:
                GN_D_optimizer.zero_grad()
                loss_buffer.backward(loss_buffer.GN_D_loss)
                # loss_buffer.GN_D_loss.backward()
                GN_D_optimizer.step()
            if PN_option.train and PN_option.gan and PN_train_D_condition and not drop_PN:
                PN_D_optimizer.zero_grad()
                loss_buffer.backward(loss_buffer.PN_D_loss)
                # loss_buffer.PN_D_loss.backward()
                PN_D_optimizer.step()
            if DN_option.train and DN_option.gan and DN_train_D_condition:
                DN_D_optimizer.zero_grad()
                loss_buffer.backward(loss_buffer.DN_D_loss)
                # loss_buffer.DN_D_loss.backward()
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

    data.clear_cache()

    if use_tensorboard:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=tensorboard_log_dir + '_test')

    GN_generator.eval()
    PN_generator.eval()
    DN_generator.eval()

    if os.path.exists(data.paths.test_dir):
        # Test with cartoon
        test_dataloader = data.get_dataloader('test/', apply_train_transform=False)
        test_iter = iter(test_dataloader)

        with torch.no_grad():
            for i in range(len(test_dataloader)):
                test_image, test_label = next(test_iter)
                test_image = test_image.to(device)

                if test_label != 2:
                    test_64 = data.desample(test_image, 4.0)
                    test_48 = data.desample(test_image, 16.0 / 3.0)
                    test_32 = data.desample(test_image, 8.0)

                    _, GN_256 = GN_generator(test_image)
                    _, GN_64 = GN_generator(test_64)
                    _, GN_48 = GN_generator(test_48)
                    _, GN_32 = GN_generator(test_32)

                    if use_tensorboard:
                        writer.add_images('TEST/INPUT/FORWARD/64/', test_64 / 2 + 0.5, i)
                        writer.add_images('TEST/INPUT/FORWARD/48/', test_48 / 2 + 0.5, i)
                        writer.add_images('TEST/INPUT/FORWARD/32/', test_32 / 2 + 0.5, i)

                        writer.add_images('TEST/GN/FORWARD/256/', GN_256 / 2 + 0.5, i)
                        writer.add_images('TEST/GN/FORWARD/64/', GN_64 / 2 + 0.5, i)
                        writer.add_images('TEST/GN/FORWARD/48/', GN_48 / 2 + 0.5, i)
                        writer.add_images('TEST/GN/FORWARD/32/', GN_32 / 2 + 0.5, i)
                    test_64 = None
                    test_48 = None
                    test_32 = None

                    if not drop_PN:
                        if PN_input_channel == 3:
                            _, PN_256 = PN_generator(GN_256)
                            _, PN_64 = PN_generator(GN_64)
                            _, PN_48 = PN_generator(GN_48)
                            _, PN_32 = PN_generator(GN_32)
                        else:
                            _, PN_256 = PN_generator(torch.cat((test_image, GN_256), dim=1))
                            _, PN_64 = PN_generator(torch.cat((test_image, GN_64), dim=1))
                            _, PN_48 = PN_generator(torch.cat((test_image, GN_48), dim=1))
                            _, PN_32 = PN_generator(torch.cat((test_image, GN_32), dim=1))
                        if use_tensorboard:
                            writer.add_images('TEST/PN/FORWARD/256/', PN_256 / 2 + 0.5, i)
                            writer.add_images('TEST/PN/FORWARD/64/', PN_64 / 2 + 0.5, i)
                            writer.add_images('TEST/PN/FORWARD/48/', PN_48 / 2 + 0.5, i)
                            writer.add_images('TEST/PN/FORWARD/32/', PN_32 / 2 + 0.5, i)
                        PN_256 = None
                        PN_64 = None
                        PN_48 = None
                        PN_32 = None

                    if use_tensorboard:
                        writer.add_images('TEST/INPUT/FORWARD/256/', test_image / 2 + 0.5, i)
                    test_image = None

                    _, DN_256 = DN_generator(GN_256)
                    _, DN_64 = DN_generator(GN_64)
                    _, DN_48 = DN_generator(GN_48)
                    _, DN_32 = DN_generator(GN_32)

                    if use_tensorboard:
                        writer.add_images('TEST/DN/FORWARD/256/', DN_256 / 2 + 0.5, i)
                        writer.add_images('TEST/DN/FORWARD/64/', DN_64 / 2 + 0.5, i)
                        writer.add_images('TEST/DN/FORWARD/48/', DN_48 / 2 + 0.5, i)
                        writer.add_images('TEST/DN/FORWARD/32/', DN_32 / 2 + 0.5, i)
                else:
                    _, DN = DN_generator(test_image)

                    if use_tensorboard:
                        writer.add_images('TEST/DN/BACKWARD/', DN / 2 + 0.5, i)
                        writer.add_images('TEST/INPUT/BACKWARD/', test_image / 2 + 0.5, i)
    else:
        print('Testing directory does not exist.')

    print('Finished Testing')

# tensorboard --logdir=C:\Users\Sinojapien\PycharmProjects\pixelization\tensorboard --port=6006
# , potentially RGB to YIQ colour space
# report division of labour, different parts
