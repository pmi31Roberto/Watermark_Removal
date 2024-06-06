import os
import torch
import telebot
import numpy as np
import cv2
from torch import nn, optim
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid
from torchsummary import summary

class SkipEncoderDecoder(nn.Module):
    def __init__(self, input_depth, num_channels_down=[128]*5, num_channels_up=[128]*5, num_channels_skip=[128]*5):
        super(SkipEncoderDecoder, self).__init__()

        self.model = nn.Sequential()
        model_tmp = self.model

        for i in range(len(num_channels_down)):
            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                model_tmp.add_module(str(len(model_tmp) + 1), Concat(1, skip, deeper))
            else:
                model_tmp.add_module(str(len(model_tmp) + 1), deeper)

            model_tmp.add_module(str(len(model_tmp) + 1), nn.BatchNorm2d(num_channels_skip[i] + (num_channels_up[i + 1] if i < (len(num_channels_down) - 1) else num_channels_down[i])))

            if num_channels_skip[i] != 0:
                skip.add_module(str(len(skip) + 1), Conv2dBlock(input_depth, num_channels_skip[i], 1, bias=False))

            deeper.add_module(str(len(deeper) + 1), Conv2dBlock(input_depth, num_channels_down[i], 3, 2, bias=False))
            deeper.add_module(str(len(deeper) + 1), Conv2dBlock(num_channels_down[i], num_channels_down[i], 3, bias=False))

            deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
                k = num_channels_down[i]
            else:
                deeper.add_module(str(len(deeper) + 1), deeper_main)
                k = num_channels_up[i + 1]

            deeper.add_module(str(len(deeper) + 1), nn.Upsample(scale_factor=2, mode='nearest'))

            model_tmp.add_module(str(len(model_tmp) + 1), Conv2dBlock(num_channels_skip[i] + k, num_channels_up[i], 3, 1, bias=False))
            model_tmp.add_module(str(len(model_tmp) + 1), Conv2dBlock(num_channels_up[i], num_channels_up[i], 1, bias=False))

            input_depth = num_channels_down[i]
            model_tmp = deeper_main

        self.model.add_module(str(len(self.model) + 1), nn.Conv2d(num_channels_up[0], 3, 1, bias=True))
        self.model.add_module(str(len(self.model) + 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class DepthwiseSeperableConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super(DepthwiseSeperableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(input_channels, input_channels, groups=input_channels, **kwargs)
        self.pointwise = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super(Conv2dBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size - 1) / 2)),
            DepthwiseSeperableConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


def input_noise(INPUT_DEPTH, spatial_size, scale=1./10):
    shape = [1, INPUT_DEPTH, spatial_size[0], spatial_size[1]]
    return torch.rand(*shape) * scale


def pil_to_np_array(pil_image):
    ar = np.array(pil_image)
    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32) / 255.


def np_to_torch_array(np_array):
    return torch.from_numpy(np_array)[None, :]


def torch_to_np_array(torch_array):
    return torch_array.detach().cpu().numpy()[0]


def read_image(path, image_size=-1):
    pil_image = Image.open(path)
    return pil_image


def save_image(np_array, step):
    pil_image = Image.fromarray((np_array * 255.0).transpose(1, 2, 0).astype('uint8'), 'RGB')
    pil_image.save(f'progress/{str(step).zfill(len(str(TRAINING_STEPS)))}.png')


def crop_image(image, crop_factor=64):
    shape = (image.size[0] - image.size[0] % crop_factor, image.size[1] - image.size[1] % crop_factor)
    bbox = [int((image.shape[0] - shape[0])/2), int((image.shape[1] - shape[1])/2), int((image.shape[0] + shape[0])/2), int((image.shape[1] + shape[1])/2)]
    return image.crop(bbox)


def get_image_grid(images, nrow=3):
    torch_images = [torch.from_numpy(x) for x in images]
    grid = make_grid(torch_images, nrow)
    return grid.numpy()


def visualize_sample(*images_np, nrow=3, size_factor=10):
    c = max(x.shape[0] for x in images_np)
    images_np = [x if (x.shape[0] == c) else np.concatenate([x, x, x], axis=0) for x in images_np]
    grid = get_image_grid(images_np, nrow)
    plt.figure(figsize=(len(images_np) + size_factor, 12 + size_factor))
    plt.axis('off')
    plt.imshow(grid.transpose(1, 2, 0))
    plt.show()


def max_dimension_resize(image_pil, mask_pil, max_dim):
    w, h = image_pil.size
    aspect_ratio = w / h
    if w > max_dim:
        h = int((h / w) * max_dim)
        w = max_dim
    elif h > max_dim:
        w = int((w / h) * max_dim)
        h = max_dim
    return image_pil.resize((w, h)), mask_pil.resize((w, h))


def preprocess_images(image_path, mask_path, max_dim):
    image_pil = read_image(image_path).convert('RGB')
    mask_pil = read_image(mask_path).convert('RGB')

    image_pil, mask_pil = max_dimension_resize(image_pil, mask_pil, max_dim)

    image_np = pil_to_np_array(image_pil)
    mask_np = pil_to_np_array(mask_pil)

    print('Visualizing mask overlap...')

    visualize_sample(image_np, mask_np, image_np * mask_np, nrow=3, size_factor=10)

    return image_np, mask_np


def remove_watermark(image_path, mask_path, max_dim, reg_noise, input_depth, lr, num_iter):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    image_np, mask_np = preprocess_images(image_path, mask_path, max_dim)

    net = SkipEncoderDecoder(input_depth, num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4])

    net = net.cuda()
    net_input = input_noise(input_depth, image_np.shape[1:]).cuda().detach()

    summary(net, input_size=(input_depth, image_np.shape[1], image_np.shape[2]))

    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of params: ', s)

    mse = torch.nn.MSELoss().cuda()
    img_var = torch.from_numpy(image_np).cuda().unsqueeze(0)
    mask_var = torch.from_numpy(mask_np).cuda().unsqueeze(0)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    p = [x for x in net.parameters()]

    optimize = optim.Adam(p, lr=lr)

    out_img = None

    for i in tqdm(range(num_iter)):
        net_input = net_input_saved + reg_noise * noise.normal_()
        out = net(net_input)
        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()
        optimize.step()
        optimize.zero_grad()
        out_img = torch_to_np_array(out)
        if i % (num_iter // 10) == 0:
            save_image(out_img, i)

    print('Final result...')
    visualize_sample(image_np, out_img, image_np * mask_np, image_np * mask_np + out_img * (1 - mask_np), nrow=3, size_factor=10)

    return torch_to_np_array(out)


TOKEN = 'YOUR_TELEGRAM_BOT_API_TOKEN'
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def start_command(message):
    bot.send_message(message.chat.id, 'Hello! I am an image inpainting bot. Send me an image and a mask and I will remove the watermark!')


@bot.message_handler(content_types=['photo'])
def handle_photos(message):
    bot.reply_to(message, 'Please send the mask now.')


@bot.message_handler(content_types=['document'])
def handle_documents(message):
    bot.reply_to(message, 'Please send the mask now.')


@bot.message_handler(func=lambda message: True, content_types=['photo', 'document'])
def handle_files(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id if message.photo else message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        image_path = os.path.join('downloads', file_info.file_path.split('/')[-1])
        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        if message.photo:
            bot.reply_to(message, 'Image received. Please send the mask.')
        else:
            bot.reply_to(message, 'Mask received. Processing...')

            mask_path = image_path
            image_path = 'downloads/image.jpg'  # Replace with the correct image path

            out_img = remove_watermark(image_path, mask_path, max_dim=512, reg_noise=0.03, input_depth=32, lr=0.001, num_iter=5000)

            output_image = Image.fromarray((out_img * 255).astype(np.uint8))
            output_image_path = 'output/output_image.png'
            output_image.save(output_image_path)

            with open(output_image_path, 'rb') as photo:
                bot.send_photo(message.chat.id, photo)
    except Exception as e:
        bot.reply_to(message, 'An error occurred: {}'.format(e))


bot.polling(none_stop=True)
