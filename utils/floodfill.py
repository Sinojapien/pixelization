import os
import sys
import glob

import numpy as np

from PIL import Image
from PIL import ImageDraw
from itertools import product


def rand_int(value, range=10):
    return value + np.random.randint(range) - int(range/2)


base_colours = [
    (215, 215, 215),
    (215, 125, 125),
    (125, 215, 125),
    (125, 125, 215),
    (215, 215, 125),
    (215, 125, 215),
    (125, 215, 215),
]
colours = []
for colour in base_colours:
    r, g, b = colour
    colours.append(colour)
    colours.append((r + rand_int(40), g + rand_int(40), b + rand_int(40)))
    colours.append((r - rand_int(40), g - rand_int(40), b - rand_int(40)))
    colours.append((r - rand_int(80), g - rand_int(80), b - rand_int(80)))
    colours.append((r - rand_int(115), g - rand_int(115), b - rand_int(115)))
for i in range(int(len(colours) / 15)):
    colours.append((255, 255, 255))
colours = [(255, 255, 255)]
num_of_colour = len(colours)


def floodfill(dir, output_dir, threshold=0, override=False):
    # https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageDraw.html#floodfill
    count = 1
    for filename in glob.iglob(dir + '*.png', recursive=True):
        print(f"{count}. " + filename)
        img = Image.open(filename)
        img = img.convert("RGB")  # not work for "RGBA"
        w, h = img.size

        pos_list = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        pos = pos_list[np.random.randint(len(pos_list))]
        fill_colour = colours[np.random.randint(num_of_colour)]
        chance = np.random.rand() < 0.2

        # for pos in pos_list:
        if chance and img.getpixel(pos) != (255, 255, 255):
            # print(fill_colour)
            ImageDraw.floodfill(img, pos, fill_colour, thresh=threshold)

        img.save(output_dir + os.path.basename(filename), "PNG")
        count += 1


# floodfill('new_data/floodfill/pixel/images_old/', 'new_data/floodfill/pixel/images/')
# floodfill('new_data/floodfill/cartoon/images_old/', 'new_data/floodfill/cartoon/images/')
# floodfill('new_data/floodfill/pixel/images/', 'new_data/pixel/images/')
# floodfill('new_data/floodfill/cartoon/images/', 'new_data/cartoon/images/')
# floodfill('new_data/floodfill/pixel_floodfill/images/', 'new_data/floodfill/pixel/images/')
# floodfill('new_data/floodfill/cartoon_floodfill/images/', 'new_data/floodfill/cartoon/images/')
# floodfill('new_data/pixel/pixel/', 'new_data/floodfill/pixel/images/')
# floodfill('new_data/cartoon/cartoon/', 'new_data/floodfill/cartoon/images/')
# floodfill('new_data/floodfill/cartoon/images_backup/', 'new_data/floodfill/cartoon/images/', threshold=160*3, override=True)

# def flood_fill(x, y, old, new):
#     # we need the x and y of the start position, the old value,
#     # and the new value    # the flood fill has 4 parts
#     # firstly, make sure the x and y are inbounds
#     if x < 0 or x >= len(field[0]) or y < 0 or y >= len(field):
#         return  # secondly, check if the current position equals the old value
#     if field[y][x] != old:
#         return
#
#     # thirdly, set the current position to the new value
#     field[y][x] = new  # fourthly, attempt to fill the neighboring positions
#     flood_fill(x + 1, y, old, new)
#     flood_fill(x - 1, y, old, new)
#     flood_fill(x, y + 1, old, new)
#     flood_fill(x, y - 1, old, new)
