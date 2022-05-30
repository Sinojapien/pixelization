import os
import sys
import glob

from PIL import Image
from itertools import product

def tile(filename, dir_in, dir_out, width, height):
	name, ext = os.path.splitext(filename)
	img = Image.open(os.path.join(dir_in, filename))
	w, h = img.size

	grid = product(range(0, h - h % height, height), range(0, w - w % width, width))
	for i, j in grid:
		box = (j, i, j + width, i + height)
		out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
		img.crop(box).save(out)

# dir = 'new_data/pixel/pixel/Terrain (16x16).png'
# dir = 'data_backup/thai drum/162922.png'
#
# tile('162924.png', 'data_backup/thai drum/', 'new_data/cartoon/cartoon/', 80, 80)

# https://stackoverflow.com/questions/765736/how-to-use-pil-to-make-all-white-pixels-transparent

def whiten(dir, output_dir):
	for filename in glob.iglob(dir + '*.png', recursive=True):
		print(filename)
		img = Image.open(filename)
		img = img.convert("RGBA")
		datas = img.getdata()

		newData = []
		for item in datas:
			if item[3] < 255:
				# newData.append((255, 255, 255, 255))
				if item[3] == 0:
					newData.append((255, 255, 255, 255))
				else:
					ratio = 1 + ((255 - item[3]) / 255)
					ratio = ratio
					RGB = []
					RGB.append(int(item[0] * ratio))
					RGB.append(int(item[1] * ratio))
					RGB.append(int(item[2] * ratio))
					for i in range(len(RGB)):
						if RGB[i] > 255: RGB[i] = 255
					newData.append((RGB[0], RGB[1], RGB[2], 255))
			else:
				newData.append(item)

		img.putdata(newData)
		img.save(output_dir + os.path.basename(filename), "PNG")

# whiten('new_data/cartoon/cartoon/', 'gggg/')
# whiten('new_data/pixel/pixel/', 'gggg/')
whiten('split_input/', 'split_output/')