import tensorflow as tf
from tensorflow.contrib import ffmpeg
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def preprocess_images(height, width_pad):
	for (dirpath, dirnames, filenames) in os.walk('training_set/images'):
		for filename in filenames:
			image = misc.imread('training_set/images/' + filename, mode='L')
			cur_height = image.shape[0]
			cur_width = image.shape[1]
			if height > cur_height:
				# should pad in the vertical direction
				top_pad = (height - cur_height) / 2
				bottom_pad = (height - cur_height) - top_pad
				pad_image = np.pad(array=image,
								   pad_width=((top_pad, bottom_pad), (width_pad, width_pad)),
								   mode='constant',
								   constant_values=(255, 255))
			else:
				# should resise to fit the vertical direction
				resized_image = misc.imresize(image, (height, image.shape[1]))
				pad_image = np.pad(array=resized_image,
								   pad_width=((0, 0), (width_pad, width_pad)),
								   mode='constant',
								   constant_values=(255, 255))
			misc.imsave('training_set/pad_images/' + filename, pad_image)

def preprocess_labels(labels):
	for (dirpath, dirnames, filenames) in os.walk('training_set/labels'):
		for filename in filenames:
			value_list = []
			label_str_file = open('training_set/labels/' + filename)
			label_str = label_str_file.read()
			for i in range(len(label_str)):
				if label_str[i] not in labels:
					continue
				value_list.append(labels.index(label_str[i]))
			value_file = open('training_set/labels/' + filename.replace(".txt", ".dat"), "wb")
			value_array = bytearray(value_list)
			value_file.write(value_array)
			value_file.close()

			label_len = len(value_list)
			shape_list = [1, label_len]
			shape_file = open('training_set/labels/' + filename.replace(".txt", ".sha"), "wb")
			shape_array = bytearray(shape_list)
			shape_file.write(shape_array)
			shape_file.close()

			index_list = []
			for i in range(len(value_list)):
				index_list.append(0)
				index_list.append(i)
			index_file = open('training_set/labels/' + filename.replace(".txt", ".ind"), "wb")
			index_array = bytearray(index_list)
			index_file.write(index_array)
			index_file.close()

def image_label_list():
	image_name_list = [ ]
	label_value_name_list = [ ]
	label_shape_name_list = [ ]
	label_index_name_list = [ ]
	for (dirpath, dirnames, filenames) in os.walk('training_set/pad_images'):
		image_name_list.extend(map(lambda x: 'training_set/pad_images/' + x, filenames))
		label_value_name_list.extend(map(lambda x: 'training_set/labels/' + x.replace('.png', '.dat'), filenames))
		label_shape_name_list.extend(map(lambda x: 'training_set/labels/' + x.replace('.png', '.sha'), filenames))
		label_index_name_list.extend(map(lambda x: 'training_set/labels/' + x.replace('.png', '.ind'), filenames))
		break
	return image_name_list, label_value_name_list, label_shape_name_list, label_index_name_list

def create_inputs(input_channel, labels, dilations):
	receptive_field = 0
	for dilation in dilations:
		receptive_field = receptive_field + dilation
	width_pad = receptive_field
	receptive_field = receptive_field * 2 + 1
	height = receptive_field
	preprocess_images(height, width_pad)
	preprocess_labels(labels)

	image_name_list, label_value_name_list, label_shape_name_list, label_index_name_list = image_label_list()

	seed = np.random.randint(1000)

	image_name_queue = tf.train.string_input_producer(image_name_list, seed=seed)
	label_value_name_queue = tf.train.string_input_producer(label_value_name_list, seed=seed)
	label_shape_name_queue = tf.train.string_input_producer(label_shape_name_list, seed=seed)
	label_index_name_queue = tf.train.string_input_producer(label_index_name_list, seed=seed)

	image_reader = tf.WholeFileReader()
	_, image_content = image_reader.read(image_name_queue)
	image_tensor = tf.image.decode_png(image_content, channels=input_channel)

	label_value_reader = tf.WholeFileReader()
	_, label_value_content = label_value_reader.read(label_value_name_queue)
	label_value_tensor = tf.decode_raw(label_value_content, tf.uint8)

	label_shape_reader = tf.WholeFileReader()
	_, label_shape_content = label_shape_reader.read(label_shape_name_queue)
	label_shape_tensor = tf.decode_raw(label_shape_content, tf.uint8)

	label_index_reader = tf.WholeFileReader()
	_, label_index_content = label_index_reader.read(label_index_name_queue)
	label_index_tensor = tf.decode_raw(label_index_content, tf.uint8)

	return image_tensor, label_value_tensor, label_shape_tensor, label_index_tensor
