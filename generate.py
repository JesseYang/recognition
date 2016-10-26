import argparse
import os
import sys
import json
from scipy import misc
import tensorflow as tf

from model import FragModel

BATCH_SIZE = 1
KLASS = 7
INPUT_CHANNEL = 1
FRAG_PARAMS = './recog_params.json'

def get_arguments():
	parser = argparse.ArgumentParser(description='Generation script')
	parser.add_argument('checkpoint', type=str,
						help='Which model checkpoint to generate from')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
						help='How many image files to process at once.')
	parser.add_argument('--klass', type=str, default=KLASS,
						help='Number of segmentation classes.')
	parser.add_argument('--input_channel', type=str, default=INPUT_CHANNEL,
						help='Number of input channel.')
	parser.add_argument('--recog_params', type=str, default=FRAG_PARAMS,
						help='JSON file with the network parameters.')
	parser.add_argument('--image', type=str,
						help='The limage waiting for processed.')
	parser.add_argument('--out_path', type=str,
						help='The output path for the segmentation result image')
	return parser.parse_args()

def check_params(recog_params):
	if len(recog_params['dilations']) - len(recog_params['channels']) != 1:
		print("The length of 'dilations' must be greater then the length of 'channels' by 1.")
		return False
	if len(recog_params['kernel_size']) != len(recog_params['dilations']):
		print("The length of 'dilations' must be equal to the length of 'kernel_size'.")
		return False
	return True

def main():
	args = get_arguments()

	with open("./recog_params.json", 'r') as f:
		recog_params = json.load(f)

	if check_params(recog_params) == False:
		return

	net = FragModel(
		input_channel=args.input_channel,
		klass=args.klass,
		batch_size=args.batch_size,
		kernel_size=recog_params['kernel_size'],
		dilations=recog_params['dilations'],
		channels=recog_params['channels'])

	input_image = tf.placeholder(tf.uint8)
	output_image = net.generate(input_image)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, args.checkpoint)

	input_image_data = misc.imread(args.image, mode='L')

	output_image_data = sess.run(output_image, feed_dict={input_image: input_image_data})
	height, width = input_image_data.shape

	receptive_field = 0
	for dilation in recog_params['dilations']:
		receptive_field = receptive_field + dilation
	width_pad = receptive_field

	for x in range(width_pad, width - width_pad):
		for y in range(height):
			if input_image_data[y][x] == 255:
				if output_image_data[0][0][x - width_pad] > 0:
					input_image_data[y][x] = 255 / 7 * output_image_data[0][0][x - width_pad]
	misc.imsave(args.out_path, input_image_data)

if __name__ == '__main__':
	main()
