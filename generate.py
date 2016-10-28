import argparse
import os
import sys
import json
from scipy import misc
import tensorflow as tf

from model import RecogModel


BATCH_SIZE = 1
INPUT_CHANNEL = 1
RECOG_PARAMS = './recog_params.json'


def get_arguments():
	parser = argparse.ArgumentParser(description='Generation script')
	parser.add_argument('checkpoint', type=str,
						help='Which model checkpoint to generate from')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
						help='How many image files to process at once.')
	parser.add_argument('--input_channel', type=str, default=INPUT_CHANNEL,
						help='Number of input channel.')
	parser.add_argument('--recog_params', type=str, default=RECOG_PARAMS,
						help='JSON file with the network parameters.')
	parser.add_argument('--image', type=str,
						help='The image waiting for processed.')
	parser.add_argument('--out_path', type=str,
						help='The output path for the result label.')
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

	with open(args.recog_params, 'r') as f:
		recog_params = json.load(f)

	# if check_params(recog_params) == False:
	#   return

	net = RecogModel(input_channel=args.input_channel,
					klass=len(recog_params['labels']),
					batch_size=args.batch_size,
					network_type=recog_params['network_type'],
					ctc_params=recog_params['ctc_params'],
					seq2seq_params=recog_params['seq2seq_params'])

	input_image = tf.placeholder(tf.uint8)
	label_result = net.generate(input_image)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, args.checkpoint)

	input_image_data = misc.imread(args.image, mode='L')

	label_result_data = sess.run(label_result, feed_dict={input_image: input_image_data})

	out_label = []
	for l in label_result_data[0][0].values:
		if l < len(recog_params['labels']):
			out_label.append(recog_params['labels'][l])
		else:
			out_label.append(' ')
	result_str = "".join(out_label)
	print result_str
	if args.out_path != None:
		text_file = open(args.out_path, 'w')
		text_file.write(result_str)
		text_file.close()

if __name__ == '__main__':
	main()
