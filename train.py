import argparse
import os
import sys
from datetime import datetime
import json
import time
import tensorflow as tf

from reader import create_inputs
from model import RecogModel


BATCH_SIZE = 1
NUM_STEPS = 5000
LEARNING_RATE = 0.0005
INPUT_CHANNEL = 1
LOGDIR_ROOT = './logdir'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
RECOG_PARAMS = './recog_params.json'
L2_REGULARIZATION_STRENGTH = 0

def get_arguments():
	parser = argparse.ArgumentParser(description='recognition network')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
						help='How many image files to process at once.')
	parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
						help='Number of training steps.')
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
						help='Learning rate for training.')
	parser.add_argument('--input_channel', type=str, default=INPUT_CHANNEL,
						help='Number of input channel.')
	parser.add_argument('--recog_params', type=str, default=RECOG_PARAMS,
						help='JSON file with the network parameters.')
	parser.add_argument('--l2_regularization_strength', type=float,
						default=L2_REGULARIZATION_STRENGTH,
						help='Coefficient in the L2 regularization. '
						'Disabled by default')
	parser.add_argument('--logdir_root', type=str, default=LOGDIR_ROOT,
						help='Root directory to place the logging '
						'output and generated model. These are stored '
						'under the dated subdirectory of --logdir_root. '
						'Cannot use with --logdir.')
	return parser.parse_args()

def get_default_logdir(logdir_root):
	print(logdir_root)
	print(STARTED_DATESTRING)
	logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
	return logdir


def check_params(recog_params):
	if len(recog_params['dilations']) - len(recog_params['channels']) != 1:
		print("The length of 'dilations' must be greater then the length of 'channels' by 1.")
		return False
	if len(recog_params['kernel_size']) != len(recog_params['dilations']):
		print("The length of 'dilations' must be equal to the length of 'kernel_size'.")
		return False
	return True

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir))
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')

def main():
	args = get_arguments()

	with open(args.recog_params, 'r') as f:
		recog_params = json.load(f)

	# if check_params(recog_params) == False:
	# 	return

	logdir_root = args.logdir_root
	logdir = get_default_logdir(logdir_root)

	image, label_value, label_shape, label_index = create_inputs(input_channel=args.input_channel,
																 labels=recog_params['labels'],
																 dilations=recog_params['ctc_params']['cnn']['dilations'])

	queue = tf.FIFOQueue(256, ['uint8', 'uint8', 'uint8', 'uint8'])
	enqueue = queue.enqueue([image, label_value, label_shape, label_index])
	input_data = queue.dequeue()

	net = RecogModel(input_channel=args.input_channel,
					klass=len(recog_params['labels']),
					batch_size=args.batch_size,
					network_type=recog_params['network_type'],
					ctc_params=recog_params['ctc_params'],
					seq2seq_params=recog_params['seq2seq_params'])

	loss = net.loss(input_data)
	optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
	trainable = tf.trainable_variables()
	optim = optimizer.minimize(loss, var_list=trainable)

	# set up logging for tensorboard
	writer = tf.train.SummaryWriter(logdir)
	writer.add_graph(tf.get_default_graph())
	summaries = tf.merge_all_summaries()

	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)

	coord = tf.train.Coordinator()
	qr = tf.train.QueueRunner(queue, [enqueue])
	qr.create_threads(sess, coord=coord, start=True)
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	saver = tf.train.Saver()

	try:
		start_time = time.time()
		for step in range(args.num_steps):
			summary, loss_value, _ = sess.run([summaries, loss, optim])
			writer.add_summary(summary, step)

			print step

			if step % 100 == 0 and step > 0:
				duration = time.time() - start_time
				print('step {:d} - loss = {:.9f}, ({:.3f} sec/100 step)'.format(step, loss_value, duration))
				start_time = time.time()
				save(saver, sess, logdir, step)
				last_saved_step = step
	except KeyboardInterrupt:
		# Introduce a line break after ^C is displayed so save message
		# is on its own line.
		print()
	finally:
		# if step > last_saved_step:
			# save(saver, sess, logdir, step)
		coord.request_stop()
		coord.join(threads)

if __name__ == '__main__':
	main()
