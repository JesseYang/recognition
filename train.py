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
NUM_STEPS = 50000
LEARNING_RATE = 0.0005
LR_DECAY_STEPS = 500
LR_DECAY_RATE = 1.0
MOMENTUM = 0.9
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
	parser.add_argument('--lr_decay_steps', type=float, default=LR_DECAY_STEPS,
						help='Learning rate decay steps.')
	parser.add_argument('--lr_decay_rate', type=float, default=LR_DECAY_RATE,
						help='Learning rate decay rate.')
	parser.add_argument('--momentum', type=float, default=MOMENTUM,
						help='Momentum for training.')
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

	height, image, label_value, label_shape, label_index = create_inputs(input_channel=args.input_channel,
																		 labels=recog_params['labels'],
																		 dilations=recog_params['ctc_params']['cnn']['dilations'],
																		 kernel_height=recog_params['ctc_params']['cnn']['kernel_height'],
																		 kernel_width=recog_params['ctc_params']['cnn']['kernel_width'],
																		 min_height=recog_params['min_height'],
																		 min_width_pad=recog_params['min_width_pad'])

	queue = tf.FIFOQueue(256, ['uint8', 'uint8', 'uint8', 'uint8'])
	enqueue = queue.enqueue([image, label_value, label_shape, label_index])
	input_data = queue.dequeue()

	net = RecogModel(input_channel=args.input_channel,
					 image_height=height,
					 klass=len(recog_params['labels']),
					 batch_size=args.batch_size,
					 network_type=recog_params['network_type'],
					 ctc_params=recog_params['ctc_params'],
					 seq2seq_params=recog_params['seq2seq_params'])

	loss = net.loss(input_data)
	global_step = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
											   global_step=global_step,
											   decay_steps=args.lr_decay_steps,
											   decay_rate=args.lr_decay_rate,
											   staircase=True)
	# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
	# 									   momentum=args.momentum)
	optimizer = tf.train.AdamOptimizer()
	trainable = tf.trainable_variables()
	optim = optimizer.minimize(loss, var_list=trainable, global_step=global_step)

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
	step_num = 100

	try:
		start_time = time.time()
		for step in range(args.num_steps):
			summary, loss_value, _ = sess.run([summaries, loss, optim])
			writer.add_summary(summary, step)

			print loss_value

			if step % step_num == 0 and step > 0:
				duration = time.time() - start_time
				print('step {:d} - loss = {:.9f}, ({:.3f} sec/{:d} step)'.format(step, loss_value, duration, step_num))
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
