import tensorflow as tf

def create_variable(name, shape):
	initializer = tf.contrib.layers.xavier_initializer_conv2d()
	variable = tf.Variable(initializer(shape=shape), name=name)
	return variable

def create_bias_variable(name, shape):
	initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	return tf.Variable(initializer(shape=shape), name)


class RecogModel(object):
	def __init__(self,
				 input_channel,
				 image_height,
				 klass,
				 batch_size,
				 network_type,
				 ctc_params,
				 seq2seq_params):
		self.input_channel = input_channel
		self.image_height = image_height
		self.klass = klass
		self.batch_size = batch_size
		self.network_type = network_type
		self.ctc_params = ctc_params
		self.seq2seq_params = seq2seq_params
		if self.network_type == 'ctc':
			self.ctc_params['cnn']['channels'].insert(0, self.input_channel)
			# the rnn is bi-directional, and the output size is as twice as the unit number in the json file
			self.ctc_params['full']['units'].insert(0, self.ctc_params['rnn']['units'][-1] * 2)
			self.ctc_params['full']['units'].append(self.klass + 1)
			self._calculate_rnn_input_size()
		self.variables = self._create_variables()

	def _calculate_rnn_input_size(self):
		receptive_field_height = 0
		for i, dilation in enumerate(self.ctc_params['cnn']['dilations']):
			receptive_field_height = receptive_field_height + dilation * (self.ctc_params['cnn']['kernel_height'][i] - 1) / 2
		cnn_output_height = self.image_height - receptive_field_height * 2
		cnn_output_channel = self.ctc_params['cnn']['channels'][-1] if len(self.ctc_params['cnn']['channels']) > 0 else 1
		self.rnn_input_size = cnn_output_height * cnn_output_channel

	def _create_variables(self):
		var = dict()

		if self.network_type == 'ctc':
			var['ctc'] = dict()
			# cnn part
			var['ctc']['cnn'] = dict()
			var['ctc']['cnn']['filters'] = list()
			for i, dilation in enumerate(self.ctc_params['cnn']['dilations']):
				var['ctc']['cnn']['filters'].append(create_variable('filter',
													[self.ctc_params['cnn']['kernel_height'][i],
													 self.ctc_params['cnn']['kernel_width'][i],
													 self.ctc_params['cnn']['channels'][i],
													 self.ctc_params['cnn']['channels'][i + 1]]))
			var['ctc']['cnn']['biases'] = list()
			for i, channel in enumerate(self.ctc_params['cnn']['channels']):
				if i == 0:
					continue
				var['ctc']['cnn']['biases'].append(create_bias_variable('bias', [channel]))
			# fully connected layer part
			var['ctc']['full'] = dict()
			var['ctc']['full']['weights'] = list()
			var['ctc']['full']['biases'] = list()
			for i, unit_num in enumerate(self.ctc_params['full']['units']):
				if i == 0:
					continue
				var['ctc']['full']['weights'].append(create_variable('weight',
													[self.ctc_params['full']['units'][i - 1],
													 self.ctc_params['full']['units'][i]]))
				if i != len(self.ctc_params['full']['units']) - 1:
					var['ctc']['full']['biases'].append(create_bias_variable('bias', [unit_num]))
		return var

	def _preprocess(self, input_data, generate=False):
		if generate == True:
			image = input_data
			image = tf.cast(tf.expand_dims(tf.expand_dims(image, 2), 0), tf.float32)
			label_value = None
			label_shape = None
			label_index = None
		else:
			image = input_data[0]
			image = tf.cast(tf.expand_dims(image, 0), tf.float32)
			label_value = input_data[1]
			label_value = tf.reshape(label_value, [-1])
			label_value = tf.cast(label_value, tf.int32)
			label_shape = input_data[2]
			label_shape = tf.reshape(label_shape, [-1])
			label_shape = tf.cast(label_shape, tf.int64)
			label_index = input_data[3]
			label_index = tf.reshape(label_index, [-1, 2])
			label_index = tf.cast(label_index, tf.int64)
		# tf.nn.conv2d(padding='SAME') always pads 0 to the input tensor,
		# thus make the value of the white pixels in the image 0
		image = 1.0 - image / 255.0
		return image, label_value, label_shape, label_index

	def _non_linear(self, subnet, input_tensor):
		if self.ctc_params[subnet]['non-linear'] == 'relu':
			return tf.nn.relu(input_tensor)
		elif self.ctc_params[subnet]['non-linear'] == 'tanh':
			return tf.nn.tanh(input_tensor)
		return tf.nn.sigmoid(input_tensor)

	def _create_network(self, input_data):
		current_layer = input_data
		if self.network_type == 'ctc':
			# cnn part
			for layer_idx, dilation in enumerate(self.ctc_params['cnn']['dilations']):
				conv = tf.nn.atrous_conv2d(value=current_layer,
										   filters=self.variables['ctc']['cnn']['filters'][layer_idx],
										   rate=dilation,
										   padding='VALID')
				with_bias = tf.nn.bias_add(conv, self.variables['ctc']['cnn']['biases'][layer_idx])
				current_layer = tf.nn.relu(with_bias)
			# rnn part
			shape = tf.shape(current_layer)
			# shape is [batch_size * height * width * channels]
			length = shape[2]
			cells_fw_list = []
			cells_bw_list = []
			for layer_idx, unit_num in enumerate(self.ctc_params['rnn']['units']):
				cells_fw_list.append(tf.nn.rnn_cell.GRUCell(unit_num))
				cells_bw_list.append(tf.nn.rnn_cell.GRUCell(unit_num))
			cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw_list)
			cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw_list)
			# transpose from [batch_size * height * width * channels] to [batch_size * width * height * channels]
			current_layer = tf.transpose(current_layer, perm=[0, 2, 1, 3])
			current_layer = tf.reshape(current_layer, [self.batch_size,
													   length,
													   self.rnn_input_size])
			current_layer, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw,
											cell_bw=cells_bw,
											inputs=current_layer,
											sequence_length=tf.expand_dims(length, 0),
											dtype=tf.float32)
			# fully connected part
			current_layer = tf.reshape(current_layer, [self.batch_size * length, -1])
			for layer_idx, unit_num in enumerate(self.ctc_params['full']['units']):
				if layer_idx == 0:
					continue
				current_layer = tf.matmul(current_layer, self.variables['ctc']['full']['weights'][layer_idx - 1])
				if layer_idx != len(self.ctc_params['full']['units']) - 1:
					with_bias = tf.nn.bias_add(current_layer, self.variables['ctc']['full']['biases'][layer_idx - 1])
					current_layer = self._non_linear('full', with_bias)
			current_layer = tf.reshape(current_layer, [self.batch_size, length, -1])
		return current_layer

	def loss(self, input_data):
		image, label_value, label_shape, label_index = self._preprocess(input_data)

		output = self._create_network(image)

		output = tf.reshape(output, [self.batch_size, -1, self.klass + 1])

		sparse_label = tf.SparseTensor(label_index, label_value, label_shape)
		loss = tf.nn.ctc_loss(inputs=output,
							  labels=sparse_label,
							  sequence_length=tf.expand_dims(tf.shape(output)[1], 0),
							  time_major=False)

		reduced_loss = tf.reduce_mean(loss)
		tf.scalar_summary('loss', reduced_loss)
		return reduced_loss

	def generate(self, image):
		image, _, _, _ = self._preprocess(input_data=image,
									generate=True)
		output = self._create_network(image)

		output = tf.reshape(output, [self.batch_size, -1, self.klass + 1])

		_output = tf.transpose(output, perm=[1, 0, 2])

		label_result = tf.nn.ctc_beam_search_decoder(inputs=_output,
													 sequence_length=tf.expand_dims(tf.shape(output)[1], 0),
													 beam_width=100)

		return label_result
