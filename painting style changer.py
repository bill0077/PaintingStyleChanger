from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto

from tensorflow.compat.v1 import InteractiveSession

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import time
import functools
from datetime import datetime

#import socket
a=0
b=0

try:

	'''
	sock=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	a=1
	fr=open('/home/sjsh/ai_not_difficult/port.txt', 'r')
	port=int(fr.readline())
	fr.close()
	sock.bind(('',port))
	print('bind')
	sock.listen(1)
	print('listen')
	client, addr_info = sock.accept()
	b=1	
	print('accept')

	config = ConfigProto()

	config.gpu_options.allow_growth = True

	session = InteractiveSession(config=config)

	data=client.recv(65535)
	st=data.decode()

	print('recieved data : ', st)

	li=st.split(',')

	'''
			
	content_path = '/home/sjshmakers/ai_not_difficult/capture/sakurajima_mai.jpg'

	style_path = '/home/sjshmakers/ai_not_difficult/paints/'+li[1]+'.jpg'

	def load_img(path_to_img):
		max_dim = 512
		img = tf.io.read_file(path_to_img)
		img = tf.image.decode_image(img, channels=3)
		img = tf.image.convert_image_dtype(img, tf.float32)

		shape = tf.cast(tf.shape(img)[:-1], tf.float32)
		long_dim = max(shape)
		scale = max_dim / long_dim
		new_shape = tf.cast(shape * scale, tf.int32)

		img = tf.image.resize(img, new_shape)
		img = img[tf.newaxis, :]
		return img

	def imshow(image, title=None):
		if len(image.shape) > 3:
			image = tf.squeeze(image, axis=0)
			plt.imshow(image)
		if title:
	    		plt.title(title)

	content_image = load_img(content_path)
	style_image = load_img(style_path)

	x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
	x = tf.image.resize(x, (224, 224))
	vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
	prediction_probabilities = vgg(x)
	prediction_probabilities.shape

	predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
	[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

	vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

	print()
	for layer in vgg.layers:
		print(layer.name)

	# Content layer where will pull our feature maps
	content_layers = ['block5_conv2'] 

	# Style layer of interest
	style_layers = ['block1_conv1',
			'block2_conv1',
			'block3_conv1', 
			'block4_conv1', 
			'block5_conv1']

	num_content_layers = len(content_layers)
	num_style_layers = len(style_layers)

	def vgg_layers(layer_names):
		""" Creates a vgg model that returns a list of intermediate output values."""
	# Load our model. Load pretrained VGG, trained on imagenet data
		vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
		vgg.trainable = False
			  
		outputs = [vgg.get_layer(name).output for name in layer_names]

		model = tf.keras.Model([vgg.input], outputs)
		return model
	style_extractor = vgg_layers(style_layers)
	style_outputs = style_extractor(style_image*255)

	#Look at the statistics of each layer's output
	for name, output in zip(style_layers, style_outputs):
		print(name)
		print("  shape: ", output.numpy().shape)
		print("  min: ", output.numpy().min())
		print("  max: ", output.numpy().max())
		print("  mean: ", output.numpy().mean())
		print()

	def gram_matrix(input_tensor):
		result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
		input_shape = tf.shape(input_tensor)
		num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
		return result/(num_locations)

	class StyleContentModel(tf.keras.models.Model):
		def __init__(self, style_layers, content_layers):
			super(StyleContentModel, self).__init__()
			self.vgg =  vgg_layers(style_layers + content_layers)
			self.style_layers = style_layers
			self.content_layers = content_layers
			self.num_style_layers = len(style_layers)
			self.vgg.trainable = False

		def call(self, inputs):
			"Expects float input in [0,1]"
			inputs = inputs*255.0
			preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
			outputs = self.vgg(preprocessed_input)
			style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
			style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
			content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}

			style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
			    
			return {'content':content_dict, 'style':style_dict}

	extractor = StyleContentModel(style_layers, content_layers)

	results = extractor(tf.constant(content_image))

	style_results = results['style']

	print('Styles:')
	for name, output in sorted(results['style'].items()):
		print("  ", name)
		print("    shape: ", output.numpy().shape)
		print("    min: ", output.numpy().min())
		print("    max: ", output.numpy().max())
		print("    mean: ", output.numpy().mean())
		print()

	print("Contents:")
	for name, output in sorted(results['content'].items()):
		print("  ", name)
		print("    shape: ", output.numpy().shape)
		print("    min: ", output.numpy().min())
		print("    max: ", output.numpy().max())
		print("    mean: ", output.numpy().mean())

	style_targets = extractor(style_image)['style']
	content_targets = extractor(content_image)['content']

	image = tf.Variable(content_image)

	def clip_0_1(image):
		return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

	opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

	style_weight=1e-2
	content_weight=1e4

	def style_content_loss(outputs):
		style_outputs = outputs['style']
		content_outputs = outputs['content']
		style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
		style_loss *= style_weight / num_style_layers
		content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
		content_loss *= content_weight / num_content_layers
		loss = style_loss + content_loss
		return loss

	@tf.function()
	def train_step(image):
		with tf.GradientTape() as tape:
			outputs = extractor(image)
			loss = style_content_loss(outputs)

			grad = tape.gradient(loss, image)
			opt.apply_gradients([(grad, image)])
			image.assign(clip_0_1(image))

	train_step(image)
	train_step(image)
	train_step(image)

	import time
	start = time.time()

	epochs = 5
	steps_per_epoch = 100

	step = 0
	for n in range(epochs):
		for m in range(steps_per_epoch):
			step += 1
			train_step(image)
			print(step, "step : 1")

	end = time.time()
	print("Total time: {:.1f}".format(end-start))

	def high_pass_x_y(image):
		x_var = image[:,:,1:,:] - image[:,:,:-1,:]
		y_var = image[:,1:,:,:] - image[:,:-1,:,:]
		return x_var, y_var

	x_deltas, y_deltas = high_pass_x_y(content_image)

	fig=plt.figure(figsize=(14,10))

	x_deltas, y_deltas = high_pass_x_y(image)

	sobel = tf.image.sobel_edges(content_image)

	def total_variation_loss(image):
		x_deltas, y_deltas = high_pass_x_y(image)
		return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

	total_variation_weight=1e8

	@tf.function()
	def train_step(image):
		with tf.GradientTape() as tape:
			outputs = extractor(image)
			loss = style_content_loss(outputs)
			loss += total_variation_weight*total_variation_loss(image)
			grad = tape.gradient(loss, image)
			opt.apply_gradients([(grad, image)])
			image.assign(clip_0_1(image))

	image = tf.Variable(content_image)

	import time
	start = time.time()

	epochs = 5
	steps_per_epoch = 100

	step = 0
	for n in range(epochs):
		for m in range(steps_per_epoch):
			step += 1
			train_step(image)
			print(step, "step : 2")
			

	end = time.time()
	print("Total time: {:.1f}".format(end-start))
			
	now=datetime.now()

	display.clear_output(wait=True)
	imshow(image.read_value())
	fig.savefig('/home/sjsh/ai_not_difficult/result/'+str(now.day)+','+str(now.hour)+','+str(now.minute)+','+str(now.second)+'.png')
	plt.title("Train step: {}".format(step))
	plt.show()
			
	session.close()

	'''
	
	client.send(b'hello')
	
	port+=1
	fw=open('/home/sjsh/ai_not_difficult/port.txt', 'w')
	fw.write(str(port))
	fw.close()

	client.close()
	sock.close()
	'''

except:
	'''
	port+=1
	fw=open('/home/sjsh/ai_not_difficult/port.txt', 'w')
	fw.write(str(port))
	fw.close()
	if b:
		client.close()
	if a:
		sock.close()
	'''
	pass
