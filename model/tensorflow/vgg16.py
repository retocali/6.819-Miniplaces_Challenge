import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 48
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.00001
dropout = 0.5
training_iters = 10000
step_display = 50
step_save = 10000
path_save = './vggnet/sessions/model.ckpt'
start_from = ''

def batch_norm_layer(x, train_phase, scope_bn):
	return batch_norm(x, decay=0.9, center=True, scale=True,
	updates_collections=None,
	is_training=train_phase,
	reuse=None,
	trainable=True,
	scope=scope_bn)
	
def vggnet(x, keep_dropout, train_phase):
	weights = {
            # Conv Layers 3x3: 64
	    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=np.sqrt(2./(3*3*64)))),
	    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2./(3*3*64)))),
            # Conv Layers 3x3 : 128
            'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=np.sqrt(2./(3*3*128)))),
	    'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2./(3*3*128)))),
            # Conv Layers 3x3 : 256 
            'wc5': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=np.sqrt(2./(3*3*256)))),
	    'wc6': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
            # Conv Layers 3x3 : 512
	    'wc7': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=np.sqrt(2./(3*3*512)))),
            'wc8': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
            'wc9': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
            'wc10':tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
            # FC 2048
	    'wfc1': tf.Variable(tf.random_normal([512, 4096], stddev=np.sqrt(2./4096))),
	    'wfc2': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
            # FC 1000
            'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./1000)))
	}

	biases = {
	    'bo': tf.Variable(tf.ones(100))
	}

	#2 Conv64 + ReLU + Pool
	conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 7, 7, 1], padding='SAME')
	conv1 = tf.nn.relu(conv1)
        conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 3, 3, 1], padding='SAME')
	conv2 = tf.nn.relu(conv2)
        pool1 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

	#2 Conv128 + ReLU + Conv
	conv3 = tf.nn.conv2d(pool1, weights['wc3'], strides=[1, 7, 7, 1], padding='SAME')
	conv3 = tf.nn.relu(conv3)
        conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 3, 3, 1], padding='SAME')
	conv4 = tf.nn.relu(conv4)
        pool2 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

	#2 Conv256 + ReLU + Conv
	conv5 = tf.nn.conv2d(pool2, weights['wc5'], strides=[1, 7, 7, 1], padding='SAME')
	conv5 = tf.nn.relu(conv5)
        conv6 = tf.nn.conv2d(conv5, weights['wc6'], strides=[1, 3, 3, 1], padding='SAME')
	conv6 = tf.nn.relu(conv6)
        pool3 = tf.nn.max_pool(conv6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

	#4 Conv512 + ReLU + Conv
	conv7 = tf.nn.conv2d(pool3, weights['wc7'], strides=[1, 7, 7, 1], padding='SAME')
	conv7 = tf.nn.relu(conv7)
        conv8 = tf.nn.conv2d(conv7, weights['wc8'], strides=[1, 3, 3, 1], padding='SAME')
	conv8 = tf.nn.relu(conv8)
        pool4 = tf.nn.max_pool(conv8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        conv9 = tf.nn.conv2d(pool4, weights['wc9'], strides=[1, 7, 7, 1], padding='SAME')
	conv9 = tf.nn.relu(conv9)
        conv10= tf.nn.conv2d(conv9, weights['wc10'],strides=[1, 3, 3, 1], padding='SAME')
	conv10= tf.nn.relu(conv10)
        pool5 = tf.nn.max_pool(conv10, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	
        # FC + ReLU + Dropout
	fc1 = tf.reshape(pool5, [-1, weights['wfc1'].get_shape().as_list()[0]])
	fc1 = tf.matmul(fc1, weights['wfc1'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, keep_dropout)
	
	# FC + ReLU + Dropout
	fc2 = tf.matmul(fc1, weights['wfc2'])
	fc2 = tf.nn.relu(fc2)
	fc2 = tf.nn.dropout(fc2, keep_dropout)

	# Output FC
	out = tf.add(tf.matmul(fc2, weights['wo']), biases['bo'])
	
	return out

# Construct dataloader
opt_data_train = {
	'data_h5': 'miniplaces_256_train.h5',
	'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
	'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
	'load_size': load_size,
	'fine_size': fine_size,
	'data_mean': data_mean,
	'randomize': True
	}
opt_data_val = {
	'data_h5': 'miniplaces_256_val.h5',
	'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
	'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
	'load_size': load_size,
	'fine_size': fine_size,
	'data_mean': data_mean,
	'randomize': False
	}
def vgg_net_run(dropout, batch_size, learning_rate, training_iters):
	# Resets the weight        
	tf.reset_default_graph();

	loader_train = DataLoaderDisk(**opt_data_train)
        loader_val = DataLoaderDisk(**opt_data_val)
        #loader_train = DataLoaderH5(**opt_data_train)
        #loader_val = DataLoaderH5(**opt_data_val)

        # tf Graph input
        x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
        y = tf.placeholder(tf.int64, None)
        keep_dropout = tf.placeholder(tf.float32)
        train_phase = tf.placeholder(tf.bool)

        # Construct model
        logits = vggnet(x, keep_dropout, train_phase)

	# Define loss and optimizer
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
	train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	# Evaluate model
	accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
	accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

	# define initialization
	init = tf.global_variables_initializer()

	# define saver
	saver = tf.train.Saver()

	# define summary writer
	writer = tf.summary.FileWriter('./vggnet/logs', graph=tf.get_default_graph())

	# Launch the graph
	with tf.Session() as sess:
		# Initialization
		if len(start_from)>1:
			saver.restore(sess, start_from)
		else:
			sess.run(init)
		
		step = 0

		while step < training_iters:
			# Load a batch of training data
			images_batch, labels_batch = loader_train.next_batch(batch_size)
			
			if step % step_display == 0:
				print '[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

				# Calculate batch loss and accuracy on training set
				l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
				print "-Iter " + str(step) + ", Training Loss= " + \
				"{:.6f}".format(l) + ", Accuracy Top1 = " + \
				"{:.4f}".format(acc1) + ", Top5 = " + \
				"{:.4f}".format(acc5)

				# Calculate batch loss and accuracy on validation set
				images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
				l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
				print "-Iter " + str(step) + ", Validation Loss= " + \
				"{:.6f}".format(l) + ", Accuracy Top1 = " + \
				"{:.4f}".format(acc1) + ", Top5 = " + \
				"{:.4f}".format(acc5)
				
			# Run optimization op (backprop)
			sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
				
			step += 1
			
			# Save model
			if step % step_save == 0:
				saver.save(sess, path_save, global_step=step)
				print "Model saved at Iter %d !" %(step)
			
		print "Optimization Finished!"


		# Evaluate on the whole validation set
		print 'Evaluation on the whole validation set...'
		num_batch = loader_val.size()/batch_size
		acc1_total = 0.
		acc5_total = 0.
		loader_val.reset()
		for i in range(num_batch):
			images_batch, labels_batch = loader_val.next_batch(batch_size)    
			acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
			acc1_total += acc1
			acc5_total += acc5
			print "Validation Accuracy Top1 = " + \
			"{:.4f}".format(acc1) + ", Top5 = " + \
			"{:.4f}".format(acc5)

		acc1_total /= num_batch
		acc5_total /= num_batch
		print 'Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total)
		
		# Write the results of the test
		with open('vggnet/results.txt', "a") as results:
		    	results.write("vggnet\ndrop={}, lr={}, iters={}, bs={}, --> accuracy = ({}, {})\n".format(dropout, learning_rate, training_iters, batch_size, acc1_total, acc5_total))   

if __name__ == '__main__':
	vgg_net_run(dropout, batch_size, learning_rate, training_iters)