import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 1
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.000025
dropout = 0.8
training_iters = 10000
step_display = 5000
step_save = 10000
path_save = './submitnet/sessions'
start_from = './alexnet/sessions/model.ckpt-30000' #str(step_save)

def batch_norm_layer(x, train_phase, scope_bn):
	return batch_norm(x, decay=0.9, center=True, scale=True,
	updates_collections=None,
	is_training=train_phase,
	reuse=None,
	trainable=True,
	scope=scope_bn)
	
def alexnet(x, keep_dropout, train_phase):
	weights = {
	    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
	    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
	    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
	    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
	    'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

	    'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
	    'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
	    'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
	}

	biases = {
	    'bo': tf.Variable(tf.ones(100))
	}

	# Conv + ReLU + Pool, 224->55->27
	conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
	conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
	conv1 = tf.nn.relu(conv1)
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

	# Conv + ReLU  + Pool, 27-> 13
	conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
	conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
	conv2 = tf.nn.relu(conv2)
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

	# Conv + ReLU, 13-> 13
	conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
	conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
	conv3 = tf.nn.relu(conv3)

	# Conv + ReLU, 13-> 13
	conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
	conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
	conv4 = tf.nn.relu(conv4)

	# Conv + ReLU + Pool, 13->6
	conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
	conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
	conv5 = tf.nn.relu(conv5)
	pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

	# FC + ReLU + Dropout
	fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
	fc6 = tf.matmul(fc6, weights['wf6'])
	fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
	fc6 = tf.nn.relu(fc6)
	fc6 = tf.nn.dropout(fc6, keep_dropout)
	
	# FC + ReLU + Dropout
	fc7 = tf.matmul(fc6, weights['wf7'])
	fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
	fc7 = tf.nn.relu(fc7)
	fc7 = tf.nn.dropout(fc7, keep_dropout)

	# Output FC
	out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
	
	return out

# Construct dataloader
opt_data_test = {
	'data_root': '../../data/images/',  # MODIFY PATH ACCORDINGLY
	'load_size': load_size,
	'fine_size': fine_size,
	'data_mean': data_mean
	}

def alex_net_run(dropout, batch_size, learning_rate, testing_iters):
    loader_test = TestDataLoaderDisk(**opt_data_test)
                
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
    keep_dropout = tf.placeholder(tf.float32)
    train_phase = tf.placeholder(tf.bool)

    # Construct model
    logits = alexnet(x, keep_dropout, train_phase)

    # Evaluate model
    predictions = tf.nn.top_k(logits, k=5)
    
    # define initialization
    init = tf.global_variables_initializer()

	# define saver
    saver = tf.train.Saver()

	# Launch the graph
    with tf.Session() as sess:
    	# Initialization
    	saver.restore(sess, start_from)
    	step = 0

    	with open('submitnet/test.txt', "w") as results:	
            while step < testing_iters:
                # Load a batch of training data
                images_batch = loader_test.next_batch(batch_size)
				
                if step % step_display == 0:
				    print '[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
				    print 'iteration: %d' %(step)
						
	# Get prediction
                values, inds = sess.run(predictions, feed_dict={x: images_batch, keep_dropout: 1., train_phase: False})
                inds = inds[0].tolist()
				# Write the results of the test
                results.write('test/' + '0'*(8-len(str(step+1))) + str(step+1) + '.jpg ' + ' '.join(str(ind) for ind in inds) + '\n')
                step += 1

            print "Prediction Generated!"
if __name__ == '__main__':
	alex_net_run(dropout, batch_size, learning_rate, training_iters)
