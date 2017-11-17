import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0001
dropout = 0.7
training_iters = 2000
step_display = 25
step_save = 500
path_save = './smallnet/sessions/'
start_from = ''

def batch_norm_layer(x, train_phase, scope_bn):
	return batch_norm(x, decay=0.9, center=True, scale=True,
	updates_collections=None,
	is_training=train_phase,
	reuse=None,
	trainable=True,
	scope=scope_bn)

def smallnet(x, keep_dropout):
	weights = {
		'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=np.sqrt(2./3*3*3))),
		'wc2': tf.Variable(tf.random_normal([3, 3, 32, 1], stddev=np.sqrt(2./3*3*32))),
		'wc3': tf.Variable(tf.random_normal([1, 1, 32, 64], stddev=np.sqrt(2./1*1*32))),
		'wf4': tf.Variable(tf.random_normal([28*28*64, 1024], stddev=np.sqrt(2./28*28*64))),
		'wo': tf.Variable(tf.random_normal([1024, 100], stddev=np.sqrt(2./100)))
	}

	biases = {
		'bc1': tf.Variable(tf.zeros(32)),
		'bc2': tf.Variable(tf.zeros(32)),
		'bc3': tf.Variable(tf.zeros(64)),
		'bo': tf.Variable(tf.zeros(100))
	}

	# Conv + BN + Relu + Pool, 224->112->56
	conv1 = tf.nn.conv2d(x, weights['wc1'], strides = [1, 2, 2, 1], padding='SAME')
	conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
	conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
	pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides = [1, 2, 2, 1], padding='SAME')
	
	# DWConv + BN + Relu, 56->56
	with tf.device('/device:GPU:0'):
		conv2= tf.nn.depthwise_conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
	conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
	conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))

	# PWConv + BN + Relu + Pool, 56->56->28	
	conv3 = tf.nn.conv2d(conv2, weights['wc3'], strides=[1,1,1,1], padding='SAME')
	conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
	conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))
	pool3 = tf.nn.max_pool(conv3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

	# FC + ReLU + Dropout 
	fc4 = tf.reshape(pool3, [-1, weights['wf4'].get_shape().as_list()[0]])
	fc4 = tf.matmul(fc4, weights['wf4'])
	fc4 = batch_norm_layer(fc4, train_phase, 'bn4')
	fc4 = tf.nn.relu(fc4)
	fc4 = tf.nn.dropout(fc4, keep_dropout)

	# Output FC
	out = tf.reshape(fc4, [-1, weights['wo'].get_shape().as_list()[0]])
	out = tf.add(tf.matmul(out, weights['wo']), biases['bo'])

	return out

opt_data_train = {
# Construct dataloader
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

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
logits = smallnet(x, keep_dropout)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
writer = tf.summary.FileWriter('./smallnet/logs/', graph=tf.get_default_graph())

# Launch the graph allowing for the gpu to be found
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
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
            "{:.4f}".format(l) + ", Accuracy Top1 = " + \
            "{:.2f}".format(acc1) + ", Top5 = " + \
            "{:.2f}".format(acc5)

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            print "-Iter " + str(step) + ", Validation Loss= " + \
            "{:.4f}".format(l) + ", Accuracy Top1 = " + \
            "{:.2f}".format(acc1) + ", Top5 = " + \
            "{:.2f}".format(acc5)
        
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
            "{:.2f}".format(acc1) + ", Top5 = " + \
            "{:.2f}".format(acc5)

    acc1_total /= num_batch
    acc5_total /= num_batch
    print 'Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total)

    with open('smallnet/results.txt', "a") as results:
        results.write("smallnet\ndrop={}, lr={}, iters={}, bs={}, --> accuracy = ({}, {})\n".format(dropout, learning_rate, training_iters, batch_size, acc1_total, acc5_total))
	
