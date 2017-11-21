import os, datetime
import numpy as np
import matplotlib.pyplot as plt
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
training_iters = 10000
step_display = 500
step_save = 10000
path_save = './resnet/sessions/model.ckpt'
start_from = ''

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def res_layer(x, weights, keys, train_phase, scopes, downsample=False, out_channels=0):
    
    if downsample:        
        
        # Two 3x3 Conv Layers
        convx_1 = tf.nn.conv2d(x, weights[keys[0]], strides=[1, 1, 1, 1], padding='SAME')
        convx_1 = batch_norm_layer(convx_1, train_phase, scopes[0])
        convx_1 = tf.nn.relu(convx_1)
        convx_2 = tf.nn.conv2d(convx_1, weights[keys[1]], strides=[1, 2, 2, 1], padding='SAME')
        convx_2 = batch_norm_layer(convx_2, train_phase, scopes[1])
        convx_2 = tf.nn.relu(convx_2)
        
        # Downsample input + pad input + add to output
        in_channels = x.get_shape().as_list()[3]
        adding_input = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        adding_input = tf.pad(adding_input, [[0,0], [0,0], [0,0], [0, out_channels - in_channels]])
        convx_2 = tf.add(convx_2, adding_input)
        return convx_2

    # Two 3x3 Conv Layers, no downsampling
    convx_1 = tf.nn.conv2d(x, weights[keys[0]], strides=[1, 1, 1, 1], padding='SAME')
    convx_1 = batch_norm_layer(convx_1, train_phase, scopes[0])
    convx_1 = tf.nn.relu(convx_1)
    convx_2 = tf.nn.conv2d(convx_1, weights[keys[1]], strides=[1, 1, 1, 1], padding='SAME')
    convx_2 = batch_norm_layer(convx_2, train_phase, scopes[1])
    convx_2 = tf.add(tf.nn.relu(convx_2), x)
    return convx_2


def resnet(x, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([7, 7, 3, 64], stddev=np.sqrt(2./(7*7*3)))),
        'wc2_1': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2./(3*3*64)))),
        'wc2_2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2./(3*3*64)))),
        'wc3_1': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2./(3*3*64)))),
        'wc3_2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=np.sqrt(2./(3*3*64)))),
        'wc4_1': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2./(3*3*128)))),
        'wc4_2': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2./(3*3*128)))),
        'wc5_1': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2./(3*3*128)))),
        'wc5_2': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=np.sqrt(2./(3*3*128)))),
        'wc6_1': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        'wc6_2': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        'wc7_1': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        'wc7_2': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=np.sqrt(2./(3*3*256)))),
        'wc8_1': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc8_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc9_1': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc9_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wo': tf.Variable(tf.random_normal([7*7*512, 100], stddev=np.sqrt(2./(3*3*512)))),
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Conv + Batch + ReLU + Pool, 224->112->56
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Res + Res, 56->56->28
    res2 = res_layer(pool1, weights, ['wc2_1', 'wc2_2'], train_phase, ['bn2_1', 'bn2_2'])
    res3 = res_layer(res2, weights, ['wc3_1', 'wc3_2'], train_phase, ['bn3_1', 'bn3_2'], True, 128)
    
    # Res + Res, 28->28->14
    res4 = res_layer(res3, weights, ['wc4_1', 'wc4_2'], train_phase, ['bn4_1', 'bn4_2'])
    res5 = res_layer(res4, weights, ['wc5_1', 'wc5_2'], train_phase, ['bn5_1', 'bn5_2'], True, 256)

    # Res + Res, 14->14->7
    res6 = res_layer(res5, weights, ['wc6_1', 'wc6_2'], train_phase, ['bn6_1', 'bn6_2'])
    res7 = res_layer(res6, weights, ['wc7_1', 'wc7_2'], train_phase, ['bn7_1', 'bn7_2'], True, 512)

    # Res + Res, 7->1
    res8 = res_layer(res7, weights, ['wc8_1', 'wc8_2'], train_phase, ['bn8_1', 'bn8_2'])
    res9 = res_layer(res8, weights, ['wc9_1', 'wc9_2'], train_phase, ['bn9_1', 'bn9_2'])

    # Avg Pool + Output FC 7->7->1
    pool10 = tf.nn.avg_pool(res9, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME')
    fc10 = tf.reshape(pool10, [-1, weights['wo'].get_shape().as_list()[0]])
    out = tf.matmul(fc10, weights['wo'])
    
    return out

# Construct dataloader
opt_data_train = {
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

def res_net_run(batch_size, learning_rate, training_iters):
    # Resets the weight        
    tf.reset_default_graph();

    loader_train = DataLoaderDisk(**opt_data_train)
    loader_val = DataLoaderDisk(**opt_data_val)
    #loader_train = DataLoaderH5(**opt_data_train)
    #loader_val = DataLoaderH5(**opt_data_val)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
    y = tf.placeholder(tf.int64, None)
    train_phase = tf.placeholder(tf.bool)

    # Construct model
    logits = resnet(x, train_phase)

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
    writer = tf.summary.FileWriter('./resnet/logs', graph=tf.get_default_graph())

    # Track loss & accuracy for plotting
    batch_losses = np.zeros((1, training_iters), dtype='f')
    batch_accuracies = np.zeros((1, training_iters), dtype='f')    

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
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, train_phase: False}) 
                print "-Iter " + str(step) + ", Training Loss= " + \
                "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                "{:.4f}".format(acc1) + ", Top5 = " + \
                "{:.4f}".format(acc5)

                # Calculate batch loss and accuracy on validation set
                images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, train_phase: False}) 
                print "-Iter " + str(step) + ", Validation Loss= " + \
                "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                "{:.4f}".format(acc1) + ", Top5 = " + \
                "{:.4f}".format(acc5)

            batch_losses[0,step] = l
            batch_accuracies[0, step] = acc5

            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, train_phase: True})
            
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
            acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, train_phase: False})
            acc1_total += acc1
            acc5_total += acc5
            print "Validation Accuracy Top1 = " + \
            "{:.4f}".format(acc1) + ", Top5 = " + \
            "{:.4f}".format(acc5)

        acc1_total /= num_batch
        acc5_total /= num_batch
        print 'Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total)

        # Write the results of the test
        with open('resnet/results.txt', "a") as results:
            results.write("resnet\nlr={}, iters={}, bs={}, --> accuracy = ({}, {})\n".format(learning_rate, training_iters, batch_size, acc1_total, acc5_total))

        print batch_accuracies
        # Create and store plot
        plt.figure(1)
        plt.subplot(211)
        plt.plot(batch_accuracies[0], 'b-')
        plt.title('Batch Accuracy on Validation Set')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        
        print batch_losses
        plt.subplot(212)
        plt.plot(batch_losses[0], 'b-')
        plt.title('Batch Loss on Validation Set')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.savefig('./resnet/plots/resnet.png')

        np.savetxt('./resnet/plots/resnet_accuracies.npy', batch_accuracies)
        np.savetxt('./resnet/plots/resnet_losses.npy', batch_losses)

if __name__ == '__main__':
    res_net_run(batch_size, learning_rate, training_iters)
