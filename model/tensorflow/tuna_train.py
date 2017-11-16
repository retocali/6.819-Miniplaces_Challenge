import alexnet_bn_train
from numpy import arange

dropout_range = [0.2, 0.7, 0.1]
learning_rate_range = [0.0001, 0.001, 0.0002]
batch_size_range = [16, 64, 16]

noise_repeating_rate = 2
training_iters = 500

constant_dropout = alexnet_bn_train.dropout
constant_learning_rate = alexnet_bn_train.learning_rate
constant_batch_size = alexnet_bn_train.batch_size

print("-"*16)
print("\nTESTING DROPOUT\n");
print("-"*16)
for dropout in arange(dropout_range[0], dropout_range[1], dropout_range[2]):
	for _ in range(noise_repeating_rate):
		alexnet_bn_train.alex_net_run(dropout, constant_batch_size, constant_learning_rate, training_iters)


print("-"*19)
print("\nTESTING BATCH_SIZE\n");
print("-"*19)
for batch_size in arange(batch_size_range[0], batch_size_range[1], batch_size_range[2]):
	for _ in range(noise_repeating_rate):
		alexnet_bn_train.alex_net_run(constant_dropout, batch_size, constant_learning_rate, training_iters)


print("-"*25)
print("\nTESTING LEARNING_RATE\n");
print("-"*25)
for learning_rate in arange(learning_rate_range[0], learning_rate_range[1], learning_rate_range[2]):
	for _ in range(noise_repeating_rate):
		alexnet_bn_train.alex_net_run(constant_dropout, constant_batch_size, learning_rate, training_iters)
	
	
