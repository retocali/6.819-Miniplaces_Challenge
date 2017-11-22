import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
graph_fontsize = 18
matplotlib.rcParams['font.sans-serif'] = "Computer Modern"
matplotlib.rcParams['font.family'] = "sans-serif"

# Import Data
resnetdirectory = './resnet/plots/'
resnetlossfile = 'resnet_losses.npy'
resnetaccuracyfile = 'resnet_accuracies.npy'

alexnetdirectory = './alexnet/plots/'
alexnetlossfile = 'alexnet_losses.npy'
alexnetaccuracyfile = 'alexnet_accuracies.npy'

resnet_losses = np.genfromtxt(resnetdirectory + resnetlossfile)
resnet_accuracies = 100*np.genfromtxt(resnetdirectory + resnetaccuracyfile)
alexnet_losses = np.genfromtxt(alexnetdirectory + alexnetlossfile)
alexnet_accuracies = 100*np.genfromtxt(alexnetdirectory + alexnetaccuracyfile)

# Manipulate Data
data_divide = 5
n = '-'
iterations = np.arange(0,10000,data_divide)

# Plot
plt.figure(1)
plt.plot(iterations, resnet_losses[::data_divide], 'b'+n, alpha=0.7, label='ResNet Losses')
plt.plot(iterations, alexnet_losses[::data_divide], 'r'+n, alpha=0.7, label='AlexNet Losses')
plt.title('AlexNet vs. ResNet Losses', fontsize=graph_fontsize)
plt.axis([-10, 10000, 0, 6])
plt.legend(loc='upper right')
plt.ylabel('Losses', fontsize=graph_fontsize)
plt.xlabel('Iteration', fontsize=graph_fontsize)
plt.savefig('./plots/losscomparison.png', bbox_inches='tight')


plt.figure(2)
plt.plot(iterations, resnet_accuracies[::data_divide], 'b'+n, alpha=0.7, label='ResNet Top 5 Accuracies')
plt.plot(iterations, alexnet_accuracies[::data_divide], 'r'+n, alpha=0.7, label='AlexNet Top 5 Accuracies')
plt.title('AlexNet vs. ResNet Accuracies', fontsize=graph_fontsize)
plt.legend(loc='lower right')
plt.ylabel('Accuracy [%]', fontsize=graph_fontsize)
plt.xlabel('Iteration', fontsize=graph_fontsize)
plt.savefig('./plots/accuracycomparison.png', bbox_inches='tight')
