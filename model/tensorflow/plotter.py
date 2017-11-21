import numpy as np
import matplotlib.pyplot as plt

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

plt.figure(1)
plt.plot(resnet_losses, 'b-', label='AlexNet Losses')
plt.plot(alexnet_losses, 'r-', label='ResNet Losses')
plt.title('AlexNet vs. ResNet Losses')
plt.axis([-10, 10000, 0, 6])
plt.legend(loc='upper right')
plt.ylabel('Losses')
plt.xlabel('Iteration')
plt.savefig('./plots/losscomparison.png', bbox_inches='tight')


plt.figure(2)
plt.plot(resnet_accuracies, 'b-', label='AlexNet Top 5 Accuracies')
plt.plot(alexnet_accuracies, 'r-', label='ResNet Top 5 Accuracies')
plt.title('AlexNet vs. ResNet Accuracies')
plt.legend(loc='lower right')
plt.ylabel('Accuracy [%]')
plt.xlabel('Iteration')
plt.savefig('./plots/accuracycomparison.png', bbox_inches='tight')
