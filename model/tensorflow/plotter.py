import numpy as np
import matplotlib.pyplot as plt

directory = './resnet/plots/'
filename = 'resnet_losses.npy'

batch_losses = np.genfromtxt(directory + filename)

print batch_losses

plt.plot(batch_losses)
plt.show()
