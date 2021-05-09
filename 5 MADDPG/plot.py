import numpy as np
import matplotlib.pyplot as plt

PATH = './history.npy'
x = np.load(PATH)

# use filter to make the line more smooth
filter_num = 20
x[:,0] = np.convolve(x[:,0], np.array([1]*filter_num)/filter_num, 'same')
x[:,1] = np.convolve(x[:,1], np.array([1]*filter_num)/filter_num, 'same')
plt.plot(x[:,0])
plt.plot(x[:,1])
plt.show()

print('reward at start:', np.average(x[0:10,0]))
print('reward at end:', np.average(x[-1-50:-1,0]))