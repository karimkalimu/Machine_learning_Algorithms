import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

#Hyperparameters
N = 5000#Size of data

#Data
mean = [16,16]
covariance = [[6,4],[4,6]]
x = np.random.multivariate_normal(mean, covariance, N)

#Preprocessing
x_h = x - np.mean(x)

#Singular Value Decomposition
u, s, vh = np.linalg.svd(x_h, full_matrices=True)

#Project data on one Direction
def Projection(Direction, Data):
    d = Direction#matris shape = d
    d = d[:,np.newaxis]#matris shape = d, 1
    p = d.T@Data.T#Projection  point ,matris shape = N
    p = p[np.newaxis,:]#matrix shape = 1 x N
    R = d@p#Reconstrat the same  points
    R = np.squeeze(R).T
    return R

d1 = Projection(vh[0], x_h) #First principle component PC1
d2 = Projection(vh[1], x_h) #PC2

plt.scatter(x[:,0],x[:,1])#Data
plt.scatter(x_h[:,0],x_h[:,1], color = 'r')#Data - mean
plt.plot(d1[:,0], d1[:,1], "k")
plt.plot(d2[:,0], d2[:,1], "k")
plt.figure(2)
plt.hist(s/np.sum(s))