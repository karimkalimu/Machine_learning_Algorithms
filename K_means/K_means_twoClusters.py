import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

#Hyperparameter 
N = 1000#size of data point
epsilon = 0.000001
maxiteration = 50

#Data
mu1 = [13,7] 
mu2 = [28,8]
cov1 = [[5, 1/2], [1/2, 2]]#Covariance for first cluster
cov2 = [[2, 1/4], [ 1/4, 7]]
x = np.zeros((2*N,2))
x[:N] = np.random.multivariate_normal(mu1, cov1, N)
x[N:] = np.random.multivariate_normal(mu2, cov2, N)

#Learning parameters
mean1, mean2 = x[int(np.random.rand(1)*N)],x[int(np.random.rand(1)*N)]
#assign two random point as mean
c1,c2 = np.full(N, True, dtype=bool),np.full(N, True, dtype=bool)
#value indicate closer to each cluster

#Calculate distense between each point to each mean
def Euclidean(x_t, mean):
    x_t = (x_t - mean)**2
    return np.sqrt(np.sum(x_t**2,axis = 1))

#Assign Points to each class/cluster
def Assigning():
    dist1 = Euclidean(x, mean1)
    dist2 = Euclidean(x, mean2)
    c_1 = (dist1 < dist2)
    #when dist1 > dist2 means that the point closer to mean2
    c_2 = (dist1 >= dist2)
    return c_1,c_2

#Recalculate the mean of each cluster
def NewMean():
    mean_1 = np.mean(x[c1])
    mean_2 = np.mean(x[c2]) 
    return mean_1, mean_2

#For animation
def plotImageLine():
    fig, ax = plt.subplots(figsize=(10,5))
    plt.scatter(x[c1,0],x[c1,1], s = 8, color = "c" )
    plt.scatter(x[c2,0],x[c2,1], s = 8, color = "k" )
    ax.grid()
    ax.set(xlabel='X1', ylabel='X2')

    # Used to keep the limits constant
    ax.set_ylim(0, 40)
    ax.set_xlim(0, 40)

    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)


images = []
Difference = mean1#if almost converge stop
for i in range(maxiteration):
    c1,c2 = Assigning()
    mean_t = mean1
    mean1, mean2 = NewMean()
    Difference = abs(np.sum(mean1 - mean_t))
    #plotImageLine()
    if Difference < epsilon :
        break  

#imageio.mimsave('./K-means_twoClusters.gif', images, fps=1)
plt.figure(2)
plt.scatter(x[:N,0],x[:N,1], s = 8, color = "r" )
plt.scatter(x[N:,0],x[N:,1], s = 8, color = "b" )
plt.figure(3)
plt.scatter(x[c1,0],x[c1,1], s = 8, color = "c" )
plt.scatter(x[c2,0],x[c2,1], s = 8, color = "k" )
;