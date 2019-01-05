import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

#Hyperparameter 
N = 1000#size of data point
maxiteration = 30
epsilon = 0.00000000001

#Data
mu1 = [5,5] 
mu2 = [11,11]
mu3 = [13,20]
mu4 = [20,30]
cov1 = [[1, 1/2], [1/2, 2]]#Covariance for first cluster
cov2 = [[2, 1/2], [ 1/2, 1]]
cov3 = [[3.5, 4/7], [ 4/7, 1.5]]
cov4 = [[2, 1/3], [ 1/3, 1.5]]
x = np.zeros((4*N,2))
x[:N] = np.random.multivariate_normal(mu1, cov1, N)
x[N:2*N] = np.random.multivariate_normal(mu2, cov2, N)
x[2*N:3*N] = np.random.multivariate_normal(mu3, cov3, N)
x[3*N:] = np.random.multivariate_normal(mu4, cov4, N)

#Learning parameters
mean1 = x[int(np.random.rand(1)*4*N)]
mean2 = x[int(np.random.rand(1)*4*N)]
mean3 = x[int(np.random.rand(1)*4*N)]
mean4 = x[int(np.random.rand(1)*4*N)]

#assign two random point as mean
c1 = np.full(N, False, dtype=bool)
c2 = np.full(N, False, dtype=bool)
c3 = np.full(N, False, dtype=bool)
c4 = np.full(N, False, dtype=bool)
#value indicate closer to each cluster

def Reset():
    global mean1
    global mean2
    global mean3
    global mean4
    mean1 = x[int(np.random.rand(1)*4*N)]
    mean2 = x[int(np.random.rand(1)*4*N)]
    mean3 = x[int(np.random.rand(1)*4*N)]
    mean4 = x[int(np.random.rand(1)*4*N)]


#L2
def Euclidean(x_t, mean):
    x_t = x_t - mean + np.random.rand(1)*epsilon
    #this randomness for not have two distances exactly equals
    return np.sqrt(np.sum(x_t**2,axis = 1))

#L1
def Manhattan(x_t, mean):
    x_t = abs(x_t - mean) + np.random.rand(1)*epsilon
    return np.sum(x_t,axis = 1)


#Assign Points to each class/cluster using Euclidean
def AssigningE():
    dist1 = Euclidean(x, mean1)
    dist2 = Euclidean(x, mean2)
    dist3 = Euclidean(x, mean3)
    dist4 = Euclidean(x, mean4)
    c_1 = ( dist1 < dist2) & ( dist1 < dist3) & (dist1 < dist4) 
    #when dist1 > dist2 means that the point closer to mean2
    c_2 = ( dist2 < dist1 ) & ( dist2 < dist3 ) & ( dist2 < dist4)
    c_3 = ( dist3 < dist2 ) & ( dist3 < dist1 ) & ( dist3 < dist4)
    c_4 = ( dist4 < dist2 ) & ( dist4 < dist3 ) & ( dist4 < dist1)
    return c_1, c_2, c_3, c_4

#Assign Points to each class/cluster using Manhattan
def AssigningM():
    dist1 = Manhattan(x, mean1)
    dist2 = Manhattan(x, mean2)
    dist3 = Manhattan(x, mean3)
    dist4 = Manhattan(x, mean4)
    c_1 = ( dist1 < dist2) & ( dist1 < dist3) & (dist1 < dist4)
    c_2 = ( dist2 < dist1 ) & ( dist2 < dist3 ) & ( dist2 < dist4)
    c_3 = ( dist3 < dist2 ) & ( dist3 < dist1 ) & ( dist3 < dist4)
    c_4 = ( dist4 < dist2 ) & ( dist4 < dist3 ) & ( dist4 < dist1)
    return c_1, c_2, c_3, c_4

#Recalculate the mean of each cluster
def NewMean():
    mean_1 = np.mean(x[c1])
    mean_2 = np.mean(x[c2]) 
    mean_3 = np.mean(x[c3]) 
    mean_4 = np.mean(x[c4]) 
    return mean_1, mean_2, mean_3, mean_4

#For animation
def plotImageLine():
    fig, ax = plt.subplots(figsize=(10,5))
    plt.scatter(x[c1,0],x[c1,1], s = 8, color = "r" )
    plt.scatter(x[c2,0],x[c2,1], s = 8, color = "b" )
    plt.scatter(x[c3,0],x[c3,1], s = 8, color = "g" )
    plt.scatter(x[c4,0],x[c4,1], s = 8, color = "y" )
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

def plot(t):
    plt.figure(t)
    plt.scatter(x[:,0],x[:,1], s = 8, color = "k" )
    plt.scatter(x[c1,0],x[c1,1], s = 8, color = "r" )
    plt.scatter(x[c2,0],x[c2,1], s = 8, color = "b" )
    plt.scatter(x[c3,0],x[c3,1], s = 8, color = "g" )
    plt.scatter(x[c4,0],x[c4,1], s = 8, color = "y" )

images = []
for i in range(maxiteration):
    c1, c2, c3, c4 = AssigningE()
    mean1, mean2, mean3, mean4 = NewMean()
    #plotImageLine()
    
#imageio.mimsave('./K-means_Euclidean_Distance.gif', images, fps=1)
plot(1)
Reset()
images = []
for i in range(maxiteration):
    c1, c2, c3, c4 = AssigningM()
    mean1, mean2, mean3, mean4 = NewMean()
    #plotImageLine()

#imageio.mimsave('./K-means_Manhattan_Distance.gif', images, fps=1)
plot(2)
plt.figure(3)
plt.scatter(x[:N,0],x[:N,1], s = 8, color = "r" )
plt.scatter(x[N:2*N,0],x[N:2*N,1], s = 8, color = "b" )
plt.scatter(x[2*N:3*N,0],x[2*N:3*N,1], s = 8, color = "g" )
plt.scatter(x[3*N:,0],x[3*N:,1], s = 8, color = "y" )
