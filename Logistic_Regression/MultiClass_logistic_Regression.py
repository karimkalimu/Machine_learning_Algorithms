import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

#Hyperparameters
N = 1000 #How many point you want in each class
MaximumIteration = 100000
epsilon = 0.0000000000000001
learningRate = 0.0003

#Data 
mean2 = [17, 30]
mean1 = [12, 12]
mean0 = [3, 2]
cov2 = [[5, -3], [-3, 10]]
cov1 = [[5, -3], [-3, 10]]
cov = [[5, -3], [-3, 10]]
x = np.zeros((3*N, 2))
y = np.zeros((3*N, 2))
#######
x[:N] = np.random.multivariate_normal(mean2, cov, N)#First cluster 
x[N:2*N] = np.random.multivariate_normal(mean1, cov, N)#Second cluster 
x[2*N:] = np.random.multivariate_normal(mean0, cov, N)#Third cluster 
y[:N, 0] = np.ones(N)
y[2*N:, 1] = np.ones(N)

#Make the mean of the data zero
x = x - np.mean(x)

#parameters
#weights for two of the classes
w2_2, w1_2, w0_2 = np.random.rand(3)
w2_1, w1_1, w0_1 = np.random.rand(3)

#sigmoid function
def sigmoid(x, w2, w1, w0):
    temp = - (w2 * x[:,0] + w1 * x[:,1] + w0)
    return 1/(1+np.exp(temp))

#Cost function -L(y^,y) = -( ylog(y^) + (1-y)log(1-y^) ) 
def CostFunction(x_t, y_t, w2, w1, w0): 
    temp = -(y_t*np.log( sigmoid(x_t,  w2, w1, w0) + epsilon ) + (1-y_t)*np.log((1 - sigmoid(x_t,  w2, w1, w0)) + epsilon)) / len(y_t)
    return np.sum(temp)

#Gradient Descent
def GD(y_t, x_t, w_2 , w_1, w_0):
    temp =  sigmoid(x_t, w_2 , w_1, w_0) - y_t
    w_2 = w_2 - learningRate*np.sum(temp*x_t[:,0])/len(temp)
    w_1 = w_1 - learningRate*np.sum(temp*x_t[:,1])/len(temp)
    w_0 = w_0 - learningRate*np.sum(temp*1)/len(temp)
    return w_2, w_1, w_0

#For animation
def plotImageLine():
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(x[x_temp1,0], x[x_temp1,1], s= 4, color = 'g')
    ax.scatter(x[x_temp0,0], x[x_temp0,1], s= 4, color = 'r')
    ax.grid()
    ax.set(xlabel='X1', ylabel='X2',title='Learning rate = {}'.format(learningRate))

    # Used to keep the limits constant
    ax.set_ylim(-15, 15)
    ax.set_xlim(-15, 15)

    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)

#ploting 
plt.figure(1)
#x_temp1 = np.asarray(np.where(sigmoid(x) > 0.5))
#x_temp0 = np.asarray(np.where(sigmoid(x) < 0.5))
plt.scatter(x[:N,0], x[:N,1], s= 4, color = 'r')
plt.scatter(x[N:2*N,0], x[N:2*N,1], s= 4, color = 'k')
plt.scatter(x[2*N:,0], x[2*N:,1], s= 4, color = 'c')
plt.figure(2)
plt.subplot(221)
x_temp2 = np.asarray(np.where(sigmoid(x, w2_2, w1_2, w0_2) > 0.5))
x_temp1 = np.asarray(np.where(sigmoid(x, w2_2, w1_2, w0_2) <= 0.5))
plt.scatter(x[x_temp2,0], x[x_temp2,1], s= 4, color = 'g')
plt.scatter(x[x_temp1,0], x[x_temp1,1], s= 4, color = 'r')
plt.subplot(222)
x_temp2 = np.asarray(np.where(sigmoid(x, w2_1, w1_1, w0_1) > 0.5))
x_temp1 = np.asarray(np.where(sigmoid(x, w2_1, w1_1, w0_1) <= 0.5))
plt.scatter(x[x_temp2,0], x[x_temp2,1], s= 4, color = 'g')
plt.scatter(x[x_temp1,0], x[x_temp1,1], s= 4, color = 'r')
#For animation
def plotImageLine():
    fig, ax = plt.subplots(figsize=(10,5))
    plt.subplot(331)
    x_temp21 = np.asarray(np.where(sigmoid(x, w2_2, w1_2, w0_2) > 0.5))
    x_temp11 = np.asarray(np.where(sigmoid(x, w2_2, w1_2, w0_2) <= 0.5))
    plt.scatter(x[x_temp21,0], x[x_temp21,1], s= 3, color = 'g')
    plt.scatter(x[x_temp11,0], x[x_temp11,1], s= 3, color = 'r')
    plt.subplot(332)
    x_temp22 = np.asarray(np.where(sigmoid(x, w2_1, w1_1, w0_1) > 0.5))
    x_temp12 = np.asarray(np.where(sigmoid(x, w2_1, w1_1, w0_1) <= 0.5))
    plt.scatter(x[x_temp22,0], x[x_temp22,1], s= 3, color = 'b')
    plt.scatter(x[x_temp12,0], x[x_temp12,1], s= 3, color = 'k')
    plt.subplot(333)
    plt.scatter(x[x_temp11,0], x[x_temp11,1], s= 3, color = 'c')
    plt.scatter(x[x_temp12,0], x[x_temp12,1], s= 3, color = 'c')
    plt.scatter(x[x_temp21,0], x[x_temp21,1], s= 3, color = 'y')
    plt.scatter(x[x_temp22,0], x[x_temp22,1], s= 3, color = 'y')
    ax.grid()
    ax.set(xlabel='X', ylabel='Y',title='Learning rate = {}'.format(learningRate))

    # Used to keep the limits constant
    ax.set_ylim(-15, 15)
    ax.set_xlim(-15, 15)

    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)

images = []
for i in range(MaximumIteration):
    w2_2, w1_2, w0_2 = GD(y[:,0], x, w2_2, w1_2, w0_2)  
    w2_1, w1_1, w0_1 = GD(y[:,1], x, w2_1, w1_1, w0_1) 
    if i % (MaximumIteration/20) == 0:
        plotImageLine()

imageio.mimsave('./Multiclass_Logistic_Regression.gif', images, fps=2)

plt.scatter(x[:N,0], x[:N,1], s= 4, color = 'b')
plt.scatter(x[N:,0], x[N:,1], s= 4, color = 'r')
plt.figure(2)
plt.subplot(221)
x_temp2 = np.asarray(np.where(sigmoid(x, w2_2, w1_2, w0_2) > 0.5))
x_temp1 = np.asarray(np.where(sigmoid(x, w2_2, w1_2, w0_2) <= 0.5))
plt.scatter(x[x_temp2,0], x[x_temp2,1], s= 4, color = 'g')
plt.scatter(x[x_temp1,0], x[x_temp1,1], s= 4, color = 'r')
plt.subplot(222)
x_temp2 = np.asarray(np.where(sigmoid(x, w2_1, w1_1, w0_1) > 0.5))
x_temp1 = np.asarray(np.where(sigmoid(x, w2_1, w1_1, w0_1) <= 0.5))
plt.scatter(x[x_temp2,0], x[x_temp2,1], s= 4, color = 'b')
plt.scatter(x[x_temp1,0], x[x_temp1,1], s= 4, color = 'k')
