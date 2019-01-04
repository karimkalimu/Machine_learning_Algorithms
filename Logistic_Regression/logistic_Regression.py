import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
#Hyperparameters
N = 1000 #How many point you want in each class
MaximumIteration = 100000
epsilon = 0.0000000000000001
learningRate = 0.00002


#Data 
mean1 = [12, 12]
mean0 = [5, 5]
cov = [[5, -3], [-3, 10]]
x = np.zeros((2*N, 2))
x_test = np.zeros((2*N, 2))
y = np.zeros(2*N)
y_test = np.zeros(2*N)
x[:N] = np.random.multivariate_normal(mean1, cov, N)#First cluster 
x_test[:N] = np.random.multivariate_normal(mean1, cov, N)#Test data for First cluster 
y[:N] = np.ones(N)
y_test[:N] = np.ones(N)
x[N:] = np.random.multivariate_normal(mean0, cov, N)#Second cluster 
x_test[N:] = np.random.multivariate_normal(mean0, cov, N)#Test data forFirst cluster 

#Make the mean of the data zero
x = x - np.mean(x)
x_test = x_test - np.mean(x_test)

#parameters
w2, w1, w0 = np.random.rand(3)

#sigmoid function
def sigmoid(x):
    temp = - (w2 * x[:,0] + w1 * x[:,1] + w0)
    return 1/(1+np.exp(temp))

#Cost function -L(y^,y) = -( ylog(y^) + (1-y)log(1-y^) ) 
def CostFunction(): 
    temp = -(y*np.log( sigmoid(x) + epsilon ) + (1-y)*np.log((1 - sigmoid(x)) + epsilon)) / len(y)
    return np.sum(temp)

#Gradient Descent
def GD(w_2 , w_1, w_0):
    temp =  sigmoid(x) - y
    w_2 = w_2 - learningRate*np.sum(temp*x[:,0])/len(temp)
    w_1 = w_1 - learningRate*np.sum(temp*x[:,1])/len(temp)
    w_0 = w_0 - learningRate*np.sum(temp*1)/len(temp)
    return w_2, w_1, w_0




#ploting 
plt.figure(1)
x_temp1 = np.asarray(np.where(sigmoid(x_test) > 0.5))
x_temp0 = np.asarray(np.where(sigmoid(x_test) <= 0.5))
plt.scatter(x_test[x_temp1,0], x_test[x_temp1,1], s= 4, color = 'k')
plt.scatter(x_test[x_temp0,0], x_test[x_temp0,1], s= 4, color = 'c')  

#For animation
def plotImageLine():
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(x[x_temp1,0], x[x_temp1,1], s= 4, color = 'g')
    ax.scatter(x[x_temp0,0], x[x_temp0,1], s= 4, color = 'r')
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
    w2, w1, w0 = GD(w2, w1, w0)  
    #each several iteration save a picture
    if i % (MaximumIteration/20) == 0:
        x_temp1 = np.asarray(np.where(sigmoid(x) > 0.5))
        x_temp0 = np.asarray(np.where(sigmoid(x) <= 0.5))
        plotImageLine()


imageio.mimsave('./Logistic_Regression.gif', images, fps=2)

#plt.hist(sigmoid())
CostFunction(),x_temp1.shape,x_temp0.shape