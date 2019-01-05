## import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import tensorflow as tf

#data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.shape)

#Preprocessing
x = x_train.reshape(59999, 28*28)#Just to make it Matrix 2D
N = 20000 #Number of data point
x = x[:N]
print(x.shape)
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

#Project data on Many Direction
def ProjectionM(Direction, Data):
    d = Direction#matris shape = c x d , c number of principle component(PC) you choice
    p = d@Data.T#Projection  point ,matris shape = c x N
    R = d.T@p#Reconstrat the same  points
    return R.T
#For animation
def plotImage():
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(Pro[9],cmap='binary')
    ax.grid()
    ax.set(xlabel='X', ylabel='Y',title='Number of PC = {}'.format(c))

    # Used to keep the limits constant
    ax.set_ylim(0, 28)
    ax.set_xlim(0, 28)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)

def plotPercentage():
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(PercentageHist[:, 0],PercentageHist[:, 1],marker = ">")
    ax.grid()
    ax.set(xlabel='X', ylabel='Y',title='Number of PC = {}'.format(c))

    # Used to keep the limits constant
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 48)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    Percentage.append(image)
    
#For animation
images = []
Percentage = []
PercentageHist = np.zeros((1000, 2))
c = 0#Number of PC
for i in range(20):
    c = (i * 2) + 1
    PercentageHist[i, 0] = c
    PercentageHist[i, 1] = np.sum(s[:c])/np.sum(s)
    Pro = ProjectionM(vh[:c], x).reshape(N, 28,28)
    plotImage()
    plotPercentage()
    
imageio.mimsave('./PCA.gif', images, fps=1)
imageio.mimsave('./PCA_Percentage.gif', Percentage, fps=1)

#plot some image
plt.figure(1)
fig, axes = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        axes[i,j].imshow(x_train[(i+2)*(j+2)])

##plot some image
plt.figure(2)
fig, axes = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        axes[i,j].imshow(Pro[(i+2)*(j+2)])