#quadratic regession
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

#Hyperparameter
MaxIteration = 145
learning_rate = 0.0000000000000000000001 #Much more sensitive here than in linear regression

#Data
x = np.linspace(100, 1000 , 1000)

#The line we want to pridect
#we add x^2 to make y depend on x in quadratic way
y = 2*x** + 5*x + 15 #so we want w1 to equal 10.5 and w0 = 15

#Residuals 
r_y = np.random.normal(0,5,1000)
r_x = np.random.normal(0,3,1000)

#Now we add the noise to the line to make it a liitle bit like normal data
y_r = y  + r_y
x_r = x + r_x

#Initialize the weights, w1 slope , w0 bias 
#we add w2 
w2, w1, w0 = np.random.rand(3)*3

#Our First predict
y_p = w2*x** + w1 * x_r + w0

#Prediction 
def Predict():
    y_p = w2*x** + w1*x_r + w0
    return y_p

#Cost Function , we will use Mean Squeared Error (MSE)
L = np.sum((y - y_p)**2)/len(y)
def calculate_Loss():
    y_p = Predict()
    L = np.sum((y - y_p)**2)/len(y)
    return L

#Create List of MSE you will get in each iteration
LL = []
LL.append(L)

#Gradient Descent 
def Gradient_descent(w_2,w_1,w_0):
    y_p = Predict()
    #and this line is the last line we change for make the linear regression quadratic regression 
    w_2 = w_2 + 2*learning_rate*np.sum((y - y_p)*(x_r*x_r))/len(y)
    w_1 = w_1 + 2*learning_rate*np.sum((y - y_p)*x_r)/len(y)
    w_0 = w_0 + 2*learning_rate*np.sum((y - y_p))/len(y)
    return w_2,w_1, w_0

# im_line,im_Loss is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
im_line, im_loss = [],[]

#For animation
def plotImageLine():
    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(x,y, label='Best Line Fit')
    plt.scatter(x_r, y_r, marker = '*',color = 'k', label='The real data')
    ax.plot(x_r,Predict(), label='Prediction Line')
    ax.legend()
    ax.grid()
    ax.set(xlabel='X', ylabel='Y',title='Learning rate = {}'.format(learning_rate))

    # Used to keep the limits constant
    ax.set_ylim(-10000, 3e18)
    ax.set_xlim(-10, 1500)

    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)

#For animation
def plotImageLoss():
    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(np.arange(len(LLL)), LLL )
    ax.grid()
    ax.set(xlabel='Iteration', ylabel='Y',title='Cost Function value'.format(learning_rate))

    # Used to keep the limits constant
    ax.set_ylim(0, 5e35)
    ax.set_xlim(-3, MaxIteration+1)

    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    imageLoss = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    imageLoss  = imageLoss.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imagesLoss.append(imageLoss)

#list for Animation image
images = []
imagesLoss = []
plotImageLine()
LLL = np.asarray(LL)
plotImageLoss()

#Loop for updateing the weights using gradient descent
counter = 0
for i in range(MaxIteration):
    counter +=1
    w2, w1, w0 = Gradient_descent(w2, w1, w0)
    LL.append(calculate_Loss())
    #Each 7 or (what ever) iteration plot to see learning progress
    if i % 6 == 0 and counter > 35:
        plotImageLine()
    if i % 8 == 0 :
        #Convert List to array for plot the loss function
        LLL = np.asarray(LL)
        plotImageLoss()

#Animation part
imageio.mimsave('./QuadraticRegressionLoss.gif', imagesLoss, fps=1)
imageio.mimsave('./QuadraticRegression.gif', images, fps=1)

#plt.axis([0, 100, 0, 100])
plt.figure(2)
plt.scatter(x_r, y_r, marker = '*',color = 'y', label='The real data')
plt.plot(x,y, label='Best Line Fit')
plt.plot(x_r,Predict(), color ='k', label='Prediction Line')
plt.legend()

#plot cost function
plt.figure(3)
plt.plot(np.arange(len(LLL)), LLL)