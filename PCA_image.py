#SYDE 675
#Matthew Cann
#20863891
#Assignment 1 Question 3

#.......................................................................IMPORTS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import cv2
import time

#..............................................................................
Directory = os.getcwd()

#.....................................................................CONSTANTS
pixels_num = 784 #28x28 

#.....................................................................DATA PREP
(images_train, labels_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
image_index = 7777 # You may select anything up to 60,000
image_index = 12000 # You may select anything up to 60,000

#Each image is Vectoized.......
#Current shape is 60000, 28x28, I want 60000, 784x1
images_train = images_train.reshape(images_train.shape[0], pixels_num)
X_data = images_train.T # Matrix of size Dxn or 784x60,000


nsample = images_train.shape[0] #number of features 60,000


#%%...................................................................FUNCTIONS
def eigen_analysis(X_data,nsample):
    X = X_data #784x60,000 data
    means = X.mean(axis=1).reshape(1,784)
    X_bar = np.subtract(X, means.T)
    denom = nsample - 1
    cov = np.matmul(X_bar,X_bar.T)/denom #784x784
    
    
    w, v = np.linalg.eig(cov)
    idx = np.argsort(w)[::-1] #Sorts the eigenvalues but in decending order using [::-1]
    w = w[idx]
    v = v[:,idx]
    return w, v, X_bar, means

def PCA(v, d, X_bar):
    U_matrix = v
    U_reduced = U_matrix[:, :d]
    W = U_reduced   # mapping function as a Dxd matrix in this case (784x2)
    Y = np.matmul(W.T,X_bar) # reduced function as a dxN matirx in this case (2, 60000)
    return W, Y

def Part_A(X_data):
    w, v, X_bar, means = eigen_analysis(X_data, nsample)
    d = 784
    W, Y = PCA(v, d, X_bar)
    return Y

def suitable_d(POV_all, w):
    '''Determines the suitable number of eigen values(w) to achieve the 
    specific proportion of variencce (POV)'''
    
    total = np.sum(w).real # Summation of all the eigen values.
    count = 0 #Sum of eigen values initialized to zero
    for i in range(len(w)):
        count += w[i].real #Adds the eigen value of index i to the count. 
        POV = count/total # Calculates POV 
        if POV >= POV_all: 
            # When the POV is equal or greater than the POV allowable it saves the value and d and breaks loop.
            POV_final = POV
            d_final = i + 1 #since i starts at 0
            break
    print('The suitable d proposed to achieve a proportion of variance POV = 95% is {} resulting in a POV of {:0.2f}%'.format(d_final,POV_final*100)    )
    return
def PCA_reconstruction(W,Y, means):
    '''PCA reconstruction takes the reduced Y(dxN) and returns the reconstructed X(DxN)'''
    X_recon = (np.dot(W, Y) + means.T).real #Reconstruts the images as a DxN matrix
    #recon = recon.T.real
    return X_recon

def z_end_script():
    print ('\nProgrammed by Matt Cann \nDate: ',\
    time.ctime(),' \nEnd of processing''\n'  )
    return

#%%........................................................................MAIN

#%%PARTA.......................................................................
'''PCA that takes X(DxN) and returns Y (dxN)'''
Y = Part_A(X_data)

#%%PARTB.......................................................................
'''Determines a suitable d using proportion of variance (POV) =95%'''
w, v, X_bar, means = eigen_analysis(X_data, nsample)
POV_all = 0.95

suitable_d(POV_all, w)
#%%PARTC.......................................................................
'''PCA reconstruction takes the reduced Y(dxN) and returns the reconstructed X(DxN) 
then applies to different values of d and average mean square error is calculated and plotetd'''

# Make list of different d values of {1, 20, 40, 60, 80, …, 760, 784} 
d_range = list(range(0, len(w), 20))[1:-1]
d_range.insert(0,1)
d_range.append(len(w))

w, v, X_bar, means= eigen_analysis(X_data, nsample)
MSE = []
for d in d_range:
    W,Y = PCA(v,d, X_bar)
    X_recon = PCA_reconstruction(W,Y, means)
    X_recon = X_recon.T # X_recon is DxN matrix, converts to NxD for easy index
    MSE.append( mean_squared_error(X_data.T, X_recon)) # Calculated the meansquared error

fig = plt.figure(20)
plt.plot(d_range,MSE)
plt.xlabel('Reconstructed d')
plt.ylabel('Mean Squared Error')
plt.savefig(Directory +'\partc2.png',bbox_inches="tight")
plt.show()

#%%PARTD.......................................................................
'''Reconstruct a sample from the class of number ‘8’ and show it as a ‘png’ image for d=
{1, 10, 50, 250, 784}.'''

index_8 = np.where(labels_train == 8)[0][0] # Finds the first index of the number 8. 

w, v, X_bar, means = eigen_analysis(X_data, nsample) # Re calc eigen values

original_image = images_train[index_8].reshape(28,28)
plt.imshow(original_image, cmap='Greys') # Show and save orginal image.
plt.imsave(Directory +'orginal.png', original_image, cmap='Greys')

for d in [1,10,50,250,784]:
    W,Y = PCA(v,d, X_bar)
    X_recon = PCA_reconstruction(W,Y, means)
    X_recon = X_recon.T # X_recon is DxN matrix, converts to NxD for easy index
    image = X_recon[index_8].reshape(28, 28)
    plt.figure(3)
    plt.imshow(image, cmap='Greys')
    #plt.imsave(Directory +'test.png', image, cmap='Greys')
    #cv2.imwrite('test.png', image, cmap='Greys')
    plt.show()

#%%PART E......................................................................
'''For the values of d= {1, 2, 3, 4, …, 784} plot eigenvalues (y-axis) versus d (x-axis).'''

d_range = list(range(0, len(w)+1, 1))[1:]

plt.figure(3)
plt.plot(d_range,w.real)
plt.xlabel('Values d')
plt.ylabel('Eigenvalues')
plt.savefig(Directory +'Eigen.png')
plt.show()

print ('\nProgrammed by Matthew Cann\nDate: ',\
    time.ctime(),' \nEnd of processing''\n'  )

z_end_script()