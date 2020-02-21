# PCA_image
This program uses PCA image compression of the MNIST dataset which contains a set of images containing the digits 0 to 9. Each image in the data set is a 28x28 image.
The program has the following functions:
1) Programs PCA that takes the image data X(DxN) and returns Y(dxN) where N is the number of
samples, D is the number of input features, and d is the number of features selected by
the PCA algorithm. 
2) Propose a suitable d using proportion of variance (POV) =95%
3) Program PCA reconstruction that takes 𝑌𝑃𝐶𝐴(dxN) and returns 𝑋̂ (DxN) (i.e., a
reconstructed image). For different values of d= {1, 20, 40, 60, 80, …, 760, 784}
reconstruct all samples and calculate the average mean square error (MSE). 
4)Reconstruct a sample from the class of number ‘8’ and show it as a ‘png’ image for d=
{1, 10, 50, 250, 784}.  
5) For the values of d= {1, 2, 3, 4, …, 784} plot eigenvalues (y-axis) versus d (x-axis).  

