import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
def neumannboundcond(f):
    nrow, ncol = f.shape
    g=f
    g[0, 0]=g[2,2]
    g[0, ncol-1]=g[2,ncol-3]
    g[nrow-1, 0]=g[nrow-3,0]
    g[nrow-1,ncol-1]=g[nrow-3, nrow-3]
    g[0, 1:-2]=g[2,1:-2]
    g[nrow-1,1:-2]=g[nrow-3, 1:-2]
    g[1:-2, 0]=g[1:-2, 2]
    g[1:-2, ncol-1]=g[1:-2, ncol-3]
    return np.float32(g)
def curvature_central(u):
    u=np.float32(u)
    u_x,u_y=np.gradient(u)
    norm=cv2.pow(np.power(u_x,2)+cv2.pow(u_y,2)+1E-10,0.5)
    N_x=cv2.divide(u_x,norm)
    N_y=cv2.divide(u_y,norm)
    N_xx,junk=np.gradient(N_x)
    junk,N_yy=np.gradient(N_y)
    return np.float32(N_xx+N_yy)
def del2(M):
    dx = 1
    dy = 1
    rows, cols = M.shape
    dx = dx * np.ones ((1, cols - 1))
    dy = dy * np.ones ((rows-1, 1))

    mr, mc = M.shape
    D = np.zeros ((mr, mc))

    if (mr >= 3):
        ## x direction
        ## left and right boundary
        D[:, 0] = (M[:, 0] - 2 * M[:, 1] + M[:, 2]) / (dx[:,0] * dx[:,1])
        D[:, mc-1] = (M[:, mc - 3] - 2 * M[:, mc - 2] + M[:, mc-1]) \
            / (dx[:,mc - 3] * dx[:,mc - 2])

        ## interior points
        tmp1 = D[:, 1:mc - 1] 
        tmp2 = (M[:, 2:mc] - 2 * M[:, 1:mc - 1] + M[:, 0:mc - 2])
        tmp3 = np.kron (dx[:,0:mc -2] * dx[:,1:mc - 1], np.ones ((mr, 1)))
        D[:, 1:mc - 1] = tmp1 + tmp2 / tmp3

    if (mr >= 3):
        ## y direction
        ## top and bottom boundary
        D[0, :] = D[0,:]  + \
            (M[0, :] - 2 * M[1, :] + M[2, :] ) / (dy[0,:] * dy[1,:])

        D[mr-1, :] = D[mr-1, :] \
            + (M[mr-3,:] - 2 * M[mr-2, :] + M[mr-1, :]) \
            / (dy[mr-3,:] * dx[:,mr-2])

        ## interior points
        tmp1 = D[1:mr-1, :] 
        tmp2 = (M[2:mr, :] - 2 * M[1:mr - 1, :] + M[0:mr-2, :])
        tmp3 = np.kron (dy[0:mr-2,:] * dy[1:mr-1,:], np.ones ((1, mc)))
        D[1:mr-1, :] = tmp1 + tmp2 / tmp3

    return np.float32(D / 4.)
def fspecial_gauss(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return np.float32(h)
def Binary_Fit(Img, u, KI, KONE, Ksigma, epsilon):
    Hu=np.float32(0.5*(1+(2/np.pi)*np.arctan(u/epsilon)))
    I=cv2.multiply(Img,Hu)
    c1=cv2.filter2D(Hu,-1,Ksigma)
    c2=cv2.filter2D(I,-1,Ksigma)
    f1=cv2.divide(c2,c1)
    f2=cv2.divide(KI-c2,KONE-c1)
    return (f1,f2)
def plot(I,u,iter):
    origin = 'lower'
    plt.imshow(I,'gray')
    plt.title("This is "+str(iter)+" th iteration")
    plt.axis('off')
    plt.axis('equal')
    plt.contour(u,0,colors = 'r',origin=origin,hold='on')
    plt.xticks([]),plt.yticks([])
    plt.show(block=False)
    plt.pause(0.1)
    plt.savefig(str(iter)+".bmp")
    plt.clf()
    #plt.close("all")
def print_property(I):
    print "max value is " + str(np.amax(I))+ " min value is " + str(np.amin(I)) + str(I.shape) +str(I.dtype)
    
    