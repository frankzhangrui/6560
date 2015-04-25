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
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return np.float32(h)

def distReg_p2(phi):
    [phi_x, phi_y] = np.gradient(phi)
    s = np.sqrt((phi_x**2.+phi_y**2.))
    a = np.logical_and(s >= 0., s<=1.)
    b = s > 1.
    ps = a*np.sin(2.*np.pi*s)/2.*np.pi +b*(s-1.)
    dps = ((ps != 0.)*ps+(ps == 0.))/((s != 0.)*s+(s == 0.))
    f = div((dps*phi_x-phi_x), (dps*phi_y-phi_y))+4.*del2(phi)
    return f

def div(nx, ny):
    [nxx, junk] = np.gradient(nx)
    [junk, nyy] = np.gradient(ny)
    f = nxx+nyy
    return f

def Dirac(x, sigma):
    f = 1/(2*sigma)*(1+np.cos(np.pi*x/sigma))
    b = np.logical_and(x<=sigma, x >= -sigma)
    f = f*b
    return f

def plot(I,u,iter):
    origin = 'lower'
    plt.imshow(I,'gray')
    plt.title("This is "+str(iter)+" th iteration")
    plt.axis('off')
    plt.axis('equal')
    plt.contour(u,0,colors = 'r',origin=origin,hold='on')
    plt.xticks([]),plt.yticks([])
    plt.show(block=False)
    plt.pause(0.25)
    plt.savefig(str(iter)+".png")
    plt.clf()
    #plt.close("all")
def print_property(I):
    print "max value is " + str(np.amax(I))+ " min value is " + str(np.amin(I)) + str(I.shape) +str(I.dtype)
    
    