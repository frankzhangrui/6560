
import numpy as np
import scipy
from Solver import *
from utils import *
import cv2
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

Img = np.float32(cv2.imread('twocells.bmp',0))
timestep = 5.
mu = 0.2/timestep
iter_inner = 5
iter_outer = 40
lambda1 = 5
alfa = -3
epsilon = 1.5
sigma = 0.5
G = fspecial_gauss((3,3),sigma)
Img_smooth = cv2.filter2D(Img,-1,G)
[Ix, Iy] = np.gradient(Img_smooth)
f = Ix**2.+Iy**2.
g = np.divide(1., 1.+f)
c0 = 2.
origin = 'lower'
initialLSF = c0*np.ones(Img.shape,np.float32)
initialLSF[10:55, 10:75] = -c0
phi = initialLSF
my_solver=My_Solver(phi, g, lambda1, mu, alfa, epsilon, timestep, iter_inner, iter_outer)
phi_list=my_solver.run()
for i in range(iter_inner*iter_outer):
    if(i%10==0):
        plot(Img,phi_list[i],i)




'''
for n in np.arange(1., (iter_outer)+1):
    phi = drlse_edge(phi, g, lambda1, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
    if np.mod(n, 2.) == 0.:
        plt.figure(2.)
        plt.imshow(Img,'gray')
        plt.axis('off')
        plt.axis('equal')
        plt.contour(phi,0,colors = 'r',origin=origin,hold='on')
        plt.pause(0.5)
        plt.clf()
    
    

#% refine the zero level contour by further level set evolution with alfa=0
alfa = 0.
iter_refine = 10.
phi = drlse_edge(phi, g, lambda1, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
finalLSF = phi
plt.figure(2.)
plt.imshow(Img,'gray')
plt.axis('off')
plt.axis('equal')
plt.contour(phi, 0,colors = 'r',origin=origin,hold='on')
plt.contour(phi, 0,colors = 'r',origin=origin,hold='on')
string = np.array(np.hstack(('Final zero level contour, ',str((np.dot(iter_outer, iter_inner)+iter_refine)), ' iterations')))
plt.title(string)
fig = plt.figure()
print finalLSF.shape
ax = fig.add_subplot(111, projection='3d')
X = np.arange(0, Img.shape[1], 1)
Y = np.arange(0, Img.shape[0], 1)
X,Y=np.meshgrid(X,Y)
print X.shape
ax.plot_surface(X,Y,-finalLSF)
#% for a better view, the LSF is displayed upside down
#plt.contour(phi, 0, colors = 'r', origin=origin,hold='on')
string = np.array(np.hstack(('Final level set function, ', str((np.dot(iter_outer, iter_inner)+iter_refine)), ' iterations')))
plt.title(string)
plt.axis('on')
[nrow, ncol] = Img.shape
plt.show()
'''