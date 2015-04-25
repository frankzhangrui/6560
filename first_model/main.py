from utils import *
from Solver import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
import sys
if __name__ == "__main__":
    Img=np.float32(cv2.imread("data/1.bmp",0));
    iterNum=300
    lambda1=1.0
    lambda2=2.0
    nu=0.004*255*255
    u=np.ones(Img.shape,np.float32)*(2);
    u[26:70,28:90]=-2;
    timestep=0.1
    mu=1
    epsilon=1.0
    sigma=3
    K=fspecial_gauss((np.uint8(4*sigma+1),np.uint8(4*sigma+1)),sigma)
    I=Img;
    KI=cv2.filter2D(Img,-1,K)
    KONE=cv2.filter2D(np.ones(Img.shape,np.float32),-1,K)
    my_solver=My_Solver(u,I,K,KI,KONE,nu,timestep,mu,lambda1,lambda2,epsilon,iterNum)
    t0 = time()
    results=my_solver.run()
    t1 = time()
    print "solving is done"+" timing of this program is " +str (t1-t0)+" seconds"
    print "now preparing to plot the result of every 10 iteration"
    for i in range(iterNum):
        if(i%10==0):
            plot(I,results[i],i)