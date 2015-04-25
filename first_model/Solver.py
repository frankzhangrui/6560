import numpy as np
import cv2
from utils import *
class My_Solver(object):
    def __init__(self,u0,Img,Ksigma,KI,KONE,nu,timestep,mu,lambda1,lambda2,epsilon,numIter):
        self.u=u0;
        self.Img=Img;
        self.Ksigma=Ksigma
        self.KI=KI
        self.KONE=KONE
        self.nu=nu
        self.timestep=timestep
        self.mu=mu
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.epsilon=epsilon
        self.numIter=numIter
        self.image_list=[u0]
    def run(self):
        u=self.u
        for k in range(self.numIter):
            u=neumannboundcond(u)
            K=curvature_central(u)
            DrcU=(self.epsilon/np.pi)/(self.epsilon**2+cv2.pow(u,2))
            f1,f2=Binary_Fit(self.Img, u, self.KI, self.KONE, self.Ksigma, self.epsilon)
            s1=self.lambda1*cv2.pow(f1,2)-self.lambda2*cv2.pow(f2,2)
            s2=self.lambda1*f1-self.lambda2*f2;
            dataForce=(self.lambda1-self.lambda2)*cv2.multiply(cv2.multiply(self.KONE,self.Img),self.Img)+cv2.filter2D(s1,-1,self.Ksigma)-2*cv2.multiply(self.Img,cv2.filter2D(s2,-1,self.Ksigma))
            A=-cv2.multiply(np.float32(DrcU),np.float32(dataForce))
            P=self.mu*(4*del2(u)-K)
            L=self.nu*cv2.multiply(np.float32(DrcU),np.float32(K))
            u=u+self.timestep*(L+P+A)
            self.image_list.append(u)
        return self.image_list


