import numpy as np
import cv2
from utils import *
class My_Solver(object):
    def __init__(self,phi, g, lambda1, mu, alfa, epsilon, timestep, iter_inner, iter_outter):
        self.phi=phi
        self.g=g
        self.lambda1=lambda1
        self.mu=mu
        self.alfa=alfa
        self.epsilon=epsilon
        self.timestep=timestep
        self.iter_inner=iter_inner
        self.iter_outter=iter_outter
        self.image_list=[phi]
    def run(self):
        phi=self.phi
        [vx, vy] = np.gradient(self.g)
        for j in range(self.iter_outter):
            for k in range(self.iter_inner):
                phi = neumannboundcond(phi)
                [phi_x, phi_y] = np.gradient(phi)
                s = np.sqrt((phi_x**2.+phi_y**2.))
                stablizer = 1e-10
                Nx = phi_x/(s+stablizer)
                Ny = phi_y/(s+stablizer)
                curvature = div(Nx, Ny)
                distRegTerm = distReg_p2(phi)
                diracPhi = Dirac(phi, self.epsilon)
                areaTerm = diracPhi*self.g
                edgeTerm = diracPhi*(vx*Nx+vy*Ny)+diracPhi*self.g*curvature
                phi = phi+self.timestep*(self.mu*distRegTerm+self.lambda1*edgeTerm+self.alfa*areaTerm)
                self.image_list.append(phi)
        return self.image_list


