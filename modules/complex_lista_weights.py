#!/usr/bin/env python3
import numpy as np
from numpy import pi, fft
import torch
from torch.nn.functional import relu, mse_loss, conv1d
from torch.nn import Module, ReLU


class ComplexLISTA_Weights(Module):
    
    def __init__(self, M, N, maxit):
        super(ComplexLISTA_Weights, self).__init__()
    
        # Real and imaginary Wg and We matricies 
        self.Wre = torch.nn.Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.Wie = torch.nn.Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.wg = torch.nn.Parameter(torch.ones([maxit+1, M]), requires_grad=True) # dimension such that it works with conv1d
        
        # alpha and lambda hyper-parameters to LASSO/ISTA
        #self.alpha = torch.nn.Parameter(torch.zeros(maxit), requires_grad=True)
        #self.lamda = torch.nn.Parameter(torch.zeros(maxit), requires_grad=True)
        self.theta = torch.nn.Parameter(torch.ones(maxit+1), requires_grad=True)
        
        # Save the passed values
        self.M = M
        self.N = N
        self.maxit = maxit

        # Create useful relu layer
        self.relu = ReLU()

        # Assuming the measurement model
        self.complex_exp = lambda x : np.exp(2j*pi*x)
        self.fgrid = fft.fftfreq(N)
        self.ula = np.arange(M)
        self.arg = np.outer(self.ula, self.fgrid)

        # Predefine useful matricies
        self.C = torch.from_numpy(np.cos(self.arg)).to(torch.float32).unsqueeze(0).unsqueeze(0)
        self.S = torch.from_numpy(np.sin(self.arg)).to(torch.float32).unsqueeze(0).unsqueeze(0)

        return
    
    def forward(self, yr, yi, epsilon=1e-10):
        
        Wret = torch.transpose(self.Wre[0], 0, 1)
        Wiet = torch.transpose(self.Wie[0], 0, 1)
                
        # Apply We branch to y to 0-th iteration
        zr = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
        zi = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
        
        # Apply soft-thresholding according to Eldar's paper.
        xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
        xr = torch.divide(zr, xabs + 1) * self.relu(xabs - self.theta[0])
        xi = torch.divide(zi, xabs + 1) * self.relu(xabs - self.theta[0])
        
        for t in range(1, self.maxit+1):

            Wret = torch.transpose(self.Wre[t], 0, 1)
            Wiet = torch.transpose(self.Wie[t], 0, 1)
        
            # Apply We branch to y to t-th iteration
            ar = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
            ai = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
            
            # Apply hg conv1d branch to x^(t) for t-th iteration
            hrgt = self.wg[t] @ self.C
            higt = self.wg[t] @ self.S
            br = conv1d(xr.unsqueeze(1), hrgt, padding='same') - conv1d(xi.unsqueeze(1), higt, padding='same')
            bi = conv1d(xi.unsqueeze(1), hrgt, padding='same') + conv1d(xr.unsqueeze(1), higt, padding='same')
            
            # Add the two branches                                                                           
            zr = ar + br.squeeze(1)
            zi = ai + bi.squeeze(1)
            
            # Apply soft-thresholding
            xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
            xr = torch.divide(zr, xabs + 1) * self.relu(xabs - self.theta[t])
            xi = torch.divide(zi, xabs + 1) * self.relu(xabs - self.theta[t])
      
        return xr, xi

if __name__ == "__main__":
    model = ComplexLISTA_Weights(64, 512, 10)
    
    print(f"{model.arg.shape = }")
    print(f"{model.C.shape = }")

    test = model.wg[0] @ model.C
    print(f"{test.shape = }")
    print(test[0, 0, :10])
    print(torch.sum(model.C[0, 0, ...], dim=0)[:10])
