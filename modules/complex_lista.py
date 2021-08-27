#!/usr/bin/env python3
import torch
from torch.nn.functional import relu, mse_loss, conv1d
from torch.nn import Module, ReLU

class ComplexLISTA(Module):
    
    def __init__(self, M, N, maxit):
        super(ComplexLISTA, self).__init__()
    
        # Real and imaginary Wg and We matricies 
        self.Wre = torch.nn.Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.Wrg = torch.nn.Parameter(torch.zeros([maxit+1, N, N]), requires_grad=True)
        self.Wie = torch.nn.Parameter(torch.zeros([maxit+1, N, M]), requires_grad=True)
        self.Wig = torch.nn.Parameter(torch.zeros([maxit+1, N, N]), requires_grad=True)
        
        # alpha and lambda hyper-parameters to LASSO/ISTA
        #self.alpha = torch.nn.Parameter(torch.zeros(maxit), requires_grad=True)
        #self.lamda = torch.nn.Parameter(torch.zeros(maxit), requires_grad=True)
        self.theta = torch.nn.Parameter(torch.ones(maxit+1), requires_grad=True)
        
        # Save the passed values
        self.M = M
        self.N = N
        self.maxit = maxit
        self.relu = ReLU()

        return
    
    def forward(self, yr, yi, epsilon=1e-10):
        
        Wret = torch.transpose(self.Wre[0], 0, 1)
        Wiet = torch.transpose(self.Wie[0], 0, 1)
        Wrgt = torch.transpose(self.Wrg[0], 0, 1)
        Wigt = torch.transpose(self.Wig[0], 0, 1)
                
        # Apply We branch to y to 0-th iteration
        zr = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
        zi = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
        
        # Apply soft-thresholding according to Eldar's paper.
        xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
        xr = torch.divide(zr, xabs + 1) * self.relu(xabs - self.theta[0])
        xi = torch.divide(zi, xabs + 1) * self.relu(xabs - self.theta[0])

        #soft = torch.divide(self.theta[0], self.theta[0])
        #soft = torch.divide(self.theta[0], torch.max(xabs, self.theta[0]))
        #soft = 1 - torch.divide(self.theta[0], relu(xabs - self.theta[0]) + self.theta[0]) 
        #soft = 1 - torch.divide(self.theta[0], self.relu(xabs - self.theta[0]) + self.theta[0]) 
        #xr = torch.multiply(zr, softr)
        #xi = torch.multiply(zi,  softi)
        
        
        for t in range(1, self.maxit+1):
            
            Wret = torch.transpose(self.Wre[0], 0, 1)
            Wiet = torch.transpose(self.Wie[0], 0, 1)
            Wrgt = torch.transpose(self.Wrg[0], 0, 1)
            Wigt = torch.transpose(self.Wig[0], 0, 1)
        
            # Apply We branch to y to t-th iteration
            ar = torch.matmul(yr, Wret) - torch.matmul(yi, Wiet)
            ai = torch.matmul(yi, Wret) + torch.matmul(yr, Wiet)
            
            # Apply Wg branch to x^(t) for t-th iteration
            br = torch.matmul(xr, Wrgt) - torch.matmul(xi, Wigt)
            bi = torch.matmul(xi, Wrgt) + torch.matmul(xr, Wigt)
            
            # Add the two branches                                                                           
            zr = ar + br
            zi = ai + bi
            
            # Apply soft-thresholding
            xabs = torch.sqrt(torch.square(zr) + torch.square(zi) + epsilon)
            #xr = zr * torch.pinverse(xabs).t() * self.relu(xabs - self.theta[t])
            #xi = zi * torch.pinverse(xabs).t() * self.relu(xabs - self.theta[t])
            xr = torch.divide(zr, xabs + 1) * self.relu(xabs - self.theta[t])
            xi = torch.divide(zi, xabs + 1) * self.relu(xabs - self.theta[t])
            #soft = torch.divide(self.theta[t], self.theta[t]) 
            #soft = torch.divide(self.theta[t], torch.max(xabs, self.theta[t]))
            #soft = 1 - torch.divide(self.theta[t], self.relu(xabs - self.theta[t]) + self.theta[t]) 
            #soft = 1
            #xr = torch.multiply(zr, softr)
            #xi = torch.multiply(zi, softi)
            
            #print(f"{yr[0,0] = }, {zr[0,0] = }, {xr[0,0] = }, {self.theta[0] = }, {soft[0,0] = }, {self.Wre[0,0] = }, {self.Wrg[0,0] = }")
      
        return xr, xi
