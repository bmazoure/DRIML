import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

class Ising():
    ''' Simulating the Ising model
        Taken from https://rajeshrinet.github.io/blog/2014/ising-model/ '''
    def __init__(self,beta=1/0.4):
        self.beta = beta
    def reset(self,N):
        config = 2*np.random.randint(2, size=(N,N))-1
        self.config = config
        return config
        
    ## monte carlo moves
    def mcmove(self, config, N, beta):
        ''' This is to execute the MC moves using 
        Metropolis algorithm such that detailed
        balance condition is satisified'''
        for i in range(N):
            for j in range(N):            
                    a = np.random.randint(0, N)
                    b = np.random.randint(0, N)
                    s =  config[a, b]
                    nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                    cost = 2*s*nb
                    if cost < 0:	
                        s *= -1
                    elif rand() < np.exp(-cost*beta):
                        s *= -1
                    config[a, b] = s
        return config
    
    def simulate_n(self,n_steps):   
        ''' This module simulates the Ising model'''
        for i in range(n_steps):
            self.mcmove(self.config, self.config.shape[0], self.beta)
        return self.config
                 
                    
    def configPlot(self, f, config, i, N, n_):
        ''' This modules plts the configuration once passed to it along with time etc '''
        X, Y = np.meshgrid(range(N), range(N))
        sp =  f.add_subplot(3, 3, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, config, cmap=plt.cm.RdBu);
        plt.title('Time=%d'%i); plt.axis('tight')    
    plt.show()