import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.sparse import spdiags,linalg,eye
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

#---------------------ISING MODEL--------------------------------#
def initialstate(N):   
    ''' 
    Generates a random spin configuration for initial condition
    '''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state



def mcmove(config, beta):
    '''
    Monte Carlo move using Metropolis algorithm 
    '''
    N=len(config)
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

#------------------------WOLFF CLUSTER--------------------------------------
J = 1

def indexes(a,b):   # search b in a
    for i in range(len(a)):
        if a[i]==b:
            return 1
    return 0

def wolff(config,T):
    n=len(config)            # size of the configuration
    p=1-np.exp(-2*J/T)       # Calculating p: probabilty for considering in cluster
    
    for k in range(n**2):
        a = np.random.randint(0, n)   # random position
        b = np.random.randint(0, n)   
        
        c=[]               # storing the positions in the same cluster
        c.append((a,b))
        
        fo=[]
        fo.append((a,b))   # F_old
        
        while fo!=[]:
            fn=[]
            for i in range(len(fo)):   # for all recorded position in cluster
                a,b=fo[i][0],fo[i][1]
                # Now considering 4 neighbouring positions
                nei=[((a+1)%n,b),(a,(b+1)%n),((a-1)%n,b),(a,(b-1)%n)]  # periodic lattice
                
                for j in range(4):
                    a1,b1=nei[j]
                    if config[a][b]==config[a1][b1] and indexes(c,nei[j])==0:   # checking if spin is same or the neighbour is already taken into count
                        
                        if rand()<p:
                            fn.append((a1,b1))      # adding the position in the cluster
                            c.append((a1,b1))
                        
            
            fo=copy.deepcopy(fn)
            
    
    # flip the signs of the cluster
    
    for i in range(len(c)):
        aa,bb=c[i]
        config[aa][bb]*=-1
    
    return config

#-----------------Calculate----------------------------


def calcEnergy(config):
    '''
    Energy of a given configuration
    '''
    energy = 0 
    N=len(config)
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy/2.  # to compensate for over-counting



def calcMag(config):
    '''
    Magnetization of a given configuration
    '''
    mag = np.sum(config)
    return mag


nt      = 32         #  number of temperature points
N       = 10        #  size of the lattice, N x N
eqSteps = 2**7       #  number of MC sweeps for equilibration
mcSteps = 2**7       #  number of MC sweeps for calculation


#T       = np.linspace(1.53, 3.28, nt); 
T       = np.linspace(1, 5, nt); 
E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 

# divide by number of samples, and by system size to get intensive values
#  MAIN PART OF THE CODE
#
emm=np.zeros(eqSteps)
emc=np.zeros(eqSteps)
steps=np.zeros(eqSteps)


for tt in range(nt):
    print("Temperature::",tt,"=",T[tt])
    config1 = initialstate(N)         # initialise
    config2 = initialstate(N)
    E1 = M1 = E2 = M2 = 0
    iT=1.0/T[tt]; iT2=iT*iT;
    
    for i in range(eqSteps):         # equilibrate
        
        wolff(config2, T[tt])
        if tt==4:
            mcmove(config1, iT)           # Monte Carlo moves
            steps[i]=i
            
            emm[i]=calcEnergy(config1)/(eqSteps*N*N)
            emc[i]=calcEnergy(config2)/(eqSteps*N*N)
            
    if tt==4:
        
        plt.plot(steps,emm,label='Metropolis using monte carlo')
        plt.plot(steps,emc, label='Wolff cluster ')
        plt.title('T=%s'%T[tt])
        plt.legend()
        plt.xlabel('No of steps::(%s)'%eqSteps)
        plt.ylabel('Energy')
        plt.show()
    
    for i in range(mcSteps):
        wolff(config2, T[tt])
        
        Ene = calcEnergy(config2)     # calculate the energy
        Mag = calcMag(config2)        # calculate the magnetisation

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag 
        E2 = E2 + Ene*Ene
    # divide by number of sites and iterations to obtain intensive values    
    E[tt] = n1*E1
    M[tt] = Mag/(N**2)
    C[tt] = (n1*E2 - n2*E1*E1)*iT2
    #n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 
    X[tt] = (M2/(mcSteps*N*N) - Mag*Mag/(mcSteps*mcSteps*N*N))*iT
    


plt.scatter(T, E, s=50, marker='o', color='IndianRed')
plt.title('After %s steps'%mcSteps)
plt.legend()
plt.xlabel('Temperature(T)')
plt.ylabel('Energy')
plt.show()

plt.scatter(T, M, s=50, marker='o', color='RoyalBlue')
plt.title('After %s steps'%mcSteps)
plt.legend()
plt.xlabel('Temperature(T)')
plt.ylabel('Magnetisation')
plt.show()

plt.scatter(T, C, s=50, marker='o', color='IndianRed')
plt.title('After %s steps'%mcSteps)
plt.legend()
plt.xlabel('Temperature(T)')
plt.ylabel('Specific Heat')
plt.show()

plt.scatter(T, X, s=50, marker='o', color='RoyalBlue')
plt.title('After %s steps'%mcSteps)
plt.legend()
plt.xlabel('Temperature(T)')
plt.ylabel('Susceptibility')
plt.show()
    
    
    
        

