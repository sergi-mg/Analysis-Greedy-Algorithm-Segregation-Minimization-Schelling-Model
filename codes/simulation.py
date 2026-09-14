# Author: Sergi Martínez Galindo

#------------------------------------------------------------------------------------------------
#Libraries
#------------------------------------------------------------------------------------------------

import numpy as np
import scipy as sp
from numba import njit
import random as random
import subprocess
import sys
import time
import os
from scipy.optimize import curve_fit


#------------------------------------------------------------------------------------------------
#Functions
#------------------------------------------------------------------------------------------------

#initial matrix
def initial_matrix(L,rho_0,proportion):
    """Returns a LxL matrix with {-1,0,1} with a 0's density rho_0 and a
    certain proportion of -1's and 1's, proportion=N+/Ntotal."""

    N_plus=int(round((proportion)*(1-rho_0)*L**2,0))
    N_minus=int(round((1-proportion)*(1-rho_0)*L**2,0))
    matrix=np.zeros((L,L),dtype="float64")

    for k in range(N_plus):
        indexes_zero=np.where(matrix==0)
        i=np.random.randint(0,len(indexes_zero[1]))
        y=indexes_zero[0][i]
        x=indexes_zero[1][i]
        matrix[y][x]=1

    for k in range(N_minus):
        indexes_zero=np.where(matrix==0)
        i=np.random.randint(0,len(indexes_zero[1]))
        y=indexes_zero[0][i]
        x=indexes_zero[1][i]
        matrix[y][x]=-1

    return matrix

#System's state evaluation 
@njit
def G_function(matrix,threshold,alpha,L,rho_0):
    """Given a matrix 2D, with values {-1,0,1}, a threshold between 0 and 1 and
    the value alpha of G, returns a tuple containing a matrix T
    containing the proportion of the other type of agents around, a matrix h
    where happy agents are given a value 1 and the other positions are 0 and
    the value of the function G, global segregation and global happines. Agent
    alone is happy."""

    #kernel 
    kernel=np.ones((3,3),dtype="float64")
    kernel[1,1]=0
    limit=1 #int(3/2)

    #number of agents
    N_a=int(round((1-rho_0)*L**2,0))

    #vectors
    unhappy=np.zeros((L**2,2),dtype=np.int64)
    vacants=np.zeros((int(round(rho_0*L**2,0)),2),dtype=np.int64)

    H=0.
    A=0.
    B=0.
    vacant_counter=0
    unhappy_counter=0

    for i in range(L):
        for j in range(L):
            if matrix[i,j]==0:
                vacants[vacant_counter,0]=i
                vacants[vacant_counter,1]=j
                vacant_counter+=1
            else:
                C=0.
                D=0.
                for k in range(i-limit,i+limit+1):
                    for l in range(j-limit,j+limit+1):
                        if not ((i==k and j==l) or (k<0 or k>L-1 or l<0 or l>L-1)):
                            #S
                            A+=((matrix[i,j]*matrix[k,l])**2+matrix[i,j]*matrix[k,l])*kernel[k-(i-limit),l-(j-limit)]
                            B+=2*kernel[k-(i-limit),l-(j-limit)]*(matrix[i,j]*matrix[k,l])**2
                            #H
                            C+=((matrix[i,j]*matrix[k,l])**2-matrix[i,j]*matrix[k,l])*kernel[k-(i-limit),l-(j-limit)]
                            D+=2*kernel[k-(i-limit),l-(j-limit)]*(matrix[i,j]*matrix[k,l])**2
                        # end if
                    #end for
                #end for
                if D!=0:
                    if C/D>threshold:
                        unhappy[unhappy_counter,0]=i
                        unhappy[unhappy_counter,1]=j
                        unhappy_counter+=1
                    #end if
                #end if
            #end if
        #end for
    #end for

    if B!=0:
        S=A/B 
    else:
        S=0

    H=(N_a-unhappy_counter)/N_a

    #G function
    G=alpha*S-(1-alpha)*H

    return G,H,S,vacants,unhappy,unhappy_counter

#Greedy algorithm
@njit
def our_model(M_i, tau, alpha, L, rho_0):
    """Executes the greedy algorithm and returns a tupple with the final
    configuration, a tupple with (T,h,G,H,S) from the G_function
    applied to the final configuration and the number of iterations."""

    M=M_i.copy()

    N_v=int(round((rho_0)*L**2,0))
    G_list=np.zeros(N_v)
    N_uh=1 #different from zero to enter the loop
    T_f=0 #number of movements

    #algorithm
    for g in range(1000):
        if g==999:
            print("Final state not reached for alpha=",alpha," and rho_0=",rho_0)
        #end if
        if N_uh==0:
            break
        #end if 
        #how many unhappy agents? where? 
        G_out,H_out,S_out,vacants,uh_list,N_uh_out=G_function(M,tau,alpha,L,rho_0)
        if N_uh_out==0:
            break
        #end if 
        #movement selection
        N_uh=N_uh_out
        for i in range(N_uh_out):
            #we chose an unhappy agent at random
            rand_uh=np.random.randint(0,N_uh)
            #we look for the most suitable movement
            for j in range(N_v):
                #new matrix
                new_M=M.copy()
                #movement
                new_M[vacants[j,0],vacants[j,1]]=M[uh_list[rand_uh,0],uh_list[rand_uh,1]]
                new_M[uh_list[rand_uh,0],uh_list[rand_uh,1]]=0
                #G calculus
                G_out,H_out,S_out,vacants_out,uh_list_out,N_uh_try=G_function(new_M,tau,alpha,L,rho_0)
                #if the agent is unhappy we discard this movement (G>>1)
                for k in range(N_uh_try):
                    if (uh_list_out[k,0]==vacants[j,0])and(uh_list_out[k,1]==vacants[j,1]):
                        G_out=10.
                        break
                    #endif
                #endfor
                G_list[j]=G_out
            #end for
            #now we look for the minimum value of G
            vacant_min=np.argmin(G_list)
            G_min=G_list[vacant_min]
            if G_min>=9.0:
                #the agent cannot move, we go to the following agent
                uh_list[rand_uh,:]=uh_list[N_uh-1,:]
                uh_list[N_uh-1,:]=[0,0]
                N_uh=N_uh-1
                if N_uh==0:
                    break
                #endif
            else:
                #we execute the movement and restart the movement selection
                M[vacants[vacant_min,0],vacants[vacant_min,1]]=M[uh_list[rand_uh,0],uh_list[rand_uh,1]]
                M[uh_list[rand_uh,0],uh_list[rand_uh,1]]=0
                T_f=T_f+1
                break
            #endif
        #end for
    #end for
    G_out,H_out,S_out,vacants_out,uh_list_out,N_uh_try=G_function(M,tau,alpha,L,rho_0)
    S_f=S_out
    H_f=H_out

    return S_f,H_f,T_f


#------------------------------------------------------------------------------------------------
#Simulation
#------------------------------------------------------------------------------------------------

# Parameters
tau=0.5
L=20

e=10**(-5)

a1=np.array([0,0.001])
a2=np.arange(0.1,0.4,0.1)
a3=np.arange(0.4,0.6,0.05)
a4=np.arange(0.6,1+e,0.025)
alpha_values=np.concatenate((a1,a2,a3))
alpha_values=np.unique(alpha_values)

rho_0_values=np.arange(0.05,0.9+e,0.05)

N_sim=500

# Saving directories
from os.path import exists
from os import makedirs
    
directory_s="../data/"
if not exists(directory_s):
    makedirs(directory_s)

"""directory_m="../matrix/"
if not exists(directory_m):
    makedirs(directory_m)

directory_f="../data_results_fortran/"
if not exists(directory_f):
    makedirs(directory_f)"""

counter=0
seed_values=np.arange(0,len(rho_0_values)*len(alpha_values)*N_sim,N_sim)
print("Start simulations")
for i_r in range(np.size(rho_0_values)):
    rho_0=rho_0_values[i_r]
    for j_a in range(np.size(alpha_values)):
        alpha=alpha_values[j_a]
        data=np.zeros((N_sim,3),dtype="float64") #change 3 to 6 if you want to also simulate on fortran

        print(counter,rho_0,alpha)

        for i in range(N_sim):

            np.random.seed(seed_values[counter]+i)

            # Initial Condition
            M_i=initial_matrix(L,rho_0,0.5)

            """fD_M=open(directory_m+"matrix.dat", "w")
                                                np.savetxt(fD_M, M_i, fmt="%d")           
                                                fD_M.flush()
                                                os.fsync(fD_M.fileno())
                                                fD_M.close()""" #only for fortran simulation

            # Python Simulation
            final_state=our_model(M_i,tau,alpha,L,rho_0)
            data[i][:3]=final_state[:]

            """# Fortran Simulation
                                    
                                                if alpha!=0:
                                                    cmd = ["./schelling.exe", str(L), str(alpha)+"d0", str(tau)+"d0", str(rho_0)+"d0", str(counter)]
                                                else:
                                                    cmd = ["./schelling.exe", str(L), "0.d0", str(tau)+"d0", str(rho_0)+"d0", str(counter)]
                                    
                                                result = subprocess.run(cmd)
                                    
                                                results=np.loadtxt("../data_results_fortran/results_simulation.dat"
                                                                   ,usecols=[-3,-2,-1])
                                    
                                                data[i][3:]=results[:]"""

        counter+=1

        #save the data
        name=directory_s+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+".dat"
        np.savetxt(name,data)

