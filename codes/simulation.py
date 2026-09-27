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

#Classic Schelling
@njit
def classic_schelling(M_i, tau, L, rho_0):
    """Executes the classic algorithm and returns a tupple with the final
    configuration, a tupple with (T,h,G,H,S) from the G_function
    applied to the final configuration and the number of iterations."""

    M=M_i.copy()

    N_v=int(round((rho_0)*L**2,0))
    N_uh=1 #different from zero to enter the loop
    T_f=0 #number of movements
    alpha=0.5 #it will not be used, only for reusing the system's state evaluation function

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
            vacants_try=vacants.copy()
            N_p_v=N_v #possible vacants
            movement=0 #check for movement
            for j in range(N_v):
                #select a random vacant
                rand_vacant=np.random.randint(0,N_p_v)
                #new matrix
                new_M=M.copy()
                #movement
                new_M[vacants_try[rand_vacant,0],vacants_try[rand_vacant,1]]=M[uh_list[rand_uh,0],uh_list[rand_uh,1]]
                new_M[uh_list[rand_uh,0],uh_list[rand_uh,1]]=0
                #new state
                G_out,H_out,S_out,vacants_out,uh_list_out,N_uh_try=G_function(new_M,tau,alpha,L,rho_0)
                #if the agent is unhappy we discard this movement and check new vacant 
                unhappy=0

                for k in range(N_uh_try):
                    if (uh_list_out[k,0]==vacants_try[rand_vacant,0])and(uh_list_out[k,1]==vacants_try[rand_vacant,1]):
                        unhappy=1
                        break
                    #endif
                #endfor

                if unhappy==0:
                    #make the movement
                    M[vacants_try[rand_vacant,0],vacants_try[rand_vacant,1]]=M[uh_list[rand_uh,0],uh_list[rand_uh,1]]
                    M[uh_list[rand_uh,0],uh_list[rand_uh,1]]=0
                    T_f=T_f+1
                    movement=1
                    break
                else:
                    #the agent cannot move, we go to the following vacant
                    vacants_try[rand_vacant,:]=vacants_try[N_p_v-1,:]
                    vacants_try[N_p_v-1,:]=[0,0]
                    N_p_v=N_p_v-1
                    if N_p_v==0:
                        break
                    #endif
                #endif

            #end for
            #we check if the agent has moved
            if movement==1:
                break
            else:
                #the agent cannot move, we go to the following agent
                uh_list[rand_uh,:]=uh_list[N_uh-1,:]
                uh_list[N_uh-1,:]=[0,0]
                N_uh=N_uh-1
                if N_uh==0:
                    break
                #endif
            #endif
        #end for
    #end for

    G_out,H_out,S_out,vacants_out,uh_list_out,N_uh_try=G_function(M,tau,alpha,L,rho_0)
    S_f=S_out
    H_f=H_out

    return S_f,H_f,T_f


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
    for g in range(1000000):
        if g==999999:
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
#Improved version
#------------------------------------------------------------------------------------------------
@njit
def systems_state(matrix,threshold,L,rho_0):


    limit=1 #Moore Neighbpurhood
    
    #outputs
    N_dif=0
    N_same=0
    N_happy=0
    Ms=np.zeros((L,L,3),dtype=np.int64)

    #loop
    for i in range(L):
        for j in range(L):
            if matrix[i,j]!=0:
                for k in range(i-limit,i+limit+1):
                    for l in range(j-limit,j+limit+1):
                        if not ((i==k and j==l) or (k<0 or k>L-1 or l<0 or l>L-1)):
                            if matrix[k,l]!=0:
                                if matrix[i,j]==matrix[k,l]:
                                    Ms[i,j,0]+=1
                                    N_same+=1
                                else:
                                    Ms[i,j,1]+=1
                                    N_dif+=1
                                #endif
                            #endif
                        #endif
                    #end for
                #end for
                if (Ms[i,j,0]+Ms[i,j,1])!=0:
                    if Ms[i,j,1]/(Ms[i,j,0]+Ms[i,j,1])<=threshold:
                        Ms[i,j,2]=1
                        N_happy+=1
                    #endif
                else:
                    Ms[i,j,2]=1
                    N_happy+=1
                #endif
            #endif
        #end for
    #end for
    N_same=N_same/2
    N_dif=N_dif/2

    return Ms,N_same,N_dif,N_happy
#endfunction

@njit
def movement(new_coord,threshold,L,rho_0,matrix,Ms):

    limit=1 #Moore Neighbpurhood

    i_n=new_coord[0]
    j_n=new_coord[1]

    Mc=Ms.copy()


    delta_h=0
    delta_Ni=0
    delta_Ndif=0

    Mc[i_n,j_n,:]=[0,0,0]

    happy=0

    sign=matrix[new_coord[0],new_coord[1]]

    for k in range(i_n-limit,i_n+limit+1):
        for l in range(j_n-limit,j_n+limit+1):
            if not ((i_n==k and j_n==l) or (k<0 or k>L-1 or l<0 or l>L-1)):
                if matrix[k,l]!=0:
                    if sign==matrix[k,l]:
                        Mc[k,l,0]+=1
                        Mc[i_n,j_n,0]+=1
                        delta_Ni+=1
                    else:
                        Mc[k,l,1]+=1
                        Mc[i_n,j_n,1]+=1
                        delta_Ndif+=1
                    #endif
                    #happines
                    if Mc[k,l,1]/(Mc[k,l,0]+Mc[k,l,1])<=threshold:
                        Mc[k,l,2]=1
                    else:
                        Mc[k,l,2]=0
                    #endif

                    if Mc[k,l,2]>Ms[k,l,2]:
                        delta_h+=1
                    elif Mc[k,l,2]<Ms[k,l,2]:
                        delta_h-=1
                    #endif
                #endif

                
            #endif
        #end for
    #end for
    if (Mc[i_n,j_n,0]+Mc[i_n,j_n,1])!=0:
        if Mc[i_n,j_n,1]/(Mc[i_n,j_n,0]+Mc[i_n,j_n,1])<=threshold:
            Mc[i_n,j_n,2]=1
            delta_h+=1
            happy=1
        #endif
    else:
        Mc[i_n,j_n,2]=1
        delta_h+=1
        happy=1
    #endif

    return delta_Ni,delta_Ndif,delta_h,happy
#end function

#Classic Schelling
@njit
def classic_schelling_2(M_i, tau, L, rho_0):
    """Executes the classic algorithm and returns a tupple with the final
    configuration, a tupple with (T,h,G,H,S) from the G_function
    applied to the final configuration and the number of iterations."""

    limit=1

    M=M_i.copy()

    N_v=int(round((rho_0)*L**2,0))
    N_uh=1 #different from zero to enter the loop
    T_f=0 #number of movements
    alpha=0.5 #it will not be used, only for reusing the system's state evaluation function

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
            vacants_try=vacants.copy()
            N_p_v=N_v #possible vacants
            move_agent=0 #check for movement
            for j in range(N_v):
                #select a random vacant
                rand_vacant=np.random.randint(0,N_p_v)

                #movement
                M[vacants_try[rand_vacant,0],vacants_try[rand_vacant,1]]=M[uh_list[rand_uh,0],uh_list[rand_uh,1]]
                M[uh_list[rand_uh,0],uh_list[rand_uh,1]]=0

                #if the agent is unhappy we discard this movement and check new vacant 
                unhappy=0

                x=vacants_try[rand_vacant,0]
                y=vacants_try[rand_vacant,1]
                sign=M[x,y]

                Ni,Nd=0,0

                for k in range(x-limit,x+limit+1):
                    for l in range(y-limit,y+limit+1):
                        if not ((x==k and y==l) or (k<0 or k>L-1 or l<0 or l>L-1)):
                            if M[k,l]!=0:
                                if sign==M[k,l]:
                                    Ni+=1
                                else:
                                    Nd+=1
                                #endif
                            #endif
                        #endif
                    #end for
                #end for

                if Ni+Nd!=0:
                    if Nd/(Nd+Ni)>tau:
                        unhappy=1
                    #endif
                #endif

                if unhappy==0:
                    T_f=T_f+1
                    move_agent=1
                    break
                else:
                    #the agent cannot move, we go to the following vacant

                    #return to original form
                    M[uh_list[rand_uh,0],uh_list[rand_uh,1]]=M[vacants_try[rand_vacant,0],vacants_try[rand_vacant,1]]
                    M[vacants_try[rand_vacant,0],vacants_try[rand_vacant,1]]=0

                    vacants_try[rand_vacant,:]=vacants_try[N_p_v-1,:]
                    vacants_try[N_p_v-1,:]=[0,0]
                    N_p_v=N_p_v-1
                    if N_p_v==0:
                        break
                    #endif
                #endif

            #end for
            #we check if the agent has moved
            if move_agent==1:
                break
            else:
                #the agent cannot move, we go to the following agent
                uh_list[rand_uh,:]=uh_list[N_uh-1,:]
                uh_list[N_uh-1,:]=[0,0]
                N_uh=N_uh-1
                if N_uh==0:
                    break
                #endif
            #endif
        #end for
    #end for

    G_out,H_out,S_out,vacants_out,uh_list_out,N_uh_try=G_function(M,tau,alpha,L,rho_0)
    S_f=S_out
    H_f=H_out

    return S_f,H_f,T_f

    

#greedy algorithm
@njit
def our_model_2(M_i, tau, alpha, L, rho_0):
    """Executes the greedy algorithm and returns a tupple with the final
    configuration, a tupple with (T,h,G,H,S) from the G_function
    applied to the final configuration and the number of iterations."""

    M=M_i.copy()

    N_v=int(round((rho_0)*L**2,0))
    G_list=np.zeros(N_v)
    N_uh=1 #different from zero to enter the loop
    T_f=0 #number of movements
        #initial state
    Ms,Ni,Ndif,Nh=systems_state(M_i,tau,L,rho_0)
    #number of agents 
    N_a=int(round((1-rho_0)*L**2,0))


    #algorithm
    for g in range(1000000):
        if g==999999:
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
            #changes in old neighbourhood (removing the agent but not adding it)
            i_rand=uh_list[rand_uh,0]
            j_rand=uh_list[rand_uh,1]
            limit=1 #Moore Neighbourhood
            M_mod=Ms.copy()
            d_h=0
            d_Ni=0
            d_Ndif=0
            sign=M[i_rand,j_rand]
            for k in range(i_rand-limit,i_rand+limit+1):
                for l in range(j_rand-limit,j_rand+limit+1):
                    if not ((i_rand==k and j_rand==l) or (k<0 or k>L-1 or l<0 or l>L-1)):
                        if M[k,l]!=0:
                            if sign==M[k,l]:
                                M_mod[k,l,0]-=1
                                d_Ni-=1
                            else:
                                M_mod[k,l,1]-=1
                                d_Ndif-=1
                            #endif
                        #endif
                        if (M_mod[k,l,0]+M_mod[k,l,1])!=0:
                            if M_mod[k,l,1]/(M_mod[k,l,0]+M_mod[k,l,1])<=tau:
                                M_mod[k,l,2]=1
                            else:
                                M_mod[k,l,2]=0
                            #endif
                        else:
                            M_mod[k,l,2]=1
                        #endif
                        if M_mod[k,l,2]>Ms[k,l,2]:
                            d_h+=1
                        elif M_mod[k,l,2]<Ms[k,l,2]:
                            d_h-=1
                        #endif
                    #endif
                #end for
            #end for
            #we look for the most suitable movement
            for j in range(N_v):
                #changes in new neighbourhood (adding the agent to its new position)

                #movement
                M[vacants[j,0],vacants[j,1]]=M[uh_list[rand_uh,0],uh_list[rand_uh,1]]
                M[uh_list[rand_uh,0],uh_list[rand_uh,1]]=0

                d_Ni_2,d_Ndif_2,d_h_2,happy=movement(vacants[j,:],tau,L,rho_0,M,M_mod)

                #G calculus
                if happy==0:
                    G_out=10.
                else:
                    new_Ni=Ni+d_Ni+d_Ni_2
                    new_Ndif=Ndif+d_Ndif+d_Ndif_2
                    new_Nh=Nh+d_h+d_h_2
                    if (new_Ni+new_Ndif)!=0:
                        G_out=alpha*new_Ni/(new_Ni+new_Ndif)+(1.-alpha)*new_Nh/N_a
                    else:
                        G_out=(1.-alpha)*new_Nh/N_a
                    #endif
                #endif

                G_list[j]=G_out

                #return to original form
                M[uh_list[rand_uh,0],uh_list[rand_uh,1]]=M[vacants[j,0],vacants[j,1]]
                M[vacants[j,0],vacants[j,1]]=0
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
                #we have to update Ms
                Ms,Ni,Ndif,Nh=systems_state(M,tau,L,rho_0)
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
#Simulation: greedy algorithm
#------------------------------------------------------------------------------------------------

print("Simulation greedy")

# Parameters
tau=0.5
L=20

e=10**(-5)

a1=np.array([0,0.001])
a2=np.arange(0.1,0.4,0.1)
a3=np.arange(0.4,0.6,0.05)
a4=np.arange(0.6,1+e,0.025)
alpha_values=np.concatenate((a1,a2,a3,a4))
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
seed_values=np.arange(0,len(rho_0_values)*len(alpha_values)*N_sim,N_sim) #first 500
#seed_values=seed_values+len(rho_0_values)*len(alpha_values)*N_sim #second 500
print("Start simulations")
for i_r in range(np.size(rho_0_values)):
    rho_0=rho_0_values[i_r]
    for j_a in range(np.size(alpha_values)):
        alpha=alpha_values[j_a]
        data=np.zeros((N_sim,3),dtype="float64") #change 3 to 6 if you want to also simulate on fortran

        for i in range(N_sim):

            np.random.seed(seed_values[counter]+i)

            # Initial Condition
            M_i=initial_matrix(L,rho_0,0.5)

            """fD_M=open(directory_m+"matrix.dat", "w")
                                                np.savetxt(fD_M, M_i, fmt="%d")           
                                                fD_M.flush()
                                                os.fsync(fD_M.fileno())
                                                fD_M.close() #only for fortran simulation"""

            # Python Simulation
            final_state=our_model_2(M_i,tau,alpha,L,rho_0)
            data[i][:3]=final_state[:]

            # Fortran Simulation
                                    
            """if alpha!=0:
                cmd = ["./schelling.exe", str(L), str(alpha)+"d0", str(tau)+"d0", str(rho_0)+"d0", str(counter)]
            else:
                cmd = ["./schelling.exe", str(L), "0.d0", str(tau)+"d0", str(rho_0)+"d0", str(counter)]

            result = subprocess.run(cmd)

            results=np.loadtxt("../data_results_fortran/results_simulation.dat"
                               ,usecols=[-3,-2,-1])

            data[i][3:]=results[:]"""

        counter+=1

        #save the data
        name=directory_s+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L)+".dat"
        np.savetxt(name,data)


print("End simulations")

#------------------------------------------------------------------------------------------------
#Simulation: classic Schelling
#------------------------------------------------------------------------------------------------

print("Simulation classic")

# Parameters
tau=0.5
L=20

e=10**(-5)


rho_0_values_c=np.arange(0.1,0.5,0.1)

N_sim=500

# Saving directories
from os.path import exists
from os import makedirs
    
directory_c="../data_classic/"
if not exists(directory_c):
    makedirs(directory_c)


counter=0
seed_values=np.arange(0,len(rho_0_values_c)*N_sim,N_sim)
print("Start simulations")
for i_r in range(np.size(rho_0_values_c)):
    rho_0=rho_0_values_c[i_r]
    print(rho_0)

    data_c=np.zeros((N_sim,3),dtype="float64") 

    for i in range(N_sim):

        np.random.seed(seed_values[counter]+i)

        # Initial Condition
        M_i=initial_matrix(L,rho_0,0.5)

        # Python Simulation
        final_state=classic_schelling_2(M_i,tau,L,rho_0)
        data_c[i][:]=final_state[:]

    #end for

    counter+=1

    #save the data
    name=directory_c+"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L)+".dat"
    np.savetxt(name,data_c)

#end for

print("End simulations")