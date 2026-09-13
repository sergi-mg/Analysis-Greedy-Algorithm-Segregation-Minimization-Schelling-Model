# Author: Sergi Martínez Galindo

#------------------------------------------------------------------------------------------------
#Libraries
#------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from os.path import exists
from os import makedirs


#------------------------------------------------------------------------------------------------
#Functions
#------------------------------------------------------------------------------------------------

@njit
def statistics(x):
    """Returns mean and its uncertainty of a 1D vector,
    computed over N_i simulations. Returns the mean value
    and its uncertainty (95%)."""
    
    #mean and standard deviation
    sumx=np.sum(x)
    sumx2=np.sum(x**2)
    N_i=x.shape[0]
    xmed=sumx/N_i
    x2med=sumx2/N_i
    s2=(N_i*(x2med-xmed**2))/(N_i-1)
    s=s2**0.5
    
    #standard deviation of the mean (CLT)
    var=s2/N_i
    if var<0:
        var=0
    sigma=(var)**0.5
    unc=1.96*sigma
    return xmed,unc

@njit
def xifres(values,uncertainties,exp_max,exp_min):
    """Rounds the values and their uncertainties (both 1D arrays) with the
    adecuate number of significant figures. Values and uncertanties
    are 1D matrices and exp_max and exp_min are the power (x) of 10^x
    corresponding to the maximum and minimum order of the uncertainties,
    if you get 0's in the output try exppanding the exponent range."""
    
    values_c=np.zeros_like(values)
    uncertainties_c=np.zeros_like(uncertainties)
    for i in range(np.size(uncertainties)):
        for k in range(exp_max,exp_min,-1):
            if uncertainties[i]==0:
                values_c[i]=values[i]
                uncertainties_c[i]=uncertainties[i]
            if uncertainties[i]>=1.95*np.power(10.0, k):
                values_c[i]=round(values[i],-k)
                uncertainties_c[i]=round(uncertainties[i],-k)
                break
    return values_c,uncertainties_c

@njit
def calc_reg(x,y):
    """Returns the parameters of a linear regression of y(x).
    y and x are numpy 1D-arrays."""
    #y=mx+b
    N=np.size(x)
    #mean values
    xmed=np.sum(x)/N
    x2med=np.sum(x**2)/N
    ymed=np.sum(y)/N
    y2med=np.sum(y**2)/N
    xymed=np.sum(x*y)/N
    #sigma's
    sigx2=x2med-xmed**2
    sigy2=y2med-ymed**2
    sigxy=xymed-xmed*ymed
    #regression parameters
    m=sigxy/sigx2
    b=ymed-m*xmed
    r=sigxy/((sigx2*sigy2)**0.5)
    #uncertainties
    dyreg=((sigy2*(1-r**2)*N)/(N-2))**0.5
    dm=dyreg/(N*sigx2)**0.5
    db=dyreg*(x2med/(N*sigx2))**0.5

    return m,dm,b,db,r


def statistics_matrix(rho_val,alpha_val,N_sim,program):
    """Reads the files for the rho and alpha (1D arrays) indicated and returns
    the mean and its uncertainty of the three variables S,H,steps for each
    pair rho-alpha: M[alpha,rho,variable] for mean and uncertainty, two components
    of the tuple. It also returns the correlation coefficient r of S(steps)
    for each pair (alpha,rho)."""
    import numpy as np
    N_alpha=np.size(alpha_val)
    N_rho=np.size(rho_val)
    mean=np.zeros((N_alpha,N_rho,3),dtype="float64")
    uncertainty=np.zeros((N_alpha,N_rho,3),dtype="float64")
    correlation=np.zeros((N_alpha,N_rho),dtype="float64")
    if program=="Python":
        mod=0
    elif program=="Fortran":
        mod=3
    else:
        print("Program should be either Python or Fortran")
    #endif
    for i in range(N_alpha):
        alpha=alpha_val[i]
        for j in range(N_rho):
            rho_0=rho_val[j]
            #reading the file
            name="../data/"+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+".dat"
            read_M=np.loadtxt(fname=name,dtype="float64")
            #variables studied
            S=read_M[:,0+mod]
            H=read_M[:,1+mod]
            steps=read_M[:,2+mod]
            #steps-segregation correlation
            (m,dm,b,db,r)=calc_reg(S, steps)
            correlation[i,j]=r
            #mean and uncertainty calculus
            S_stats=statistics(S)
            H_stats=statistics(H)
            steps_stats=statistics(steps)
            #saving the results
            mean[i,j,0]=S_stats[0]
            mean[i,j,1]=H_stats[0]
            mean[i,j,2]=steps_stats[0]

            uncertainty[i,j,0]=S_stats[1]
            uncertainty[i,j,1]=H_stats[1]
            uncertainty[i,j,2]=steps_stats[1]

    return mean,uncertainty,correlation


def get_edges(values):
        edges=np.empty(len(values)+1)
        edges[1:-1]=(values[:-1]+values[1:])/2
        edges[0]=values[0]-(values[1]-values[0])/2
        edges[-1]=values[-1]+(values[-1]-values[-2])/2
        return edges


#------------------------------------------------------------------------------------------------
#Reading the data
#------------------------------------------------------------------------------------------------
e=0.00001

a1=np.array([0,0.001])
a2=np.arange(0.05,0.4,0.05)
a3=np.arange(0.4,1+e,0.025)
alpha_values=np.concatenate((a1,a2,a3))
alpha_values=np.unique(alpha_values)

r1=np.arange(0.01,0.15,0.01)
r2=np.arange(0.15,0.9+e,0.05)
rho_0_values=np.concatenate((r1,r2))
rho_0_values=np.unique(rho_0_values)

N_sim=100

#Python results
R_p=statistics_matrix(rho_0_values,alpha_values,N_sim,"Python")

results_p=np.zeros((len(alpha_values),len(rho_0_values),3))
d_results_p=np.zeros((len(alpha_values),len(rho_0_values),3))
for v in range (3):
    for i in range(np.size(rho_0_values)):
        results_p[:,i,v],d_results_p[:,i,v]=xifres(R_p[0][:,i,v],R_p[1][:,i,v],10,-10)

#Fortran results
R_f=statistics_matrix(rho_0_values,alpha_values,N_sim,"Fortran")
results_f=np.zeros((len(alpha_values),len(rho_0_values),3))
d_results_f=np.zeros((len(alpha_values),len(rho_0_values),3))
for v in range (3):
    for i in range(np.size(rho_0_values)):
        results_f[:,i,v],d_results_f[:,i,v]=xifres(R_f[0][:,i,v],R_f[1][:,i,v],10,-10)


#------------------------------------------------------------------------------------------------
#Plots
#------------------------------------------------------------------------------------------------        
directory_s="../plots/"
if not exists(directory_s):
    makedirs(directory_s)

alpha_edges=get_edges(alpha_values)
rho_edges=get_edges(rho_0_values)

titles=[r"$\mathcal{S}$",r"$\mathcal{H}$",r"$\mathcal{T}$"]
variables=["S","H","T"]

#Python

for v in range(3):

    Z=results_p[:,:,v]

    plt.figure(figsize=(10,6))
    plt.pcolormesh(alpha_edges,rho_edges,Z.T,cmap='viridis')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\rho_0$')
    cbar=plt.colorbar()
    cbar.set_label(titles[v])
    name=variables[v]
    plt.savefig(directory_s+name+"_python.pdf",bbox_inches="tight")
    plt.close()

#Fortran

for v in range(3):

    Z=results_f[:,:,v]

    plt.figure(figsize=(10,6))
    plt.pcolormesh(alpha_edges,rho_edges,Z.T,cmap='viridis')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\rho_0$')
    cbar=plt.colorbar()
    cbar.set_label(titles[v])
    name=variables[v]
    plt.savefig(directory_s+name+"_fortran.pdf",bbox_inches="tight")
    plt.close()

