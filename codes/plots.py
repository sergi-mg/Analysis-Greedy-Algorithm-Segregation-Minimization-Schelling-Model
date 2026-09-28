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
def xifres_escalar(value,error,exp_max,exp_min):
    for k in range(exp_max,exp_min,-1):
        if error>=1.95*np.power(10.0, k):
            value_c=round(value,-k)
            error_c=round(error,-k)
            break
    return value_c,error_c

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


def statistics_matrix(rho_val,alpha_val,N_sim,program,n_files=1,L_b=False,L_value=0):
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
            if L_b==True:
                name="../data/"+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L_value)+".dat"
            #endif
            read_M=np.loadtxt(fname=name,dtype="float64")
            #variables studied
            S=read_M[:,0+mod]
            H=read_M[:,1+mod]
            steps=read_M[:,2+mod]
            if n_files==2:
                name="../data/"+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_2.dat"
                if L_b==True:
                    name="../data/"+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L_value)+"_2.dat"
                #endif
                read_M=np.loadtxt(fname=name,dtype="float64")
                #variables studied
                S_2=read_M[:,0+mod]
                H_2=read_M[:,1+mod]
                steps_2=read_M[:,2+mod]
                S=np.concatenate((S,S_2))
                H=np.concatenate((H,H_2))
                steps=np.concatenate((steps,steps_2))
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

def raw_data(rho_0,alpha_val,N_sim,index,n_files=1,L_b=False,L_value=0):
    """Reads the files for the indicated rho_0 and returns the
    raw data (N_sim*n_files,N_alpha) of the final state (expected 
    file 3 columns - only python). Input:
        - N: number of nodes.
        - N_sim: number of simulations.
        - rho_0: vacantd ensity value.
        - alpha_val: 1D array (N_alpha)
        - index: integer indicating the variable: 0-S, 1-H, 2-T. 
        - n_files: number of files to read (1 or 2)"""

    
    directory="../data/"
    
    N_alpha=np.size(alpha_val)
    results=np.zeros((N_sim*n_files,N_alpha),dtype="float64")
    
    
    for j in range(N_alpha):
        alpha=alpha_val[j]
        #reading the file
        name="../data/"+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+".dat"
        if L_b==True:
                name="../data/"+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L_value)+".dat"
        read_M=np.loadtxt(fname=name,dtype="float64")
        #variables studied
        R_1=read_M[:,index]

        if n_files==2:
            name="../data/"+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_2.dat"
            if L_b==True:
                    name="../data/"+"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L_value)+"_2.dat"
            read_M=np.loadtxt(fname=name,dtype="float64")
            #variables studied
            R_2=read_M[:,index]
            results[:,j]=np.concatenate((R_1,R_2))
        elif n_files==1:
            results[:,j]=R_1
 
    return results


#------------------------------------------------------------------------------------------------
#L=20, N_sim=1000
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#Reading the data
#------------------------------------------------------------------------------------------------
e=0.00001

a1=np.array([0,0.001])
a2=np.arange(0.1,0.4,0.1)
a3=np.arange(0.4,0.6,0.05)
a4=np.arange(0.6,1+e,0.025)
alpha_values=np.concatenate((a1,a2,a3,a4))
alpha_values=np.unique(alpha_values)

rho_0_values=np.arange(0.05,0.9+e,0.05)

N_sim=500

R_p=statistics_matrix(rho_0_values,alpha_values,N_sim,"Python",n_files=2)

results_p=np.zeros((len(alpha_values),len(rho_0_values),3))
d_results_p=np.zeros((len(alpha_values),len(rho_0_values),3))
for v in range (3):
    for i in range(np.size(rho_0_values)):
        results_p[:,i,v],d_results_p[:,i,v]=xifres(R_p[0][:,i,v],R_p[1][:,i,v],10,-10)


#------------------------------------------------------------------------------------------------
#Heatmaps - S, H, T
#------------------------------------------------------------------------------------------------        
directory_s="../plots/"
if not exists(directory_s):
    makedirs(directory_s)

alpha_edges=get_edges(alpha_values)
rho_edges=get_edges(rho_0_values)

titles=[r"$\mathcal{S}$",r"$\mathcal{H}$",r"$\mathcal{T}$"]
variables=["S","H","T"]

#Python
zmin=[0.7,0.996,0]
zmax=[0.9,1,200]
for v in range(3):

    Z=results_p[:,:,v]

    plt.figure(figsize=(10,6))
    plt.pcolormesh(alpha_edges,rho_edges,Z.T,cmap='viridis',vmin=zmin[v],
    vmax=zmax[v])
    plt.xlabel(r'$\alpha$',fontsize=18)
    plt.ylabel(r'$\rho_0$',fontsize=18)
    cbar=plt.colorbar()
    cbar.set_label(titles[v],fontsize=18)
    name=variables[v]
    #plt.ylim(0,0.9)
    ticks = np.arange(0.1, 1, 0.1)
    plt.yticks(ticks)
    plt.tick_params(axis='both', labelsize=16)
    plt.savefig(directory_s+name+"_heatmap.pdf",bbox_inches="tight")
    plt.close()


#------------------------------------------------------------------------------------------------
#Boxplots - S
#------------------------------------------------------------------------------------------------   

#Boxplots 
N_sim=500
#data
data=[]
rho_values=np.arange(0.1,0.5,0.1)
for i in range(len(rho_values)):
    rho_0=rho_values[i]
    data.append(raw_data(rho_0,alpha_values,N_sim,0,n_files=2))


#data_classic
data_c=[]
N_sim=1000
rho_values=np.arange(0.1,0.5,0.1)
for i in range(len(rho_values)):
    rho_0=rho_values[i]
    #reading the file
    name="../data_classic/"+"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+".dat"
    read_M=np.loadtxt(fname=name,dtype="float64")
    #variables studied
    data_c.append(read_M[:,0])

#limits
y_min=[0.74,0.68,0.65,0.65]
y_max=[0.95,0.95,0.95,0.95]
#plot
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(2, 2, figsize=(16,10))
fig.subplots_adjust(wspace=0.2)
counter=0
for i in range(2):
    for j in range(2):
        rho_0=rho_values[counter]
        results=data[counter]
        ax[i][j].boxplot(
            results,
            positions=alpha_values,
            widths=0.015,
            showfliers=True,
            flierprops=dict(
            marker='x',        # tipo de marcador
            markersize=2,      # tamaño
            linestyle='none'   # sin líneas
        ), medianprops=dict(
            color=cmap(0.67),      # color de la mediana
            linewidth=2      # grosor opcional
        )
        )
        
        
        """means = results.mean(axis=0)
        ax[i][j].plot(alpha_values,means,linestyle='none',marker='o',markersize=2,
                color=cmap(0))"""

        constant_results=data_c[counter]
        p25,p50,p75=np.percentile(constant_results,[25,50,75])
        IRC=p75-p25

        ax[i,j].axhspan(p25-1.5*IRC,p75+1.5*IRC,alpha=0.15,color=cmap(0.2))
        ax[i,j].axhspan(p25,p75,alpha=0.3,color=cmap(0.2))
        ax[i,j].axhline(p50,linestyle='--',color=cmap(0.2))
        
        ax[i][j].set_xlim(-0.02, 1.02)
        #ax[i][j].set_ylim(0.65,0.95)
        ticks = np.arange(0, 1.1, 0.2)

        ax[i][j].set_xticks(ticks)
        ax[i][j].set_ylim(y_min[counter],y_max[counter])
        ax[i][j].tick_params(axis='both', labelsize=16)
        ax[i][j].set_xticklabels([f"{t:.1f}" for t in ticks])
        
        ax[i][j].set_ylabel(r"$\mathcal{S}(\alpha,\rho_0=$"+str(round(rho_0,1))+")",fontsize=18)
        if i!=0:
            ax[i][j].set_xlabel(r"$\alpha$",fontsize=18)

        counter+=1


from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Greedy algorithm'),
Line2D([0],[0],linestyle='--',color=cmap(0.2),label='Classic algorithm')
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .9),ncol=2)
name="S_boxplot"
plt.savefig(directory_s+name+".pdf",bbox_inches="tight")
plt.close()



#------------------------------------------------------------------------------------------------
#Mean Value + 1.96 sigma uncertainty - S
#------------------------------------------------------------------------------------------------   

#data_classic
data_c=[]
N_sim=1000
rho_values=np.arange(0.1,0.5,0.1)
rho_values_heatmaps=np.arange(0.05,0.9+e,0.05)
for i in range(len(rho_values)):
    rho_0=rho_values[i]
    #reading the file
    name="../data_classic/"+"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+".dat"
    read_M=np.loadtxt(fname=name,dtype="float64")
    #variables studied
    data_c.append(read_M[:,0])

#limits
y_min=[0.82,0.76,0.725,0.7]
y_max=[0.875,0.88,0.88,0.9]
#plot
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(2, 2, figsize=(16,10))
fig.subplots_adjust(wspace=0.2)
counter=0
for i in range(2):
    for j in range(2):
        rho_0=rho_values[counter]
        index=1+2*counter
        results=results_p[:,index,0]
        d_results=d_results_p[:,index,0]
        ax[i][j].errorbar(alpha_values,results,xerr=0,yerr=d_results,marker="o",markersize=3.5,linestyle="None",
            color=cmap(0.67))

        constant_results=data_c[counter]
        mean,unc=statistics(constant_results)
        mean,unc=xifres_escalar(mean,unc,2,-6)

        ax[i,j].axhspan(mean-unc,mean+unc,alpha=0.3,color=cmap(0.2))
        ax[i,j].axhline(mean,linestyle='--',color=cmap(0.2))
        
        ax[i][j].set_xlim(-0.02, 1.02)
        #ax[i][j].set_ylim(0.65,0.95)
        ticks = np.arange(0, 1.1, 0.2)

        ax[i][j].set_xticks(ticks)
        ax[i][j].set_ylim(y_min[counter],y_max[counter])
        ax[i][j].tick_params(axis='both', labelsize=16)
        ax[i][j].set_xticklabels([f"{t:.1f}" for t in ticks])
        
        ax[i][j].set_ylabel(r"$\mathcal{S}(\alpha,\rho_0=$"+str(round(rho_0,1))+")",fontsize=18)
        if i!=0:
            ax[i][j].set_xlabel(r"$\alpha$",fontsize=18)

        counter+=1


from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Greedy algorithm'),
Line2D([0],[0],linestyle='--',color=cmap(0.2),label='Classic algorithm')
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .9),ncol=2)
name="S_mean"
plt.savefig(directory_s+name+".pdf",bbox_inches="tight")
plt.close()

#------------------------------------------------------------------------------------------------
#Boxplots - T
#------------------------------------------------------------------------------------------------   

#Boxplots 
N_sim=500
#data
data=[]
rho_values=np.arange(0.1,0.5,0.1)
for i in range(len(rho_values)):
    rho_0=rho_values[i]
    data.append(raw_data(rho_0,alpha_values,N_sim,2,n_files=2))


#data_classic
data_c=[]
N_sim=1000
rho_values=np.arange(0.1,0.5,0.1)
for i in range(len(rho_values)):
    rho_0=rho_values[i]
    #reading the file
    name="../data_classic/"+"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+".dat"
    read_M=np.loadtxt(fname=name,dtype="float64")
    #variables studied
    data_c.append(read_M[:,2])

#limits
y_min=[100,50,50,40]
y_max=[275,275,250,250]
#plot
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(2, 2, figsize=(16,10))
fig.subplots_adjust(wspace=0.2)
counter=0
for i in range(2):
    for j in range(2):
        rho_0=rho_values[counter]
        results=data[counter]
        ax[i][j].boxplot(
            results,
            positions=alpha_values,
            widths=0.015,
            showfliers=True,
            flierprops=dict(
            marker='x',        # tipo de marcador
            markersize=2,      # tamaño
            linestyle='none'   # sin líneas
        ), medianprops=dict(
            color=cmap(0.67),      # color de la mediana
            linewidth=2      # grosor opcional
        )
        )
        
        
        """means = results.mean(axis=0)
        ax[i][j].plot(alpha_values,means,linestyle='none',marker='o',markersize=2,
                color=cmap(0))"""

        constant_results=data_c[counter]
        p25,p50,p75=np.percentile(constant_results,[25,50,75])
        IRC=p75-p25

        ax[i,j].axhspan(p25-1.5*IRC,p75+1.5*IRC,alpha=0.15,color=cmap(0.2))
        ax[i,j].axhspan(p25,p75,alpha=0.3,color=cmap(0.2))
        ax[i,j].axhline(p50,linestyle='--',color=cmap(0.2))
        
        ax[i][j].set_xlim(-0.02, 1.02)
        ax[i][j].set_ylim(y_min[counter],y_max[counter])
        ticks = np.arange(0, 1.1, 0.2)

        ax[i][j].set_xticks(ticks)
        #ax[i][j].set_yticks(np.arange(-1, 1.01, 0.5))
        ax[i][j].tick_params(axis='both', labelsize=16)
        ax[i][j].set_xticklabels([f"{t:.1f}" for t in ticks])
        
        ax[i][j].set_ylabel(r"$\mathcal{T}(\alpha,\rho_0=$"+str(round(rho_0,1))+")",fontsize=18)
        if i!=0:
            ax[i][j].set_xlabel(r"$\alpha$",fontsize=18)

        counter+=1


from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Greedy algorithm'),
Line2D([0],[0],linestyle='--',color=cmap(0.2),label='Classic algorithm')
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .9),ncol=2)
name="T_boxplot"
plt.savefig(directory_s+name+".pdf",bbox_inches="tight")
plt.close()



#------------------------------------------------------------------------------------------------
#Mean Value + 1.96 sigma uncertainty - T
#------------------------------------------------------------------------------------------------   

#data_classic
data_c=[]
N_sim=1000
rho_values=np.arange(0.1,0.5,0.1)
rho_values_heatmaps=np.arange(0.05,0.9+e,0.05)
for i in range(len(rho_values)):
    rho_0=rho_values[i]
    #reading the file
    name="../data_classic/"+"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+".dat"
    read_M=np.loadtxt(fname=name,dtype="float64")
    #variables studied
    data_c.append(read_M[:,2])

#limits
y_min=[130,100,80,60]
y_max=[190,200,190,170]
#plot
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(2, 2, figsize=(16,10))
fig.subplots_adjust(wspace=0.2)
counter=0
for i in range(2):
    for j in range(2):
        rho_0=rho_values[counter]
        index=1+2*counter
        results=results_p[:,index,2]
        d_results=d_results_p[:,index,2]
        ax[i][j].errorbar(alpha_values,results,xerr=0,yerr=d_results,marker="o",markersize=3.5,linestyle="None",
            color=cmap(0.67))

        constant_results=data_c[counter]
        mean,unc=statistics(constant_results)
        mean,unc=xifres_escalar(mean,unc,2,-6)

        ax[i,j].axhspan(mean-unc,mean+unc,alpha=0.3,color=cmap(0.2))
        ax[i,j].axhline(mean,linestyle='--',color=cmap(0.2))
        
        ax[i][j].set_xlim(-0.02, 1.02)
        ax[i][j].set_ylim(y_min[counter],y_max[counter])
        ticks = np.arange(0, 1.1, 0.2)

        ax[i][j].set_xticks(ticks)
        #ax[i][j].set_yticks(np.arange(-1, 1.01, 0.5))
        ax[i][j].tick_params(axis='both', labelsize=16)
        ax[i][j].set_xticklabels([f"{t:.1f}" for t in ticks])
        
        ax[i][j].set_ylabel(r"$\mathcal{T}(\alpha,\rho_0=$"+str(round(rho_0,1))+")",fontsize=18)
        if i!=0:
            ax[i][j].set_xlabel(r"$\alpha$",fontsize=18)

        counter+=1


from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Greedy algorithm'),
Line2D([0],[0],linestyle='--',color=cmap(0.2),label='Classic algorithm')
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .9),ncol=2)
name="T_mean"
plt.savefig(directory_s+name+".pdf",bbox_inches="tight")
plt.close()


#------------------------------------------------------------------------------------------------
#L={20,40}, N_sim=500
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#Reading the data
#------------------------------------------------------------------------------------------------
e=0.00001

a1=np.array([0,0.001])
a2=np.arange(0.1,0.4,0.1)
a3=np.arange(0.4,0.6,0.05)
a4=np.arange(0.6,1+e,0.025)
alpha_values=np.concatenate((a1,a2,a3,a4))
alpha_values=np.unique(alpha_values)

rho_0_values=np.arange(0.05,0.9+e,0.05)

L_values=[20,40]

N_sim=500

results_p=np.zeros((len(alpha_values),len(rho_0_values),3,2))
d_results_p=np.zeros((len(alpha_values),len(rho_0_values),3,2))

j=0
for L in L_values:
    R_p=statistics_matrix(rho_0_values,alpha_values,N_sim,"Python",n_files=1,L_b=True,L_value=L)


    for v in range (3):
        for i in range(np.size(rho_0_values)):
            results_p[:,i,v,j],d_results_p[:,i,v,j]=xifres(R_p[0][:,i,v],R_p[1][:,i,v],10,-10)

    j+=1


#------------------------------------------------------------------------------------------------
#Heatmaps - S, H, T
#------------------------------------------------------------------------------------------------        
directory_s="../plots/"
if not exists(directory_s):
    makedirs(directory_s)

alpha_edges=get_edges(alpha_values)
rho_edges=get_edges(rho_0_values)

titles=[r"$\mathcal{S}$",r"$\mathcal{H}$",r"$\mathcal{T}$"]
variables=["S","H","T"]

#Python
zmin=[0.7,0.996,0]
zmax=[0.9,1,200]
subtitles=["$L=20$","$L=40$"]
for v in range(3):

    Z=results_p[:,:,v]

    fig,ax=plt.subplots(1,2,figsize=(20,6))

    for j in range(2):
        Z=results_p[:,:,v,j]

        im=ax[j].pcolormesh(
            alpha_edges,rho_edges,Z.T,
            cmap='viridis',
            vmin=zmin[v],
            vmax=zmax[v]
        )

        ax[j].set_xlabel(r'$\alpha$',fontsize=18)
        if j==0:
            ax[j].set_ylabel(r'$\rho_0$',fontsize=18)
        else:
            ax[j].set_ylabel('')

        ticks=np.arange(0.1,1,0.1)
        ax[j].set_yticks(ticks)
        ax[j].tick_params(axis='both',labelsize=16)

        ax[j].set_title(subtitles[j],fontsize=18,y=-0.25)

    cbar=fig.colorbar(im,ax=ax)
    cbar.set_label(titles[v],fontsize=18)

    plt.savefig(directory_s+variables[v]+"_heatmap_L.pdf",bbox_inches="tight")
    plt.close()


#------------------------------------------------------------------------------------------------
#Boxplots - S
#------------------------------------------------------------------------------------------------   

#Boxplots 
N_sim=500
#data
data=[]
L_values=[20,40]
for i in range(len(L_values)):
    L=L_values[i]
    rho_0=0.2
    data.append(raw_data(rho_0,alpha_values,N_sim,0,n_files=1,L_b=True,L_value=L))


#data_classic
data_c=[]
N_sim=500
for i in range(len(L_values)):
    L=L_values[i]
    #reading the file
    name="../data_classic/"+"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L)+".dat"
    read_M=np.loadtxt(fname=name,dtype="float64")
    #variables studied
    data_c.append(read_M[:,0])

#limits
y_min=[0.68,0.68]
y_max=[0.95,0.95]
#plot
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(1,2, figsize=(16,5))
fig.subplots_adjust(wspace=0.2)
counter=0
for j in range(2):
    L=L_values[j]
    results=data[j]
    ax[j].boxplot(
        results,
        positions=alpha_values,
        widths=0.015,
        showfliers=True,
        flierprops=dict(
        marker='x',        # tipo de marcador
        markersize=2,      # tamaño
        linestyle='none'   # sin líneas
    ), medianprops=dict(
        color=cmap(0.67),      # color de la mediana
        linewidth=2      # grosor opcional
    )
    )
    
    
    """means = results.mean(axis=0)
    ax[j].plot(alpha_values,means,linestyle='none',marker='o',markersize=2,
            color=cmap(0))"""

    constant_results=data_c[j]
    p25,p50,p75=np.percentile(constant_results,[25,50,75])
    IRC=p75-p25

    ax[j].axhspan(p25-1.5*IRC,p75+1.5*IRC,alpha=0.15,color=cmap(0.2))
    ax[j].axhspan(p25,p75,alpha=0.3,color=cmap(0.2))
    ax[j].axhline(p50,linestyle='--',color=cmap(0.2))
    
    ax[j].set_xlim(-0.02, 1.02)
    #ax[j].set_ylim(0.65,0.95)
    ticks = np.arange(0, 1.1, 0.2)

    ax[j].set_xticks(ticks)
    #ax[j].set_ylim(y_min[j],y_max[j])
    ax[j].tick_params(axis='both', labelsize=16)
    ax[j].set_xticklabels([f"{t:.1f}" for t in ticks])
    if j==0:
        ax[j].set_ylabel(r"$\mathcal{S}(\alpha,\rho_0)$",fontsize=18)
    ax[j].set_xlabel(r"$\alpha$",fontsize=18)
    ax[j].set_title(subtitles[j],fontsize=18,y=-0.25)



from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Greedy algorithm'),
Line2D([0],[0],linestyle='--',color=cmap(0.2),label='Classic algorithm')
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .9),ncol=2)
name="S_boxplot"
plt.savefig(directory_s+name+"_L.pdf",bbox_inches="tight")
plt.close()



#------------------------------------------------------------------------------------------------
#Mean Value + 1.96 sigma uncertainty - S
#------------------------------------------------------------------------------------------------   

#data_classic
data_c=[]
N_sim=500
L_values=[20,40]
for i in range(len(L_values)):
    rho_0=0.2
    L=L_values[i]
    #reading the file
    name="../data_classic/"+"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L)+".dat"
    read_M=np.loadtxt(fname=name,dtype="float64")
    #variables studied
    data_c.append(read_M[:,0])

#limits
y_min=[0.76,0.76]
y_max=[0.88,0.88]
#plot
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(1,2, figsize=(16,5))
fig.subplots_adjust(wspace=0.2)

for j in range(2):
    rho_0=0.2
    L=L_values[j]
    index=3
    results=results_p[:,index,0,j]
    d_results=d_results_p[:,index,0,j]
    ax[j].errorbar(alpha_values,results,xerr=0,yerr=d_results,marker="o",markersize=3.5,linestyle="None",
        color=cmap(0.67))

    constant_results=data_c[j]
    mean,unc=statistics(constant_results)
    mean,unc=xifres_escalar(mean,unc,2,-6)

    ax[j].axhspan(mean-unc,mean+unc,alpha=0.3,color=cmap(0.2))
    ax[j].axhline(mean,linestyle='--',color=cmap(0.2))
    
    ax[j].set_xlim(-0.02, 1.02)
    #ax[j].set_ylim(0.65,0.95)
    ticks = np.arange(0, 1.1, 0.2)

    ax[j].set_xticks(ticks)
    ax[j].set_ylim(y_min[j],y_max[j])
    ax[j].tick_params(axis='both', labelsize=16)
    ax[j].set_xticklabels([f"{t:.1f}" for t in ticks])
    
    if j==0:
        ax[j].set_ylabel(r"$\mathcal{S}(\alpha,\rho_0)$",fontsize=18)
    ax[j].set_xlabel(r"$\alpha$",fontsize=18)
    ax[j].set_title(subtitles[j],fontsize=18,y=-0.25)



from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Greedy algorithm'),
Line2D([0],[0],linestyle='--',color=cmap(0.2),label='Classic algorithm')
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .9),ncol=2)
name="S_mean"
plt.savefig(directory_s+name+"_L.pdf",bbox_inches="tight")
plt.close()

#------------------------------------------------------------------------------------------------
#Boxplots - T
#------------------------------------------------------------------------------------------------   

#Boxplots 
N_sim=500
#data
data=[]
L_values=[20,40]
for i in range(len(L_values)):
    L=L_values[i]
    rho_0=0.2
    data.append(raw_data(rho_0,alpha_values,N_sim,2,n_files=1,L_b=True,L_value=L))


#data_classic
data_c=[]
N_sim=500
for i in range(len(L_values)):
    L=L_values[i]
    #reading the file
    name="../data_classic/"+"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L)+".dat"
    read_M=np.loadtxt(fname=name,dtype="float64")
    #variables studied
    data_c.append(read_M[:,2])

#limits
y_min=[50,50]
y_max=[275,275]
#plot
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(1,2, figsize=(16,5))
fig.subplots_adjust(wspace=0.2)
counter=0
for j in range(2):
    L=L_values[j]
    results=data[j]
    ax[j].boxplot(
        results,
        positions=alpha_values,
        widths=0.015,
        showfliers=True,
        flierprops=dict(
        marker='x',        # tipo de marcador
        markersize=2,      # tamaño
        linestyle='none'   # sin líneas
    ), medianprops=dict(
        color=cmap(0.67),      # color de la mediana
        linewidth=2      # grosor opcional
    )
    )
    
    
    """means = results.mean(axis=0)
    ax[j].plot(alpha_values,means,linestyle='none',marker='o',markersize=2,
            color=cmap(0))"""

    constant_results=data_c[j]
    p25,p50,p75=np.percentile(constant_results,[25,50,75])
    IRC=p75-p25

    ax[j].axhspan(p25-1.5*IRC,p75+1.5*IRC,alpha=0.15,color=cmap(0.2))
    ax[j].axhspan(p25,p75,alpha=0.3,color=cmap(0.2))
    ax[j].axhline(p50,linestyle='--',color=cmap(0.2))
    
    ax[j].set_xlim(-0.02, 1.02)
    #ax[j].set_ylim(0.65,0.95)
    ticks = np.arange(0, 1.1, 0.2)

    ax[j].set_xticks(ticks)
    ax[j].set_ylim(y_min[j],y_max[j])
    ax[j].tick_params(axis='both', labelsize=16)
    ax[j].set_xticklabels([f"{t:.1f}" for t in ticks])
    if j==0:
        ax[j].set_ylabel(r"$\mathcal{T}(\alpha,\rho_0)$",fontsize=18)
    ax[j].set_xlabel(r"$\alpha$",fontsize=18)
    ax[j].set_title(subtitles[j],fontsize=18,y=-0.25)



from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Greedy algorithm'),
Line2D([0],[0],linestyle='--',color=cmap(0.2),label='Classic algorithm')
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .9),ncol=2)
name="T_boxplot"
plt.savefig(directory_s+name+"_L.pdf",bbox_inches="tight")
plt.close()




#------------------------------------------------------------------------------------------------
#Mean Value + 1.96 sigma uncertainty - T
#------------------------------------------------------------------------------------------------   

#data_classic 100 200
#data_classic
data_c=[]
N_sim=500
L_values=[20,40]
for i in range(len(L_values)):
    rho_0=0.2
    L=L_values[i]
    #reading the file
    name="../data_classic/"+"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L)+".dat"
    read_M=np.loadtxt(fname=name,dtype="float64")
    #variables studied
    data_c.append(read_M[:,0])

#limits
y_min=[100,100]
y_max=[200,200]
#plot
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(1,2, figsize=(16,5))
fig.subplots_adjust(wspace=0.2)

for j in range(2):
    rho_0=0.2
    L=L_values[j]
    index=3
    results=results_p[:,index,0,j]
    d_results=d_results_p[:,index,0,j]
    ax[j].errorbar(alpha_values,results,xerr=0,yerr=d_results,marker="o",markersize=3.5,linestyle="None",
        color=cmap(0.67))

    constant_results=data_c[j]
    mean,unc=statistics(constant_results)
    mean,unc=xifres_escalar(mean,unc,2,-6)

    ax[j].axhspan(mean-unc,mean+unc,alpha=0.3,color=cmap(0.2))
    ax[j].axhline(mean,linestyle='--',color=cmap(0.2))
    
    ax[j].set_xlim(-0.02, 1.02)
    #ax[j].set_ylim(0.65,0.95)
    ticks = np.arange(0, 1.1, 0.2)

    ax[j].set_xticks(ticks)
    ax[j].set_ylim(y_min[j],y_max[j])
    ax[j].tick_params(axis='both', labelsize=16)
    ax[j].set_xticklabels([f"{t:.1f}" for t in ticks])
    
    if j==0:
        ax[j].set_ylabel(r"$\mathcal{T}(\alpha,\rho_0)$",fontsize=18)
    ax[j].set_xlabel(r"$\alpha$",fontsize=18)
    ax[j].set_title(subtitles[j],fontsize=18,y=-0.25)



from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Greedy algorithm'),
Line2D([0],[0],linestyle='--',color=cmap(0.2),label='Classic algorithm')
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .9),ncol=2)
name="T_mean"
plt.savefig(directory_s+name+"_L.pdf",bbox_inches="tight")
plt.close()