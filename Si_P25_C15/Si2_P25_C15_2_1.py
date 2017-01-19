# -*- coding: utf-8 -*-
"""
Note- returns ReSampledMatrix, not original sampledmatrix

@author: dfs1
"""

import numpy as np
import CDSAXSfunctions as CD
from multiprocessing import Pool
import time


Intensity=np.loadtxt('Si2_P25_C15_Int.txt')
Qx = np.loadtxt('Si2_P25_C15_Qx.txt')
Qz = np.loadtxt('Si2_P25_C15_Qz.txt')


Trapnumber = 2

DW = 1.42
I0 = .0005
Bk =1.16
Pitch = 99.8
FillH = 76.6;


SLD2=1.05
SPAR=np.zeros([4]) 
SPAR[0]=DW; SPAR[1]=I0; SPAR[2]=Bk;SPAR[3]=SLD2;

TPAR=np.zeros([Trapnumber+1,3])


TPAR[0,0]=13.2; TPAR[0,1]=57.6; TPAR[0,2]=1;
TPAR[1,0]=10.8; TPAR[1,1]=62.96; TPAR[1,2]=1;
TPAR[2,0]=7.43; TPAR[2,1]=0; TPAR[2,2]=1;

X1 = 25.3
X2 = 25.2
(Coord,FillT)= CD.IMECCoordAssign(TPAR,Trapnumber,X1,X2,FillH,SLD2,Pitch)

(FITPAR,FITPARLB,FITPARUB)=CD.IMEC_PBA(TPAR,SPAR,X1,X2,FillH,Trapnumber)

MCPAR=np.zeros([7])
MCPAR[0] = 12 # Chainnumber
MCPAR[1] = len(FITPAR)
MCPAR[2] = 150000 #stepnumber
MCPAR[3] = 6 #randomchains
MCPAR[4] = 1000 # Resampleinterval
MCPAR[5] = 100 # stepbase
MCPAR[6] = 300 # steplength


#Function definitions for MCMC optimization
def SimInt_IMEC(FITPAR):
    TPAR=np.ones([Trapnumber+1,3])
    T1=np.reshape(FITPAR[0:(Trapnumber+1)*2],(Trapnumber+1,2))
    TPAR[:,0:2]=T1
    X1=FITPAR[Trapnumber*2+2]
    X2=FITPAR[Trapnumber*2+3]
    FillH=FITPAR[Trapnumber*2+4]
    SPAR=FITPAR[Trapnumber*2+5:Trapnumber*2+9]
    
    
    
    (Coord,FillT)= CD.IMECCoordAssign(TPAR,Trapnumber,X1,X2,FillH,SPAR[3],Pitch)
    F1 = CD.FreeFormTrapezoid(Coord[:,:,0],Qx,Qz,Trapnumber)
    F2 = CD.FreeFormTrapezoid(Coord[:,:,1],Qx,Qz,Trapnumber)
    F3 = CD.FreeFormTrapezoid(Coord[:,:,2],Qx,Qz,Trapnumber)
    F4 = CD.FreeFormTrapezoid(Coord[:,:,3],Qx,Qz,Trapnumber)
    F5 = CD.FreeFormTrapezoid(Coord[:,:,4],Qx,Qz,FillT)
    F6 = CD.FreeFormTrapezoid(Coord[:,:,5],Qx,Qz,FillT)
    F7 = CD.FreeFormTrapezoid(Coord[:,:,6],Qx,Qz,FillT)
    F8 = CD.FreeFormTrapezoid(Coord[:,:,7],Qx,Qz,FillT)    
    Formfactor=(F1+F2+F3+F4+F5+F6+F7+F8)
    M=np.power(np.exp(-1*(np.power(Qx,2)+np.power(Qz,2))*np.power(SPAR[0],2)),0.5)
    Formfactor=Formfactor*M
    Formfactor=abs(Formfactor)
    SimInt = np.power(Formfactor,2)*SPAR[1]+SPAR[2]
    return SimInt


def MCMCInit_IMEC(FITPAR,FITPARLB,FITPARUB,MCPAR):
    
    MCMCInit=np.zeros([int(MCPAR[0]),int(MCPAR[1])+1])
    
    for i in range(int(MCPAR[0])):
        if i <MCPAR[3]: #reversed from matlab code assigns all chains below randomnumber as random chains
            for c in range(int(MCPAR[1])-3):
                MCMCInit[i,c]=FITPARLB[c]+(FITPARUB[c]-FITPARLB[c])*np.random.random_sample()
            MCMCInit[i,int(MCPAR[1])-3:int(MCPAR[1])]=FITPAR[int(MCPAR[1])-3:int(MCPAR[1])]
            SimInt=SimInt_IMEC(MCMCInit[i,:])
            C=np.sum(CD.Misfit(Intensity,SimInt))
            
            MCMCInit[i,int(MCPAR[1])]=C
            
        else:
            MCMCInit[i,0:int(MCPAR[1])]=FITPAR
            SimInt=SimInt_IMEC(MCMCInit[i,:])
            C=np.sum(CD.Misfit(Intensity,SimInt))
            MCMCInit[i,int(MCPAR[1])]=C
           
    return MCMCInit
    
def MCMCInit_IMECUniform(FITPAR,FITPARLB,FITPARUB,MCPAR):
    
    MCMCInit=np.zeros([int(MCPAR[0]),int(MCPAR[1])+1])
    
    for i in range(int(MCPAR[1])-3):
        if FITPARUB[i]==FITPARLB[i]:
            MCMCInit[:,i]=FITPAR[i]
        else:
            A= np.arange(FITPARLB[i],FITPARUB[i]+0.0001,(FITPARUB[i]-FITPARLB[i])/(int(MCPAR[0])-1))
            R=np.random.rand(int(MCPAR[0]))
            ind=R.argsort()
            A=A[ind]
            MCMCInit[:,i]=A
    MCMCInit[:,int(MCPAR[1])-3:int(MCPAR[1])]=FITPAR[int(MCPAR[1])-3:int(MCPAR[1])]        
    
    for i in range(int(MCPAR[0])):
       SimInt=SimInt_IMEC(MCMCInit[i,:])
       C=np.sum(CD.Misfit(Intensity,SimInt))
       MCMCInit[i,int(MCPAR[1])]=C
        
    return MCMCInit
    
def MCMC_IMEC(MCMC_List):
    
    MCMCInit=MCMC_List
    
    L = int(MCPAR[1])
    Stepnumber= int(MCPAR[2])
        
    SampledMatrix=np.zeros([Stepnumber,L+1]) 
    SampledMatrix[0,:]=MCMCInit
    Move = np.zeros([L+1])
    
    ChiPrior = MCMCInit[L]
    for step in np.arange(1,Stepnumber,1): 
        Temp = SampledMatrix[step-1,:].copy()
        for p in range(L-1):
            StepControl = MCPAR[5]+MCPAR[6]*np.random.random_sample()
            Move[p] = (FITPARUB[p]-FITPARLB[p])/StepControl*(np.random.random_sample()-0.5) # need out of bounds check
            Temp[p]=Temp[p]+Move[p]
            if Temp[p] < FITPARLB[p]:
                Temp[p]=FITPARLB[p]+(FITPARUB[p]-FITPARLB[p])/1000
            elif Temp[p] > FITPARUB[p]:
                Temp[p]=FITPARUB[p]-(FITPARUB[p]-FITPARLB[p])/1000
        SimPost=SimInt_IMEC(Temp)
        ChiPost=np.sum(CD.Misfit(Intensity,SimPost))
        if ChiPost < ChiPrior:
            SampledMatrix[step,0:L]=Temp[0:L]
            SampledMatrix[step,L]=ChiPost
            ChiPrior=ChiPost
            
        else:
            MoveProb = np.exp(-0.5*np.power(ChiPost-ChiPrior,2))
            if np.random.random_sample() < MoveProb:
                SampledMatrix[step,0:L]=Temp[0:L]
                SampledMatrix[step,L]=ChiPost
                ChiPrior=ChiPost
            else:
                SampledMatrix[step,:]=SampledMatrix[step-1,:]
    AcceptanceNumber=0;
    Acceptancetotal=len(SampledMatrix[:,1])

    for i in np.arange(1,len(SampledMatrix[:,1]),1):
        if SampledMatrix[i,0] != SampledMatrix[i-1,0]:
            AcceptanceNumber=AcceptanceNumber+1
    AcceptanceProbability=AcceptanceNumber/Acceptancetotal
    print(AcceptanceProbability)
    ReSampledMatrix=np.zeros([int(MCPAR[2])/int(MCPAR[4]),len(SampledMatrix[1,:])])

    c=-1
    for i in np.arange(0,len(SampledMatrix[:,1]),MCPAR[4]):
        c=c+1
        ReSampledMatrix[c,:]=SampledMatrix[i,:]
    return (ReSampledMatrix)


#Initialization and optimization
MCMCInitial=MCMCInit_IMEC(FITPAR,FITPARLB,FITPARUB,MCPAR)

MCMC_List=[0]*int(MCPAR[0])
for i in range(int(MCPAR[0])):
    MCMC_List[i]=MCMCInitial[i,:]
    

    




start_time = time.perf_counter()
if __name__ =='__main__':  
    pool = Pool(processes=24)
    F=pool.map(MCMC_IMEC,MCMC_List)
    F=tuple(F)
    np.save('Si2_P25_C15_Fit2_1',F)
    end_time=time.perf_counter()   
    print(end_time-start_time)    

