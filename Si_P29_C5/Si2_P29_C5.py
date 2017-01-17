# -*- coding: utf-8 -*-
"""
Note- returns ReSampledMatrix, not original sampledmatrix

@author: dfs1
"""

import numpy as np
import CDSAXSfunctions as CD
from multiprocessing import Pool
import time


Intensity=np.loadtxt('Si2_P29_C5_Int.txt')
Qx = np.loadtxt('Si2_P29_C5_Qx.txt')
Qz = np.loadtxt('Si2_P29_C5_Qz.txt')


Trapnumber = 3

DW = 1.69
I0 = 0.2
Bk = 1.2
Pitch = 83.7
FillH = 65;

SLD2=1.2
SPAR=np.zeros([4]) 
SPAR[0]=DW; SPAR[1]=I0; SPAR[2]=Bk;SPAR[3]=Pitch;

TPAR=np.zeros([Trapnumber+1,3])

TPAR[0,0]=15; TPAR[0,1]=60; TPAR[0,2]=1;
TPAR[1,0]=10; TPAR[1,1]=8; TPAR[1,2]=1;
TPAR[2,0]=8; TPAR[2,1]=50; TPAR[2,2]=1;
TPAR[3,0]=5; 

X1 = 18
X2 = 21
(Coord,FillT)= CD.IMECCoordAssign(TPAR,Trapnumber,X1,X2,FillH,SLD2,Pitch)

(FITPAR,FITPARLB,FITPARUB)=CD.IMEC_PBA(TPAR,SPAR,X1,X2,FillH,Trapnumber)

MCPAR=np.zeros([7])
MCPAR[0] = 24 # Chainnumber
MCPAR[1] = len(FITPAR)
MCPAR[2] = 150000 #stepnumber
MCPAR[3] = 0 #randomchains
MCPAR[4] = 1000 # Resampleinterval
MCPAR[5] = 150 # stepbase
MCPAR[6] = 200 # steplength

MCMCInitial=np.load('Si2_P29_C5_Init1.npy')
FITPARLB=np.load('Si2_P29_C5_FITPARLB1.npy')
FITPARUB=np.load('Si2_P29_C5_FITPARUB1.npy')



MCMC_List=[0]*int(MCPAR[0])
for i in range(int(MCPAR[0])):
    MCMC_List[i]=MCMCInitial[i,:]
    
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
                Temp[p]=Temp[p]-Move[p]
            elif Temp[p] > FITPARUB[p]:
                Temp[p]=Temp[p]+Move[p]
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
        if SampledMatrix[i,19] != SampledMatrix[i-1,19]:
            AcceptanceNumber=AcceptanceNumber+1
    AcceptanceProbability=AcceptanceNumber/Acceptancetotal
    print(AcceptanceProbability)
    ReSampledMatrix=np.zeros([int(MCPAR[2])/int(MCPAR[4]),len(SampledMatrix[1,:])])

    c=-1
    for i in np.arange(0,len(SampledMatrix[:,1]),MCPAR[4]):
        c=c+1
        ReSampledMatrix[c,:]=SampledMatrix[i,:]
    return (ReSampledMatrix)
    
def SimInt_IMEC(FITPAR):
    TPAR=np.reshape(FITPAR[0:(Trapnumber+1)*3],(Trapnumber+1,3))
    X1=FITPAR[Trapnumber*3+3]
    X2=FITPAR[Trapnumber*3+4]
    FillH=FITPAR[Trapnumber*3+5]
    SPAR=FITPAR[Trapnumber*3+6:Trapnumber*5+10]
    
    
    
    (Coord,FillT)= CD.IMECCoordAssign(TPAR,Trapnumber,X1,X2,FillH,SLD2,Pitch)
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



start_time = time.perf_counter()
if __name__ =='__main__':  
    pool = Pool(processes=24)
    F=pool.map(MCMC_IMEC,MCMC_List)
    F=tuple(F)
    np.save('Si2_P29_C5_Fit1',F)
    end_time=time.perf_counter()   
    print(end_time-start_time)    

