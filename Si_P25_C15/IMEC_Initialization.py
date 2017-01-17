# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:03:39 2016

@author: dfs1
"""

import numpy as np
import CDSAXSfunctions as CD
import CDplot as CDp
import matplotlib.pyplot as plt
import time
Intensity=np.loadtxt('Si2_P21_C19_Int.txt')
Qx = np.loadtxt('Si2_P21_C19_Qx.txt')
Qz = np.loadtxt('Si2_P21_C19_Qz.txt')


Trapnumber = 3
DW = 2
I0 = .0005
Bk =1.1
Pitch = 83.7
FillH = 65;

SLD2=1.05
SPAR=np.zeros([4]) 
SPAR[0]=DW; SPAR[1]=I0; SPAR[2]=Bk;SPAR[3]=SLD2;

TPAR=np.zeros([Trapnumber+1,3])

TPAR[0,0]=15; TPAR[0,1]=60; TPAR[0,2]=1;
TPAR[1,0]=10; TPAR[1,1]=8; TPAR[1,2]=1;
TPAR[2,0]=8; TPAR[2,1]=50; TPAR[2,2]=1;
TPAR[3,0]=5; 

X1 = 18
X2 = 21
(Coord,FillT)= CD.IMECCoordAssign(TPAR,Trapnumber,X1,X2,FillH,SLD2,Pitch)
(FITPAR,FITPARLB,FITPARUB)=CD.IMEC_PBA(TPAR,SPAR,X1,X2,FillH,Trapnumber)
plt.figure(10)
CDp.plotPSPVP(Coord,Trapnumber)



MCPAR=np.zeros([7])
MCPAR[0] = 24 # Chainnumber
MCPAR[1] = len(FITPAR)
MCPAR[2] = 10 #stepnumber
MCPAR[3] = 18 #randomchains
MCPAR[4] = 1 # Resampleinterval
MCPAR[5] = 40 # stepbase
MCPAR[6] = 200 # steplength

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
    
    
    
SimInt = SimInt_IMEC(FITPAR)
plt.figure(1)
plt.semilogy(Qz[:,3],Intensity[:,3],'.')
plt.semilogy(Qz[:,3],SimInt[:,3])



start_time = time.perf_counter()
MCMCInitialU=MCMCInit_IMEC(FITPAR,FITPARLB,FITPARUB,MCPAR)
end_time=time.perf_counter()   
print(end_time-start_time)
#MCMCInitialU=MCMCInitialU[MCMCInitialU[:,int(MCPAR[1])].argsort(),]

M=MCMCInitialU[0:24,:]

np.save('Si2_P21_C19_Init1',M)
np.save('Si2_P21_C19_FITPARLB1',FITPARLB)
np.save('Si2_P21_C19_FITPARUB1',FITPARUB)


plt.figure(2)
CDp.plotPSPVP(Coord,Trapnumber)
Sim=SimInt_IMEC(MCMCInitialU[1,:])
plt.figure(3)
plt.semilogy(Qz[:,0],Intensity[:,0],'.')
plt.semilogy(Qz[:,0],Sim[:,0])

plt.semilogy(Qz[:,3],Intensity[:,3],'.')
plt.semilogy(Qz[:,3],Sim[:,3])

C=np.sum(CD.Misfit(Intensity,Sim))
plt.figure(4)
CDp.PlotQzCut(Qz,Sim,Intensity,11)