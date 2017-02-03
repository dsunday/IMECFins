# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:18:57 2017

@author: dfs1
"""

import numpy as np
import CDSAXSfunctions as CD
import CDplot as CDp
import matplotlib.pyplot as plt

Intensity=np.loadtxt('Si2_P21_C19_Int.txt')
Qx = np.loadtxt('Si2_P21_C19_Qx.txt')
Qz = np.loadtxt('Si2_P21_C19_Qz.txt')

Trapnumber = 3
Pitch = 83.7
SampledMatrix=np.load('Si2_P21_C19_Fit3_1.npy')

#AcceptanceNumber=0;
#Acceptancetotal=len(SampledMatrix[:,1,1])*len(SampledMatrix[1,:,1])
#
#for r in range(len(SampledMatrix[:,1,1])):
#    for i in np.arange(1,len(SampledMatrix[1,:,1]),1):
#        if SampledMatrix[r,i,19] != SampledMatrix[r,i-1,19]:
#            AcceptanceNumber=AcceptanceNumber+1
#AcceptanceProbability=AcceptanceNumber/Acceptancetotal
#
#ReSampledMatrix=np.zeros([24000,len(SampledMatrix[1,1,:])])
#c=-1
#for r in range(len(SampledMatrix[:,1,1])):
#    for i in np.arange(0,len(SampledMatrix[1,:,1]),100):
#     c=c+1
#     ReSampledMatrix[c,:]=SampledMatrix[r,i,:]
#     
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


TOP=SampledMatrix[9,149,:] # sorts the sampled matrix for the best solutions

SimInt1=SimInt_IMEC(TOP)
plt.figure(2)
CDp.PlotQzCut(Qz,SimInt1,Intensity,16)  
ChiPost=np.sum(CD.Misfit(Intensity,SimInt1))
OptTPAR=np.ones([Trapnumber+1,3])
T1=np.reshape(TOP[0:(Trapnumber+1)*2],(Trapnumber+1,2))
OptTPAR[:,0:2]=T1
OptX1=TOP[Trapnumber*2+2]
OptX2=TOP[Trapnumber*2+3]
OptFillH=TOP[Trapnumber*2+4]
OptSPAR=TOP[Trapnumber*2+5:Trapnumber*2+9]
OptSLD2=OptSPAR[3]
(OptCoord,FillT)= CD.IMECCoordAssign(OptTPAR,Trapnumber,OptX1,OptX2,OptFillH,OptSLD2,Pitch)
plt.figure(1)
CDp.plotPSPVP(OptCoord,Trapnumber)

plt.figure(7)
plt.semilogy(Qz[:,0],Intensity[:,0],'.')
plt.semilogy(Qz[:,0],SimInt1[:,0])

     
