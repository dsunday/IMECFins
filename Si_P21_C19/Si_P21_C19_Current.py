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

DW = 2.2
I0 = .0001
Bk =1.13
Pitch = 83.7
FillH = 55;

SLD2=1.2
SPAR=np.zeros([4]) 
SPAR[0]=DW; SPAR[1]=I0; SPAR[2]=Bk;SPAR[3]=SLD2;

TPAR=np.zeros([Trapnumber+1,3])

TPAR[0,0]=15.53; TPAR[0,1]=55; TPAR[0,2]=0.8;
TPAR[1,0]=9; TPAR[1,1]=6.4; TPAR[1,2]=0.8;
TPAR[2,0]=6.4; TPAR[2,1]=40; TPAR[2,2]=0.8;
TPAR[3,0]=4; 

X1 = 15.46
X2 = 22
(Coord,FillT)= CD.IMECCoordAssign(TPAR,Trapnumber,X1,X2,FillH,SLD2,Pitch)
(FITPAR,FITPARLB,FITPARUB)=CD.IMEC_PBA(TPAR,SPAR,X1,X2,FillH,Trapnumber)

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

    
plt.figure(1)
CDp.plotPSPVP(Coord,Trapnumber)

plt.figure(2)
plt.semilogy(Qz[:,0],Intensity[:,0],'.')
plt.semilogy(Qz[:,0],SimInt[:,0])

C=np.sum(CD.Misfit(Intensity,SimInt))

plt.figure(3)
CDp.PlotQzCut(Qz,SimInt,Intensity,11)