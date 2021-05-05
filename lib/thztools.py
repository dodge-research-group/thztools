import math as m
import numpy as np

xd = [0, .125, 0.250000000000000, 0.375000000000000, .5, .625,.75, .875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875]
xi = [.1, .2 , .3 , .4, .5, .625]
yd = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1] #xd and yd chosen for xdata and ydata
N = len(xd)
NI = len(xi) #number of interpolants



def trigcardinal(x, N):  
	if x == 0:  #takes care of the situations where xi-x[k] is 0, limit as approaching 0 is 1
		tau = 1
	elif N % 2 == 1:  #for the normal odd N case
		tau = m.sin((N * m.pi * x) / 2) / (N * m.sin((m.pi * x) / 2))
	elif N % 2 == 0:  #normal even N case
		tau = m.sin((N * m.pi * x) / 2) / (N * m.tan((m.pi * x) / 2))
	return tau  


def triginterp(xi, xd, yd):
  
  ptotal = []  #defines the variable ptotal as a list, will be filled with all the tau values
  
  
  ## The next for loop computes all the differences between the interpolants and
  ## the given x-values. For example: x = [1, 2] xi = [3 4], then xf[0] = [2 3]
  ## 
  ## In the subloop, the xf and y value indexed to each other and are used to calculate tau
  
  
  for j in range(0, N): # calculating xi-xk to be used in trigcardinal funciton
    x = np.repeat(xd[j], NI)  #builds a 1 by NI(number of interpolants) matrix composed of one of the given x values
    xf = np.subtract(xi, x)  #this performs the (xf -xk) calculation to create the matrices to be used in trigcardinal
    yindex = yd[j] #this is the y value in the tau calculation
    
    for k in range(0, NI):  #looping from 0 to NI (# of interpolants) - 1, we calculate tau and append it to list ptotal
        tau = m[k] + yindex * trigcardinal(xf[k] ,N) 
        ptotal.append(tau)
        
 
 
  #Arranges all tau values into N 1xNI matrices. Ex. if there 8 data points and 3
  #interpolants, then there will be 24 taus in the ptotal list. This code rearranges
  #that 1x24 list into 8 1x3 lists to be summed. 
  
  xy = np.array(ptotal).reshape(N,NI) 
  
  
  
  P = np.zeros_like(xi) #creates a zero matrix for P
  
  for i in range(0,len(xy)):  #loops through the xy matrix to sum all the tau values
    P =  P + xy[i]
  print(P) #the resulting matrix is the interpolated values

triginterp(xi, xd, yd)
