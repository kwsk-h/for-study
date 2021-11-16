# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:57:38 2021

@author: kwsk0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math


#parameter
mu, sigma, pi = np.array([3, -2]), np.array([1, 1]), np.array([0.6, 0.4])
samp = 500
c = 2

#gmm
x = [np.random.normal(mu[i],np.sqrt(sigma[i]),int(samp*pi[i])) for i in range(c)]
X = np.append(x[0], x[1])

t = np.arange(-5,15,0.04)
pdf = [norm.pdf(t,mu[i],np.sqrt(sigma[i])) for i in range(c)]
PDF = pi[0]*pdf[0]+pi[1]*pdf[1]

#plot
fig, ax = plt.subplots()
ax.hist(X,bins=30, color='cyan', ec='black')
ax.set_xlim(-5,7)
ax2 = ax.twinx()
ax2.tick_params(axis='y')#, colors='white'
ax2.plot(t,PDF, color='black')
ax2.plot(t,pi[0]*pdf[0], color='blue', linestyle='--')
ax2.plot(t,pi[1]*pdf[1], color='red', linestyle='--')
ax2.set_xlim(-5,7)
ax2.set_ylim(0,0.25)


'''-----------------
    parameter推定
-----------------'''
#convert X -> sample num, pdf -> p(x_t|w_i)
Xcnv = np.array((X+5)/0.04, dtype=int)

#step1 parameter set
mu = np.array([-2, -3])
bef_likelihood = -99999
ct = 0
while(True):
    ct+=1
    #step2 calculate P(w_i|x_t)
    pdf = [norm.pdf(t,mu[i],np.sqrt(sigma[i])) for i in range(c)]
    pxw = np.array([[pdf[i][Xcnv[t]] for t in range(samp)] for i in range(c)])  # p(x_t|w_i)
    px = np.sum(pi*pxw.T, axis=1)  # p(x_t)
    Pwx = (pi*pxw.T).T/px  # P(w_i|x_t)

    #step3 parameter update
    mu = np.sum((Pwx*X), axis=1)/np.sum(Pwx, axis=1)

    #step4 likelihood
    likelihood = np.sum(np.log(px))
    if(likelihood - bef_likelihood < 10):
        break
    else:
        bef_likelihood = likelihood
