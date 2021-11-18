# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:18:29 2021

@author: kwsk0
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def gaussian(x, mu, sigma):
    # 分散共分散行列の行列式
    det = np.linalg.det(sigma)
    # print(det)
    # 分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)
    n = x.ndim
    # print(inv)
    return np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))


def pltGaussian(mu, sigma, pi):
    clis = ['red', 'green', 'blue', 'cyan', 'black']
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    for i in range(c):
        mu_ = mu[i]
        sigma_ = sigma[i]
        Z = gaussian(z, mu_, sigma_).reshape(x.shape)

        ax1.scatter(XYs[0], XYs[1], marker=".", alpha=.5)
        ax1.contour(x, y, Z, levels=[np.sort(Z.reshape([1, -1]))[0][int(0.95*Z.size)]], linewidths=int(10*pi[i]), colors=clis[i])
    ax1.grid(True)
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)


samp = 1000
c = 5
rg = np.random.default_rng()

ans_mu = np.array([[-4, 4], [-3, -2], [0, -3], [2, 3], [4, -4]])
ans_sigma = np.array([[[1.2, -0.3], [-0.3, 0.4]],
                      [[0.5, 0.4], [0.4, 0.6]],
                      [[0.3, 0.5], [0.5, 1.8]],
                      [[0.6, -0.6], [-0.6, 1.6]],
                      [[0.2, 0], [0, 0.2]]])
ans_pi = [0.4, 0.2, 0.2, 0.1, 0.1]

x, y = np.meshgrid(np.arange(-8, 8, 0.1), np.arange(-8, 8, 0.1))
z = np.c_[x.ravel(),y.ravel()]

fig, ax = plt.subplots(figsize=(10, 10))
for i in range(c):
    mu = ans_mu[i]
    sigma = ans_sigma[i]
    labelname = str(mu)+" "+str(sigma)

    Z = gaussian(z, mu, sigma).reshape(x.shape)
    X, Y = rg.multivariate_normal(mu, sigma, int(ans_pi[i]*samp)).T

    # convert X Y -> sample num
    if i == 0:
        XYs = [X, Y]
        XYs_cnv = [np.array((X+8)*10, dtype=int), np.array((Y+8)*10, dtype=int)]
    else:
        XYs = np.hstack((XYs, [X, Y]))
        XYs_cnv = np.hstack((XYs_cnv, [np.array((X+8)*10, dtype=int), np.array((Y+8)*10, dtype=int)]))

    ax.scatter(X, Y, marker=".", label=labelname, alpha=.5)
    ax.contour(x, y, Z, levels=[np.sort(Z.reshape([1, -1]))[0][int(0.95*Z.size)]], linewidths=int(10*ans_pi[i]))

ax.grid(True)
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
# ax.legend(title="(mean / covariance)", bbox_to_anchor=(1.1, 1.02),)


'''-----------------
    parameter推定
-----------------'''
# step1 parameter set
mu = np.array([[-4, 0], [-3, 0], [0, 0], [2, 0], [4, 0]])
sigma = np.array([[[1, 0], [0, 1]],
                  [[1, 0], [0, 1]],
                  [[1, 0], [0, 1]],
                  [[1, 0], [0, 1]],
                  [[1, 0], [0, 1]]])
pi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
pis = pi
ct = 0
i = 0
while(True):
    ct += 1
    pltGaussian(mu, sigma, pi)

    # step2 calculate P(w_i|x_t)
    pdf = np.array([gaussian(z, mu[i], sigma[i]).reshape(x.shape) for i in range(c)])
    pxw = np.array([[pdf[i][XYs_cnv[0][t]][XYs_cnv[1][t]] for t in range(samp)] for i in range(c)])  # p(x_t|w_i)
    px = np.sum(pi*pxw.T, axis=1)  # p(x_t)
    Pwx = (pi*pxw.T).T/px  # P(w_i|x_t)
    N = np.sum(Pwx, axis=1)

    # step3 parameter update
    pi = N/samp
    mu = np.array([np.sum(Pwx[i]*XYs, axis=1)/N[i] for i in range(c)])
    sigma = np.array([np.sum([Pwx[i][t]*(np.identity(2).T*(XYs.T[t]-mu[i])*(XYs.T[t]-mu[i]).T+(np.ones((2, 2))-np.identity(2))*np.prod(XYs.T[t]-mu[i]))/N[i] for t in range(samp)], axis=0) for i in range(c)])

    pis = np.vstack((pis, pi))

    # step4 likelihood
    pdf = np.array([gaussian(z, mu[i], sigma[i]).reshape(x.shape) for i in range(c)])
    pxw = np.array([[pdf[i][XYs_cnv[0][t]][XYs_cnv[1][t]] for t in range(samp)] for i in range(c)])  # p(x_t|w_i)
    px = np.sum(pi*pxw.T, axis=1)  # p(x_t)
    likelihood = np.sum(np.log(px))
    print(ct, likelihood)
    if(ct==1):
        likelihoods = likelihood
    else:
        likelihoods = np.append(likelihoods, likelihood)
        if(likelihoods[-1] - likelihoods[-2] < 1):
            pltGaussian(mu, sigma, pi)
            break
