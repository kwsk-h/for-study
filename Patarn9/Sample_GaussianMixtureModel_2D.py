# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 06:34:09 2021

@author: kwsk0
"""

import numpy as np
import math
#from matplotlib.mlab import bivariate_normal
import matplotlib.pyplot as plt

Colors = ["red", "green", "m", "blue"]
N = 1000
iter_num = 100
'''
Means = [[2, 2], [-2, -2], [2, -2], [-2, 2]]
Covs = [[[0.5, 0], [0, 0.25]], [[0.3, 0], [0, 0.6]],
        [[0.5, 0], [0, 0.25]], [[0.3, 0], [0, 0.6]]]
Pi = [0.1, 0.2, 0.3, 0.4]
'''
Means = [[-4, 4], [-4, -4], [0, 0], [4, 4], [4, -4]]
Covs = [[[0.8, 0], [0, 0.4]],
        [[0.5, 0], [0, 0.6]],
        [[0.3, 0], [0, 0.8]],
        [[0.6, 0], [0, 0.6]],
        [[0.9, 0], [0, 0.2]]]
Pi = [0.3, 0.2, 0.2, 0.2, 0.1]
K = len(Pi)
d = len(Means[0])
rg = np.random.default_rng()


def gaussian(x, y, mu, sigma):
    x = np.matrix([x, y])
    mu = np.matrix(mu)
    sig = np.matrix(sigma)
    assert sig.shape == (d, d), sig.shape
    if np.linalg.det(sig) * (2 * np.pi)**sig.ndim < 0:
        print("error!")
    a = np.sqrt(np.linalg.det(sig) * (2 * np.pi)**sig.ndim)
    b = np.linalg.det(-0.5 * (x - mu) * sig.I * (x - mu).T)
    return np.exp(b) / a


def likelihood(X, Y, pi, mu, sigma):
    L = sum([np.log(sum([pi[i] * gaussian(X[t], Y[t], mu[i], sigma[i]) for i in range(K)]))
             for t in range(N)])
    return L


def initialize(X, Y, classes):
    pi = [1 / classes] * classes
    mu = [np.random.uniform(min(X), max(X), d) for _ in range(classes)]
    sigma = [np.eye(d) for _ in range(classes)]
    return pi, mu, sigma


def expectation(X, Y, pi, mu, sigma):
    r = []
    for t in range(len(X)):
        denom = sum([pi[l] * gaussian(X[t], Y[t], mu[l], sigma[l])
                     for l in range(K)])
        numer = np.array([pi[i] * gaussian(X[t], Y[t], mu[i], sigma[i])
                          for i in range(K)])
        r.append(numer / denom)
    assert np.array(r).shape == (N, K)
    return r


def maximization(X, Y, r):
    N_k = np.sum(r, axis=0)
    assert len(N_k) == K
    pi = N_k / N
    assert len(pi) == K
    mu = []
    sigma = []
    for i in range(K):
        mu_i = np.zeros(d)
        for t in range(N):
            mu_i += r[t][i] * np.array([X[t], Y[t]])
        mu_i /= N_k[i]
        mu.append(mu_i)
        sigma_i = np.zeros((d, d))
        for t in range(N):
            temp = np.array([X[t], Y[t]]) - mu[i]
            sigma_i += r[t][i] * \
                np.matrix(temp).reshape(d, 1) * np.matrix(temp).reshape(1, d)
        sigma_i /= N_k[i]
        sigma.append(sigma_i)
    assert np.array(mu).shape == (K, d), mu.shape
    assert np.array(sigma).shape == (K, d, d), sigma.shape
    return pi, mu, sigma


def show_param(pi, mu, sigma):
    print("pi:{}".format(pi))
    print("mu:")
    for i in range(K):
        print(mu[i])
    print("sigma:")
    for i in range(K):
        print(sigma[i])

'''
def EM_plot(X, Y, r, iter, mu, sigma):
    plt.title = "iter{}".format(iter)
    # クラスタリング
    for t in range(N):
        i = np.argmax([r[t][i] for i in range(K)])
        plt.scatter(X[t], Y[t],
                    c=Colors[i], marker="o", s=5)
    # 平均点と等高線の表示
    xlist = np.linspace(min(X), max(X), 100)
    ylist = np.linspace(min(Y), max(Y), 100)
    x, y = np.meshgrid(xlist, ylist)
    for i in range(K):
        plt.scatter(mu[i][0], mu[i][1], c="k", marker="o")
        z = bivariate_normal(x, y, np.sqrt(sigma[i][0][0]), np.sqrt(
            sigma[i][1][1]), mu[i][0], mu[i][1], sigma[i][0][1])
        cs = plt.contour(x, y, z, 3, colors='k', linewidths=1)

    plt.show()
'''

def gaussian2(x, mu, sigma):
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
    for i in range(K):
        mu_ = mu[i]
        sigma_ = sigma[i]
        Z = gaussian2(z, mu_, sigma_).reshape(x.shape)

        ax1.scatter(XYs[0], XYs[1], marker=".", alpha=.5)
        ax1.contour(x, y, Z, levels=[np.sort(Z.reshape([1, -1]))[0][int(0.95*Z.size)]], linewidths=int(10*pi[i]), colors=clis[i])
    ax1.grid(True)
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)


def EM(X, Y, classes, iter_num):
    pi, mu, sigma = initialize(X, Y, classes)
    pltGaussian(mu, sigma, pi)
    for j in range(iter_num):
        prev = likelihood(X, Y, pi, mu, sigma)
        r = expectation(X, Y, pi, mu, sigma)
        pi, mu, sigma = maximization(X, Y, r)
        now = likelihood(X, Y, pi, mu, sigma)
        print("iter:{} likelihood:{}".format(j + 1, now))
        if now - prev < 0.01:
            print("converged!")
            show_param(pi, mu, sigma)
            pltGaussian(mu, sigma, pi)
            #EM_plot(X, Y, r, j + 1, mu, sigma)
            return pi, mu, sigma
        if (j + 1) % 5 == 0:
            pltGaussian(mu, sigma, pi)
            #EM_plot(X, Y, r,  j + 1, mu, sigma)
    print("not converged")
    return pi, mu, sigma


if __name__ == "__main__":
    x, y = np.meshgrid(np.arange(-8, 8, 0.1), np.arange(-8, 8, 0.1))
    z = np.c_[x.ravel(),y.ravel()]
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(K):
        mu = Means[i]
        sigma = Covs[i]
        labelname = str(mu)+" "+str(sigma)

        Z = gaussian2(z, mu, sigma).reshape(x.shape)
        X, Y = rg.multivariate_normal(mu, sigma, int(Pi[i]*N)).T

        # convert X Y -> sample num
        if i == 0:
            XYs = [X, Y]
            XYs_cnv = [np.array((X+8)*10, dtype=int), np.array((Y+8)*10, dtype=int)]
        else:
            XYs = np.hstack((XYs, [X, Y]))
            XYs_cnv = np.hstack((XYs_cnv, [np.array((X+8)*10, dtype=int), np.array((Y+8)*10, dtype=int)]))

        ax.scatter(X, Y, marker=".", label=labelname, alpha=.5)
        ax.contour(x, y, Z, levels=[np.sort(Z.reshape([1, -1]))[0][int(0.95*Z.size)]], linewidths=int(10*Pi[i]))

    ax.grid(True)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    # ax.legend(title="(mean / covariance)", bbox_to_anchor=(1.1, 1.02),)

    EM(XYs[0], XYs[1], K, iter_num)