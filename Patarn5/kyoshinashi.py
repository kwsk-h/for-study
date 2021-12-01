# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:22:14 2021

@author: kwsk0
"""
import numpy as np
import matplotlib.pyplot as plt


pi_ans = np.array([0.1, 0.4, 0.5])
w = [1,2,2,2,2,3,3,3,3,3]

X1 = [0,0,1,1,1,1,1,1,1,1]
X2 = [0,0,0,0,1,1,1,1,1,1]
X3 = [0,0,0,0,0,0,0,1,1,1]
X = np.array([X1, X2, X3])
theta = np.array([[0.8, 0.2],[0.6, 0.4], [0.3, 0.7]])

n = 10000
count = 0  # 奇数回数
x = np.zeros(n)
for t in range(n):
    dice = w[np.random.randint(0, 10)]
    x[t] = X[dice-1][np.random.randint(0, 10)]
    count += 1 if x[t] else 0

r = np.array([count,n-count])


pi = np.array([0.3, 0.5, 0.2])
pi_time = np.zeros((51,3))
pi_time[0] = pi
P = np.zeros((3, 2))
for x in range(50):
    for i in range(3):
        for k in range(2):
            P[i][k] = pi[i]*theta[i][k]/np.inner(pi, theta[:, k])
        pi[i] = np.inner(r, P[i, :])/n

    pi_time[x+1] = pi

fig = plt.figure(figsize=(11, 6))  # plot in new figure
plt.rcParams['font.size'] = 20  # フォントの大きさ
# xlabelがはみでてしまうので、subplotのサイズ調整
fig.subplots_adjust(bottom=0.2)
plt.title("r1 = " + str(count)+", r2 = " + str(n-count))
plt.subplot(1, 1, 1)
for i in range(3):
    plt.plot(pi_time[:, i], label = "ω" + str(i+1))
plt.grid(True)
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', borderaxespad=0, ncol=4, fontsize=20)