# -*- coding: utf-8 -*-
"""
Created on Fri May 21 23:12:58 2021

@author: kwsk0
"""

import numpy as np
import matplotlib.pyplot as plt

COIN = 2  # coin 1,2,3

X1 = [0,0,1,1,1,1,1,1,1,1]
X2 = [0,0,0,0,1,1,1,1,1,1]
X3 = [0,0,0,0,0,0,0,1,1,1]
X = np.array([X1, X2, X3])

# [T, H]
theta1 = [0.2, 0.8]
theta2 = [0.4, 0.6]
theta3 = [0.7, 0.3]
theta = np.array([theta1, theta2, theta3])

pi1 = 0.1
pi2 = 0.4
pi3 = 0.5
pi = np.array([pi1, pi2, pi3])

N = 100
P = np.zeros((3, N))
x = np.zeros(N)
count = 0  # 表回数

for n in range(N):
    x[n] = X_n = X[COIN-1][np.random.randint(0, 10)]
    count += 1 if x[n] else 0
    for num in range(3):
        if n == 0:
            P[num][n] = (theta[num][X_n]/np.inner(pi, theta[:, X_n]))*pi[num]
        else:
            P[num][n] = (theta[num][X_n]/np.inner(P[:, n-1], theta[:, X_n]))*P[num][n-1]


times = np.arange(0, N)
fig = plt.figure(figsize=(11, 6))  # plot in new figure
plt.rcParams['font.size'] = 20  # フォントの大きさ
# xlabelがはみでてしまうので、subplotのサイズ調整
fig.subplots_adjust(bottom=0.2)
plt.title("coin" + str(COIN) + ", r = " + str(count))

plt.subplot(1, 1, 1)
plt.bar(times, x, width=1.0, color = "black", alpha=0.2)
plt.grid(True)

for num in range(3):
    plt.subplot(1, 1, 1)
    plt.plot(P[num][:], label = "ω" + str(num+1))
    plt.grid(True)

plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', borderaxespad=0, ncol=4, fontsize=20)