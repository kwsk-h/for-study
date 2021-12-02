# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:06:29 2021

@author: kwsk0
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PYP:
    """Pitman-Yor process."""

    def __init__(self, num, alpha, beta):
        """Params num <- int."""
        self.N = num  # 何人まで見るか
        self.n = 1  # n番目の人
        self.alpha = alpha
        self.beta = beta
        self.c = 1  # 使用テーブル数
        self.transition = pd.DataFrame([(self.n, self.c, 1)], columns=['n', 'c', '1'])  # 推移
        self.table = np.array([1])  # 一人目は既に着席済み
        self.table_name = ['1']

    def exist_table(self, i):
        """既存のテーブルに着席. Params i <- int."""
        self.table[i-1] += 1  # テーブルiに一人追加

    def new_table(self):
        """新しいテーブルに着席."""
        self.c += 1
        self.table = np.append(self.table, 1)  # テーブルを追加
        self.table_name = np.append(self.table_name, str(self.c))

    def using_table(self):
        """2人目～N人目まで順に計算."""
        while(self.n < self.N):
            self.n += 1
            if self.n%1000 == 0:
                print(self.n, end='-')
            sit = np.array([(self.table[i] - self.beta)/(self.n-1+self.alpha) for i in range(self.c)])
            sit = np.append(sit, (self.alpha+self.beta*self.c)/(self.n-1+self.alpha))
            table_name = np.append(self.table_name, str(self.c+1))
            flag = np.random.choice(table_name, p=sit)
            if flag == str(self.c+1):
                self.new_table()
            else:
                self.exist_table(int(flag))

            n_data = [(self.n, self.c) + tuple(sit)]
            n_col = np.append(['n', 'c'], table_name)
            dfnow = pd.DataFrame(n_data, columns=n_col)
            self.transition = self.transition.append(dfnow, ignore_index=True)
        print(';')


class doPYP:
    """PYP実行."""

    def __init__(self):
        """Params num <- int."""
        self.alpha = 2
        self.beta = [0, 0.2, 0.3, 0.4]
        self.do(1000)
        self.dolog(100000)
        self.dff
        self.tables

    def do(self, num):
        """Do PYP β比較."""
        for b in self.beta:
            pyp = PYP(num, self.alpha, b)
            pyp.using_table()
            df = pyp.transition.set_index('n')
            result = pyp.table
            if b == self.beta[0]:
                self.dff = df['c']
            else:
                self.dff = pd.concat([self.dff, df['c']], axis='columns')

        fig = plt.figure()  # reset用
        self.dff = self.dff.set_axis(['β = '+str(x) for x in self.beta], axis='columns')
        ax = self.dff.plot(figsize=(10, 6), grid=True)
        ax.set_xlabel('n', fontsize=20)
        ax.set_ylabel('c', fontsize=20)
        plt.savefig('Pitman-Yor_n-c.png', bbox_inches="tight", pad_inches=0.1)

    def dolog(self, num):
        """Do PYP table-n_i log."""

        pyp = [PYP(num, self.alpha, b) for b in [0, 0.8]]
        [pyp[i].using_table() for i in range(2)]
        self.tables = [pyp[i].table for i in range(2)]

        fig2 = plt.figure()  # reset用
        ax2 = fig2.add_subplot(111)
        ax2.plot(np.arange(1, pyp[0].c+1), self.tables[0])
        ax2.plot(np.arange(1, pyp[1].c+1), self.tables[1])
        ax2.set_xlabel('table', fontsize=20)
        ax2.set_ylabel('n_i', fontsize=20)
        ax2.set_yscale('log')  # メイン: y軸をlogスケールで描く
        ax2.set_xscale('log')
        plt.savefig('Pitman-Yor_table_log.png', bbox_inches="tight", pad_inches=0.1)

do = doPYP()
