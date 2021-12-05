# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 04:22:51 2021

@author: kwsk0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CRP:
    """Chinese restaurant process."""

    def __init__(self, num, alpha):
        """Params num <- int."""
        self.N = num  # 何人まで見るか
        self.n = 1  # n番目の人
        self.alpha = alpha
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
            sit = np.array([self.table[i]/(self.n-1+self.alpha) for i in range(self.c)])
            sit = np.append(sit, self.alpha/(self.n-1+self.alpha))
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


class doCRP:
    """CRP実行."""

    def __init__(self, loop):
        """Params num <- int."""
        self.num = 1000
        self.alpha = [2, 10]
        for i in range(loop):
            self.do(i)

    def do(self, i):
        """Do CRP."""
        for a in self.alpha:
            crp = CRP(self.num, a)
            crp.using_table()
            df = crp.transition.set_index('n')
            result = np.sort(crp.table)[::-1]
            if a == self.alpha[0]:
                dff = df

            print(i, a, crp.c)
            fig2 = plt.figure(figsize=(8, 8))
            ax2 = fig2.add_subplot(111)
            ax2.bar(np.arange(1, crp.c+1), result)
            ax2.set_xlabel('table', fontsize=20)
            ax2.set_ylabel('n_i', fontsize=20)
            ax2.grid(True)
            fig2.savefig('table'+str(crp.c)+'_'+str(i)+'.png', bbox_inches="tight", pad_inches=0.1)

        fig = plt.figure()  # reset用
        dfc = pd.concat([dff['c'], df['c']], axis='columns').set_axis(['α = '+str(x) for x in self.alpha], axis='columns')
        ax = dfc.plot(figsize=(10, 6), grid=True)
        ax.set_xlabel('n', fontsize=20)
        ax.set_ylabel('c', fontsize=20)
        plt.savefig('n-c'+'_'+str(i)+'.png', bbox_inches="tight", pad_inches=0.1)


do = doCRP(3)
