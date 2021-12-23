#pragma once

#include<stdio.h>
#include<iostream>
#include<vector>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include <fstream>
#include <string>
#include <random>
#include "vecMACRO.h"

using namespace std;

class NN
{
public:
	int _D;				//入力次元
	int _C;				//クラス数
	int _M;				//中間層の要素数
	int _Mk;			//中間層の層数
	double _epsilon;	//学習率
	int _batch_size; //batchサイズの指定

	//各層の重みや各層の出力に利用する構造体
	template <typename T>
	struct LAYER {
		vector<T> First;				//入力層から中間層へ
		vector<vector<T>> Mid;	//中間層から中間層へ
		vector<T> Last;				//中間層から出力層へ
	};


	//---------------------------------------------------------------
	//各層の重み _weight 
	//_weight.First[_D + 1][_M]
	//_weight.Mid[_Mk - 1][_M + 1][_M]
	//_weight.Last[_M + 1][_C]
	//---------------------------------------------------------------
	LAYER<vector<double>> _weight;

	//---------------------------------------------------------------
	//各層の出力 _output[sample] = LAYER<double> 
	//_output[n].First[_D]
	//_output[n].Mid[_Mk][_M]
	//_output[n].Last[_C]
	//---------------------------------------------------------------
	vector<LAYER<double>> _output;

	//Constructor
	NN(int D, int C, int M, int Mk, double epsilon, int batch_size);//Parameter setting
	//Destructor
	~NN();

	//重みの初期値設定
	void setWeight();

	//vector sum
	double sigma(const vector<double>& V);

	//return 1/(1+exp(-s))
	vector<double> sigmoid(vector<double>& s);

	//return y(1-y)
	double d_sigmaoid(double y);

	//前向き計算　input_data：入力データ( = input_datas[n])，n：サブセット内の番号
	void forward(const vector<double> input_data, int n);

	//誤差逆伝搬 重み更新　delta：誤差，n：サブセット内の番号
	void backward(const vector<double>& delta, int n);

	//学習　input_datas：入力データ，input_labels：教師ラベル
	void Learning(const vector<vector<double>> input_datas, const vector<vector<double>> input_labels);

	//出力結果チェック用　input_datas：入力データ，input_labels：教師ラベル
	void check(const vector<vector<double>> input_datas, const vector<vector<double>> input_labels);
};