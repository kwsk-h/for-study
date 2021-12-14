#pragma once

#include<stdio.h>
#include<iostream>
#include<vector>
#include<fstream>
#include<string>
#include<algorithm>
#include"vecMACRO.h"
using namespace std;

class ConfusionMatrix {
private:
	int C;  //クラス数
	int N; //総サンプル数
	vector<vector<double>> OUT;
	vector<vector<double>> Label;
	vector<vector<double>> confusionMatrix;  //confusionMatrix[真クラス][予測クラス]

public:
	//Constructor
	ConfusionMatrix();
	ConfusionMatrix(const vector<vector<double>> out, const vector<vector<double>> label);

	//Destructor
	~ConfusionMatrix();

	//settting init Pram
	void setPram(const vector<vector<double>> out, const vector<vector<double>> label);
	//
	vector<vector<double>>  getConfusionMatrix();
	double getAccuracy();
	double getAccuracy(int classNum);

};

//--------------------------------
//Constructor
// 
// out[sample][class] ：事後確率
// label[sample][class] ：正解ラベル
//--------------------------------
ConfusionMatrix::ConfusionMatrix()
{

}
ConfusionMatrix::ConfusionMatrix(const vector<vector<double>> out, const vector<vector<double>> label)
{
	setPram(out, label);
}

//Destructor
ConfusionMatrix::~ConfusionMatrix()
{

}

//--------------------------------
//settting init Pram
// 
// out[sample][class] ：事後確率
// label[sample][class] ：正解ラベル
//--------------------------------
void ConfusionMatrix::setPram(const vector<vector<double>> out, const vector<vector<double>> label)
{
	OUT = out;
	Label = label;
	C = out[0].size();
	N = out.size();
	confusionMatrix = make_v<double>(C, C);
	fill_v(confusionMatrix, 0);
	getConfusionMatrix();
}

//--------------------------------
//getConfusionMatrix
// 
// return confusionMatrix[真クラス][予測クラス] ：混合行列
//--------------------------------
vector<vector<double>> ConfusionMatrix::getConfusionMatrix()
{
	for (int i = 0; i < N; i++) {
		//事後確率最大が予測クラス
		vector<double>::iterator piter = max_element(OUT[i].begin(), OUT[i].end()); //最大値のイテレータ取得
		size_t predict = distance(OUT[i].begin(), piter);//最大値	となるクラスを取得

		//真クラスを取得
		vector<double>::iterator citer = max_element(Label[i].begin(), Label[i].end()); //最大値のイテレータ取得
		size_t correct = distance(Label[i].begin(), citer);//最大値	となるクラスを取得

		//混合行列に配置
		confusionMatrix[correct][predict] += 1;
	}

	return confusionMatrix;
}

double ConfusionMatrix::getAccuracy()
{
		int count = 0;	//正解数
		int ct = 0;		//総数
	for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) {
		if (i == j) count += confusionMatrix[i][i];
		ct += confusionMatrix[i][j];
	}
	return (double)count / (double)ct;
}
