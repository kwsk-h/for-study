#pragma once

#include<stdio.h>
#include<iostream>
#include<vector>
#include<fstream>
#include<string>
#include<algorithm>
#include <numeric>
#include"vecMACRO.h"
using namespace std;

class ConfusionMatrix {
private:
	int C;  //クラス数
	int N; //総サンプル数
	vector<vector<double>> OUT;
	vector<vector<double>> Label;
	vector<vector<double>> confusionMatrix;  //confusionMatrix[真クラス][予測クラス]
	double Accuracy;
	vector<double> Recall;
	vector<double> Precision;
	vector<double> Fmeasure;

	void calcConfusionMatrix();
	void calcAccuracy();
	void calcRecalls();
	void calcPrecisions();
	void calcFmeasure();
public:
	//Constructor
	ConfusionMatrix();
	ConfusionMatrix(const vector<vector<double>> out, const vector<vector<double>> label);

	//settting init Pram
	void setPram(const vector<vector<double>> out, const vector<vector<double>> label);

	//getter
	vector<vector<double>>  getConfusionMatrix();
	double getAccuracy();
	double getRecall(int classNum);
	vector<double> getRecall();
	double getPrecision(int classNum);
	vector<double> getPrecision();
	double getFmeasure(int classNum);
	vector<double> getFmeasure();

	//save
	void resultSave(string savename);
};
