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
	int C;  //�N���X��
	int N; //���T���v����
	vector<vector<double>> OUT;
	vector<vector<double>> Label;
	vector<vector<double>> confusionMatrix;  //confusionMatrix[�^�N���X][�\���N���X]

public:
	//Constructor
	ConfusionMatrix(const vector<vector<double>> out, const vector<vector<double>> label);

	//Destructor
	~ConfusionMatrix();


	//
	vector<vector<double>>  getConfusionMatrix();
	double getAccuracy();

};

//--------------------------------
//Constructor
// 
// out[sample][class] �F����m��
// label[sample][class] �F�������x��
//--------------------------------
ConfusionMatrix::ConfusionMatrix(const vector<vector<double>> out, const vector<vector<double>> label)
{
	OUT = out;
	Label = label;
	C = out[0].size();
	N = out.size();
	confusionMatrix = make_v<double>(C, C);
	fill_v(confusionMatrix, 0);
}

//Destructor
ConfusionMatrix::~ConfusionMatrix()
{

}

//--------------------------------
//getConfusionMatrix
// 
// return confusionMatrix[�^�N���X][�\���N���X] �F�����s��
//--------------------------------
vector<vector<double>> ConfusionMatrix::getConfusionMatrix()
{
	for (int i = 0; i < N; i++) {
		//����m���ő傪�\���N���X
		vector<double>::iterator piter = max_element(OUT[i].begin(), OUT[i].end()); //�ő�l�̃C�e���[�^�擾
		size_t predict = distance(OUT[i].begin(), piter);//�ő�l	�ƂȂ�N���X���擾

		//�^�N���X���擾
		vector<double>::iterator citer = max_element(Label[i].begin(), Label[i].end()); //�ő�l�̃C�e���[�^�擾
		size_t correct = distance(Label[i].begin(), citer);//�ő�l	�ƂȂ�N���X���擾

		//�����s��ɔz�u
		confusionMatrix[correct][predict] += 1;
	}

	return confusionMatrix;
}

double ConfusionMatrix::getAccuracy()
{
	int count = 0;	//����
	int ct = 0;		//����
	for (auto y : OUT) {
		//�ŏI�w�o��(����m��)���ő�̃N���X
		vector<double>::iterator iter = max_element(y.begin(), y.end()); //�ő�l�擾
		size_t index = distance(y.begin(), iter);//�ő�l�̃C�e���[�^�擾

		//�e�X�g���x���ƈ�v�����
		if (Label[ct][index] == 1) count++;
		ct++;
	}

	return (double)count / (double)ct;
}
