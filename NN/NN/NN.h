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
	int _D;				//���͎���
	int _C;				//�N���X��
	int _M;				//���ԑw�̗v�f��
	int _Mk;			//���ԑw�̑w��
	double _epsilon;	//�w�K��
	int _batch_size; //batch�T�C�Y�̎w��

	//�e�w�̏d�݂�e�w�̏o�͂ɗ��p����\����
	template <typename T>
	struct LAYER {
		vector<T> First;				//���͑w���璆�ԑw��
		vector<vector<T>> Mid;	//���ԑw���璆�ԑw��
		vector<T> Last;				//���ԑw����o�͑w��
	};


	//---------------------------------------------------------------
	//�e�w�̏d�� _weight 
	//_weight.First[_D + 1][_M]
	//_weight.Mid[_Mk - 1][_M + 1][_M]
	//_weight.Last[_M + 1][_C]
	//---------------------------------------------------------------
	LAYER<vector<double>> _weight;

	//---------------------------------------------------------------
	//�e�w�̏o�� _output[sample] = LAYER<double> 
	//_output[n].First[_D]
	//_output[n].Mid[_Mk][_M]
	//_output[n].Last[_C]
	//---------------------------------------------------------------
	vector<LAYER<double>> _output;

	//Constructor
	NN(int D, int C, int M, int Mk, double epsilon, int batch_size);//Parameter setting
	//Destructor
	~NN();

	//�d�݂̏����l�ݒ�
	void setWeight();

	//vector sum
	double sigma(const vector<double>& V);

	//return 1/(1+exp(-s))
	vector<double> sigmoid(vector<double>& s);

	//return y(1-y)
	double d_sigmaoid(double y);

	//�O�����v�Z�@input_data�F���̓f�[�^( = input_datas[n])�Cn�F�T�u�Z�b�g���̔ԍ�
	void forward(const vector<double> input_data, int n);

	//�덷�t�`�� �d�ݍX�V�@delta�F�덷�Cn�F�T�u�Z�b�g���̔ԍ�
	void backward(const vector<double>& delta, int n);

	//�w�K�@input_datas�F���̓f�[�^�Cinput_labels�F���t���x��
	void Learning(const vector<vector<double>> input_datas, const vector<vector<double>> input_labels);

	//�o�͌��ʃ`�F�b�N�p�@input_datas�F���̓f�[�^�Cinput_labels�F���t���x��
	void check(const vector<vector<double>> input_datas, const vector<vector<double>> input_labels);
};