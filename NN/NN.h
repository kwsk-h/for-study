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

using namespace std;

//�������z��m�ۋy�я������p�}�N��
template<typename T>
vector<T> make_v(size_t a) { return vector<T>(a); }
template<typename T, typename... Ts>
auto make_v(size_t a, Ts... ts)
{
	return vector<decltype(make_v<T>(ts...))>(a, make_v<T>(ts...));
}
template<typename T, typename V>
typename enable_if<is_class<T>::value == 0>::type
fill_v(T& t, const V& v) { t = v; }
template<typename T, typename V>
typename enable_if<is_class<T>::value != 0>::type
fill_v(T& t, const V& v)
{
	for (auto& e : t) fill_v(e, v);
}
//�}�N�������܂�

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

	LAYER<vector<double>> _weight;	//�e�w�̏d��
	LAYER<double> _output;					//�e�w�̏o��

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

	//�O�����v�Z�@input_data�F���̓f�[�^
	void forward(const vector<double> input_data);

	//�덷�t�`�� �d�ݍX�V�@delta�F�덷
	void backward(const vector<double>& delta);

	//�w�K
	void Learning();
};