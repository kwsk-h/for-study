#pragma once
/*
LLGMN program

�����w�K�C�ꊇ�w�K����
TA�ꊇ�w�K����

�\�\�\�\�\�����ݒ�\�\�\�\�\
���͎����� D					�@2
�N���X�� K						�@4
�R���|�[�l���g�� M	�@2
�w�K�� ��								0.1

�cTA�p�c
 beta							0.8
smp_time					0.001
�\�\�\�\�\�\�\�\�\�\�\�\�\�\
*/

#include<stdio.h>
#include<iostream>
#include<vector>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<fstream>
#include<string>
#include<random>
#include<algorithm>
#include<iomanip>
#include "vecMACRO.h"
using namespace std;

class LLGMN
{
public:	
	int _D;				//���͎���
	int _H;				//����`�ϊ���̎���
	int _K;				//�N���X
	int _M;				//�R���|�[�l���g��
	double _epsilon;	//�w�K��
	vector<vector<vector<double>>> _weight; //�d�ݗp�@weight[K][M][H]

	//LLGMN.cpp
	LLGMN(int D = 2, int K = 4, int M = 2, double epsilon = 0.1, double beta = 0.7, double time = 1.0, double smp_time = 0.001);	//�R���X�g���N�^
	~LLGMN();	//�f�X�g���N�^
	void input_conversion(const vector<double>& input_data, vector<double>& converted_input); //����`�ϊ��p
	void set_weight(void); //�d�݂̏����l�ݒ�p
	void forward(vector<double>& input_data, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O); //�������v�Z�p
	void backward_online(vector<double>input_label, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O, vector<vector<vector<double>>>& grad); //�t�����v�Z�C�����w�K
	void backward_batch(vector<vector<double>>input_label, vector<vector<vector<double>>>& O_2, vector<vector<double>>& Y, vector<vector<double>>& O, vector<vector<vector<double>>>& grad); //�t�����v�Z�C�ꊇ�w�K
	void weight_update(vector<vector<vector<double>>>& grad); //�t�����v�Z�C�d�ݍX�V�����p
	void learn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�����w�K�p
	void learn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�ꊇ�w�K�p
	vector<vector<double>> test(vector<vector<double>>& test_data, vector<vector<double>>& test_label); //�e�X�g�p
	double getAccuracy(vector<vector<double>>& Y, const vector<vector<double>>& label); //���𗦌v�Z return Accuracy;


	//TA_LLGMN�p
	double _beta;					//�w�K�p�����[�^��
	double _time;					//�w�K����
	double _eta;					// �]���֐��̏����l�ɂ���ǓI�Ȋw�K���̌���
	double _gamma;			// �]���֐��̕ϓ��ɂ��Ǐ��I�Ȋw�K���̒���
	double _smp_time;			//�T���v�����O���ԇ�t[s]
	//TA_LLGMN.cpp
	void w_update(vector<vector<vector<double>>>& grad);	//�t�����v�Z�C�d�ݍX�V�����p
	void gamma_update(double J0, vector<vector<vector<double>>>& grad);		//���X�V
	void set_eta(double J);	//�ŃZ�b�g
	void talearn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�����w�K�p
	void talearn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�ꊇ�w�K�p
};

