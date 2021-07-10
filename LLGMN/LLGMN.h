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

class LLGMN
{
public:	
	int _D = 2;//���͎���
	int _H = 1 + _D * (_D + 3) / 2;//����`�ϊ���̎���
	int _K = 4;//�N���X
	int _M = 2;//�R���|�[�l���g��
	double _epsilon = 0.1;//�w�K��
	vector<vector<vector<double>>> _weight = make_v<double>(_K, _M, _H); //�d�ݗp�@weight[K][M][H]
	//LLGMN.cpp
	void input_conversion(const vector<double>& input_data, vector<double>& converted_input); //����`�ϊ��p
	void set_weight(void); //�d�݂̏����l�ݒ�p
	void forward(vector<double>& input_data, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O); //�������v�Z�p
	void backward_online(vector<double>input_label, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O, vector<vector<vector<double>>>& grad); //�t�����v�Z�C�����w�K
	void backward_batch(vector<vector<double>>input_label, vector<vector<vector<double>>>& O_2, vector<vector<double>>& Y, vector<vector<double>>& O, vector<vector<vector<double>>>& grad); //�t�����v�Z�C�ꊇ�w�K
	void weight_update(vector<vector<vector<double>>>& grad); //�t�����v�Z�C�d�ݍX�V�����p
	void learn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�����w�K�p
	void learn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�ꊇ�w�K�p
	void test(vector<vector<double>>& test_data, vector<vector<double>>& test_label); //�e�X�g�p


	//TA_LLGMN�p
	double _beta = 0.7;					//�w�K�p�����[�^��
	double _time = 1.0;					//�w�K����
	double _eta = 0.0;					// �]���֐��̏����l�ɂ���ǓI�Ȋw�K���̌���
	double _gamma = 0.0;			// �]���֐��̕ϓ��ɂ��Ǐ��I�Ȋw�K���̒���
	double _smp_time = 0.001;	//�T���v�����O���ԇ�t[s]
	//TA_LLGMN.cpp
	void w_update(vector<vector<vector<double>>>& grad);	//�t�����v�Z�C�d�ݍX�V�����p
	void gamma_update(double J0, vector<vector<vector<double>>>& grad);		//���X�V
	void set_eta(double J);	//�ŃZ�b�g
	void talearn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�����w�K�p
	void talearn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�ꊇ�w�K�p
};

