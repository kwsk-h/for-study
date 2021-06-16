#pragma once
/*
LLGMN program

�����w�K�C�ꊇ�w�K����

�\�\�\�\�\�����ݒ�\�\�\�\�\
���͎����� D					�@2
�N���X�� K						�@4
�R���|�[�l���g�� M	�@2
�w�K�� ��								0.1
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
using namespace std;

class LLGMN
{
public:
	int _D = 2;//���͎���
	int _H = 1 + _D * (_D + 3) / 2;//����`�ϊ���̎���
	int _K = 4;//�N���X
	int _M = 2;//�R���|�[�l���g��
	double _epsilon = 0.1;//�w�K��
	vector<vector<vector<double>>> _weight; //�d�ݗp�@weight[K][M][H]
	void input_conversion(const vector<double>& input_data, vector<double>& converted_input); //����`�ϊ��p
	void set_weight(vector<vector<vector<double>>>& weight); //�d�݂̏����l�ݒ�p
	void forward(vector<double>& input_data, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O); //�������v�Z�p
	void backward_online(vector<double>input_label, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O, vector<vector<vector<double>>>& grad); //�t�����v�Z�C�����w�K
	void backward_batch(vector<vector<double>>input_label, vector<vector<vector<double>>>& O_2, vector<vector<double>>& Y, vector<vector<double>>& O, vector<vector<vector<double>>>& grad); //�t�����v�Z�C�ꊇ�w�K
	void weight_update(vector<vector<vector<double>>>& grad); //�t�����v�Z�C�d�ݍX�V�����p
	void learn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�����w�K�p
	void learn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�ꊇ�w�K�p
	void test(vector<vector<double>>& test_data, vector<vector<double>>& test_label); //�e�X�g�p
};

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