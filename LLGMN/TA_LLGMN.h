#pragma once
/*
TA_LLGMN program

�N���XLLGMN���p��
�^�[�~�i���A�g���N�^����

�\�\�\�\�\�����ݒ�\�\�\�\�\
���͎����� D					�@2
�N���X�� K						�@4
�R���|�[�l���g�� M	�@2
TA�p beta
TA�p eta
�\�\�\�\�\�\�\�\�\�\�\�\�\�\
*/

#include "LLGMN.h"

class TA_LLGMN : public LLGMN
{
public:
	double beta;			  //�w�K�p�����[�^��
	double time;		  //�w�K�񐔁i�w�K����:d_t��p���Čv�Z����j
	double eta;			  // �]���֐��̏����l�ɂ���ǓI�Ȋw�K���̌���
	double gamma;	  // �]���֐��̕ϓ��ɂ��Ǐ��I�Ȋw�K���̒���

	void weight_update(vector<vector<vector<double>>>& grad); //�t�����v�Z�C�d�ݍX�V�����p
	void gamma_update(double J0, vector<vector<vector<double>>>& grad);
	void set_eta(double J);
};

//�d�ݍX�V�̊֐�
void TA_LLGMN::weight_update(vector<vector<vector<double>>>& grad)
{
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				_weight[i][j][k] -=  eta * gamma * grad[i][j][k];
			}
		}
	}
}

void TA_LLGMN::gamma_update(double J, vector<vector<vector<double>>>& grad)
{
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				gamma = pow(J, beta) / pow(grad[i][j][k], 2);
			}
		}
	}
}

void TA_LLGMN::set_eta(double J0)
{
	eta = pow(J0, (1.0 - beta)) / (double)(time * (1.0 - beta));
}