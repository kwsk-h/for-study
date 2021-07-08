#pragma once
/*
TA_LLGMN program

クラスLLGMNを継承
ターミナルアトラクタ導入

―――――初期設定―――――
入力次元数 D					　2
クラス数 K						　4
コンポーネント数 M	　2
TA用 beta
TA用 eta
――――――――――――――
*/

#include "LLGMN.h"

class TA_LLGMN : public LLGMN
{
public:
	double beta;			  //学習パラメータβ
	double time;		  //学習回数（学習時間:d_tを用いて計算する）
	double eta;			  // 評価関数の初期値による大局的な学習率の決定
	double gamma;	  // 評価関数の変動による局所的な学習率の調整

	void weight_update(vector<vector<vector<double>>>& grad); //逆方向計算，重み更新部分用
	void gamma_update(double J0, vector<vector<vector<double>>>& grad);
	void set_eta(double J);
};

//重み更新の関数
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