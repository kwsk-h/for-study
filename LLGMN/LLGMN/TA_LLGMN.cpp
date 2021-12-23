#include "LLGMN.h"

//重み更新の関数
void LLGMN::w_update(vector<vector<vector<double>>>& grad)
{
	double d_weight = 0.0;		 //現在の重み変化率
	double d_pre_weight = 0.0; //一時刻前の重み変化率
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				d_weight = (-1.0) * _eta * _gamma * grad[i][j][k];
				_weight[i][j][k] += (_smp_time / 2.0) * (d_weight + d_pre_weight);
				d_pre_weight = d_weight;
			}
		}
	}
}

void LLGMN::gamma_update(double J, vector<vector<vector<double>>>& grad)
{
	double sum_grad = 0.0;
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				if (_finite(pow(grad[i][j][k], 2)) != 0) sum_grad += pow(grad[i][j][k], 2);
			}
		}
	}
	_gamma = pow(J, _beta) / sum_grad;
}

void LLGMN::set_eta(double J0)
{
	_eta = pow(J0, (1.0 - _beta)) / (double)(_time * (1.0 - _beta));
}

void LLGMN::talearn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label) // なんかおかしい
{
	int count = 0;
	int size = input_data.size();
	double J = 0;//評価関数
	auto O = make_v<double>(size, _H); //１層目出力用
	auto O_2 = make_v<double>(size, _K, _M); //２層目出力用
	auto Y = make_v<double>(size, _K); //３層目出力用
	auto grad = make_v<double>(_K, _M, _H); //重み更新 微分部用
	set_weight();//重みweight初期生成
	cout << "count" << "\tJ" << "\tAccuracy" << endl;
	while (count < (_time / _smp_time))
	{
		//初期化
		fill_v(O, 0);
		fill_v(O_2, 0);
		fill_v(Y, 0);
		
		for (int n = 0; n < size; n++)
		{
			//初期化
			J = 0;
			fill_v(grad, 0);
			//層計算
			forward(input_data[n], O_2[n], Y[n], O[n]);

			//評価関数
			for (int i = 0; i < _K; i++)
			{
				J += -input_label[n][i] * log(Y[n][i]);
			}
			//重み更新
			if (count == 0 && n == 0) set_eta(J);
			backward_online(input_label[n], O_2[n], Y[n], O[n], grad); //勾配計算部
			gamma_update(J, grad); 
			w_update(grad);
		}
		count++;
		//Accuracy
		CM.setPram(Y, input_label);
		cout << fixed << setprecision(5) << count << "\t" << J << "\t" << CM.getAccuracy() << endl;
		if (J < 1e-5) break;
	}	
	cout << endl;
}

void LLGMN::talearn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label)
{	
	int count = 0;
	int n, i;
	int size = input_data.size();
	double J = 0; //評価関数
	auto O = make_v<double>(size, _H); //１層目出力用
	auto O_2 = make_v<double>(size, _K, _M); //２層目出力用
	auto Y = make_v<double>(size, _K); //３層目出力用	
	auto grad = make_v<double>(_K, _M, _H); //重み更新 微分部用
	set_weight();//重みweight初期生成
	cout << "count" << "\tJ" << "\tAccuracy" << endl;
	while (count < (_time / _smp_time))
	{
		//初期化
		J = 0;
		fill_v(O, 0);
		fill_v(O_2, 0);
		fill_v(Y, 0);
		fill_v(grad, 0);

		for (n = 0; n < size; n++)
		{
			//層計算
			forward(input_data[n], O_2[n], Y[n], O[n]);
			//cout << Y[n][0] <<", "<< Y[n][1] << ", " << Y[n][2] << ", " << Y[n][3] << endl;
		}
		
		//評価関数
		for (n = 0; n < size; n++)
		{
			for (i = 0; i < _K; i++)
			{
				J += -input_label[n][i] * log(Y[n][i]) / size;
			}
		}
		//重み更新
		if (count == 0) set_eta(J);
		backward_batch(input_label, O_2, Y, O, grad); //勾配計算部
		//cout << _eta << " " << _gamma << endl;
		gamma_update(J, grad);
		w_update(grad);

		count++;
		//Accuracy
		CM.setPram(Y, input_label);
		cout << fixed << setprecision(5) << count << "\t" << J << "\t" << CM.getAccuracy() << endl;
	}
	cout << endl;
}