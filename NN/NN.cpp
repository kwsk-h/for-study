#include "NN.h"

NN::NN(int D, int K, int M, int Mk, double epsilon, int batch_size) {//Constructor
	//parameter set
	_D = D;
	_K = K;
	_M = M;
	_Mk = Mk;
	_epsilon = epsilon;
	_batch_size = batch_size;

	//weight vector set
	_weight.First = make_v<double>(_D, _M); 
	_weight.Mid = make_v<double>(_Mk, _M, _M);
	_weight.Last = make_v<double>(_M, _K);
	setWeight();

	//output vector set
	_output.First = make_v<double>(_D);	// = input_data
	_output.Mid = make_v<double>(_Mk, _M);
	_output.Last = make_v<double>(_K);
}

NN::~NN() {//Destructor

}

double NN::sigma(const vector<double>& V)
{
	double sum = 0;
	for (auto v : V)
	{
		sum += v;
	}
	return sum;
}

vector<double> NN::sigmoid(vector<double>& s)
{
	vector<double> y(s.size());
	for (int i = 0; i < s.size(); i++)
	{
		y[i] = 1 / (1 + exp(-1 * s[i]));
	}
	return y;
}

double NN::d_sigmaoid(double y)
{
	return y * (1 - y);
}

void NN::forward(const vector<double> input_data)
{
	_output.First = input_data;
	//“ü—Í‘w‚©‚ç’†ŠÔ‘w‚Ö
	vector<double> input_FirstLayer(_M, 0);
	for (int d = 0; d < _D; d++)
	{
		for (int m = 0; m < _M; m++)
		{
			input_FirstLayer[m] += _weight.First[d][m] * input_data[d];
		}
	}
	_output.Mid[0] = sigmoid(input_FirstLayer);

	//’†ŠÔ‘w‚©‚ç’†ŠÔ‘w‚Ö
	vector<vector<double>> input_MidLayer(_Mk - 1, vector < double>(_M, 0));
	for (int mk = 0; mk < _Mk - 1; mk++)
	{
		for (int m_bef = 0; m_bef < _M; m_bef++)
		{
			for (int m_aft = 0; m_aft < _M; m_aft++)
			{
				input_MidLayer[mk][m_aft] += _weight.Mid[mk][m_bef][m_aft] * _output.Mid[mk][m_bef];
			}
		}
		_output.Mid[mk] = sigmoid(input_MidLayer[mk]);
	}

	//’†ŠÔ‘w‚©‚ço—Í‘w‚Ö
	vector<double> input_LastLayer(_K, 0);
	for (int m = 0; m < _M; m++)
	{
		for (int k = 0; k < _K; k++)
		{
			input_LastLayer[k] += _weight.Last[m][k] * input_data[m];
		}
	}
	_output.Last = sigmoid(input_LastLayer);
}
