#include "NN.h"

random_device rnd;     // 非決定的な乱数生成器を生成
mt19937 mt(rnd());      // メルセンヌ・ツイスタの32ビット版、引数は初期シード値

//---------------------------
// Constructor
//---------------------------
NN::NN(int D, int C, int M, int Mk, double epsilon, int batch_size) {
	//parameter set
	_D = D;
	_C = C;
	_M = M;
	_Mk = Mk;
	_epsilon = epsilon;
	_batch_size = batch_size;

	_output = make_v<LAYER<double>>(_batch_size);
	for (int n = 0; n < _batch_size; n++) {
		//weight vector set
		_weight.First = make_v<double>(_D + 1, _M);
		_weight.Mid = make_v<double>(_Mk, _M + 1, _M);
		_weight.Last = make_v<double>(_M + 1, _C);
		setWeight();

		//output vector set
		_output[n].First = make_v<double>(_D);	// = input_data
		_output[n].Mid = make_v<double>(_Mk, _M);
		_output[n].Last = make_v<double>(_C);
	}
}

//---------------------------
// Destructor
//---------------------------
NN::~NN() {

}

//---------------------------
// Set init Weight
//---------------------------
void NN::setWeight() {
	uniform_real_distribution<> rand_abs1(-1, 1);        // [-1, 1] 範囲の一様乱数
	for (int n = 0; n < _batch_size; n++) {
		for (auto& V : _weight.First) for (auto& v : V) {
			v = rand_abs1(mt);
		}
		for (auto& VV : _weight.Mid) for (auto& V : VV) for (auto& v : V) {
			v = rand_abs1(mt);
		}
		for (auto& V : _weight.Last) for (auto& v : V) {
			v = rand_abs1(mt);
		}
	}
}

//---------------------------
// 1D vec -> double sum
//---------------------------
double NN::sigma(const vector<double>& V)
{
	double sum = 0;
	for (auto v : V)
	{
		sum += v;
	}
	return sum;
}

//---------------------------
// sigmoid y = 1 / (1 + exp(-x));
//---------------------------
vector<double> NN::sigmoid(vector<double>& s)
{
	vector<double> y(s.size());
	for (int i = 0; i < s.size(); i++)
	{
		y[i] = 1 / (1 + exp(-1 * s[i]));
	}
	return y;
}

//---------------------------
// differential sigmoid dy/dx = y(1-y)
//---------------------------
double NN::d_sigmaoid(double y)
{
	return y * (1 - y);
}

//---------------------------
// Forward calculation
//---------------------------
void NN::forward(const vector<double> input_data, int n)
{
	_output[n].First = input_data;
	//入力層から中間層へ
	vector<double> input_FirstLayer(_M, 0);
	for (int m = 0; m < _M; m++)
	{
		for (int d = 0; d < _D; d++)
		{
			input_FirstLayer[m] += _weight.First[d][m] * _output[n].First[d];
		}
		input_FirstLayer[m] += _weight.First[_D][m] * 1; // bias
	}
	_output[n].Mid[0] = sigmoid(input_FirstLayer);

	//中間層から中間層へ
	vector<vector<double>> input_MidLayer(_Mk - 1, vector < double>(_M, 0));
	for (int mk = 0; mk < _Mk - 1; mk++)
	{
		for (int m_aft = 0; m_aft < _M; m_aft++)
		{
			for (int m_bef = 0; m_bef < _M; m_bef++)
			{
				input_MidLayer[mk][m_aft] += _weight.Mid[mk][m_bef][m_aft] * _output[n].Mid[mk][m_bef];
			}
			input_MidLayer[mk][m_aft] += _weight.Mid[mk][_M][m_aft] * 1; // bias
		}
		_output[n].Mid[mk + 1] = sigmoid(input_MidLayer[mk]);
	}

	//中間層から出力層へ
	vector<double> input_LastLayer(_C, 0);
	for (int c = 0; c < _C; c++)
	{
		for (int m = 0; m < _M; m++)
		{
			input_LastLayer[c] += _weight.Last[m][c] * _output[n].Mid[_Mk - 1][m];
		}
		input_LastLayer[c] += _weight.Last[_M][c] * 1; // bias
	}
	_output[n].Last = sigmoid(input_LastLayer);
}

//---------------------------
// Back propagation_weight
//---------------------------
void NN::backward(const vector<double>& delta, int n) {
	vector<double> dLdY(_C, 0);
	auto dLdO = make_v<double>(_Mk, _M);
	fill_v(dLdO, 0);

	//出力層から中間層へ
	for (int c = 0; c < _C; c++)
	{
		dLdY[c] += 2 * delta[c];	//2(Y - t)
		for (int m = 0; m < _M; m++)
		{			
			_weight.Last[m][c] -= _epsilon * _output[n].Mid[_Mk - 1][m] * d_sigmaoid(_output[n].Last[c]) * dLdY[c];  // update weight
			dLdO[_Mk - 1][m] += _weight.Last[m][c] * d_sigmaoid(_output[n].Last[c]) * dLdY[c];
		}
		_weight.Last[_M][c] -= _epsilon * 1 * d_sigmaoid(_output[n].Last[c]) * dLdY[c]; // update bias weight
	}

	//中間層から中間層へ
	for (int mk = _Mk - 1; mk > 0; mk--)
	{
		for (int m_aft = 0; m_aft < _M; m_aft++)
		{
			for (int m_bef = 0; m_bef < _M; m_bef++)
			{
				_weight.Mid[mk - 1][m_bef][m_aft] -= _epsilon * _output[n].Mid[mk - 1][m_bef] * d_sigmaoid(_output[n].Mid[mk - 1][m_aft]) * dLdO[mk][m_aft]; // update weight
				dLdO[mk - 1][m_bef] += _weight.Mid[mk - 1][m_bef][m_aft] * d_sigmaoid(_output[n].Mid[mk - 1][m_aft]) * dLdO[mk][m_aft];
			}
			_weight.Mid[mk - 1][_M][m_aft] -= _epsilon * 1 * d_sigmaoid(_output[n].Mid[mk - 1][m_aft]) * dLdO[mk][m_aft]; // update bias weight
		}
	}

	//中間層から入力層へ
	for (int m = 0; m < _M; m++)
	{
		for (int d = 0; d < _D; d++)
		{
			_weight.First[d][m] -= _epsilon * _output[n].First[d] * d_sigmaoid(_output[n].Mid[0][m]) * dLdO[0][m];  // update weight
		}
		_weight.First[_D][m] -= _epsilon * 1 * d_sigmaoid(_output[n].Mid[0][m]) * dLdO[0][m]; // update bias weight
	}

}

//---------------------------
// minibatch Learning
//---------------------------
void NN::Learning(const vector<vector<double>> input_datas, const vector<vector<double>> input_labels) {
	int N = input_datas.size();
	uniform_int_distribution<> randN(0, N - 1);
	auto delta = make_v<double>(_C);
	double L = 999;
	int num = 0;
	ofstream ofs("Debug/training_log.csv");
	if (ofs.fail()) {	// ファイルオープンに失敗したらそこで終了
		cerr << "cannot open the file - 'trainLog.txt'" << endl;
		exit(1);
	}
	while (L > 0.0001 && num < 1e+5)
	{
		L = 0;
		num++;
		//minibatch
		for (int t = 0; t < N / _batch_size; t++)
		{
			fill_v(delta, 0.0);
			for (int i = 0; i < _batch_size; i++)
			{
				int n = randN(mt);
				//int n = i + _batch_size * t;
				forward(input_datas[n], i);

				//clac error
				for (int c = 0; c < _C; c++)
				{
					delta[c] += (_output[i].Last[c] - input_labels[n][c]) / _batch_size;
					L += pow(delta[c], 2) / _C;
				}
			}
			for (int i = 0; i < _batch_size; i++)
			{
				backward(delta, i);
			}
		}
		cout << num << " : " << L << endl;
		ofs << num << "," << L << endl;
	}ofs.close();
}
