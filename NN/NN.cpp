#include "NN.h"

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

	//weight vector set
	_weight.First = make_v<double>(_D + 1, _M);
	_weight.Mid = make_v<double>(_Mk, _M + 1, _M);
	_weight.Last = make_v<double>(_M + 1, _C);
	setWeight();

	//output vector set
	_output.First = make_v<double>(_D);	// = input_data
	_output.Mid = make_v<double>(_Mk, _M);
	_output.Last = make_v<double>(_C);
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
	random_device rnd;     // ”ñŒˆ’è“I‚È—”¶¬Ší‚ğ¶¬
	mt19937 mt(rnd());      // ƒƒ‹ƒZƒ“ƒkEƒcƒCƒXƒ^‚Ì32ƒrƒbƒg”ÅAˆø”‚Í‰ŠúƒV[ƒh’l
	uniform_real_distribution<> rand_abs1(-1, 1);        // [-1, 1] ”ÍˆÍ‚Ìˆê—l—”

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
void NN::forward(const vector<double> input_data)
{
	_output.First = input_data;
	//“ü—Í‘w‚©‚ç’†ŠÔ‘w‚Ö
	vector<double> input_FirstLayer(_M, 0);
	for (int m = 0; m < _M; m++)
	{
		for (int d = 0; d < _D; d++)
		{
			input_FirstLayer[m] += _weight.First[d][m] * _output.First[d];
		}
		input_FirstLayer[m] += _weight.First[_D][m] * 1; // bias
	}
	_output.Mid[0] = sigmoid(input_FirstLayer);

	//’†ŠÔ‘w‚©‚ç’†ŠÔ‘w‚Ö
	vector<vector<double>> input_MidLayer(_Mk - 1, vector < double>(_M, 0));
	for (int mk = 0; mk < _Mk - 1; mk++)
	{
		for (int m_aft = 0; m_aft < _M; m_aft++)
		{
			for (int m_bef = 0; m_bef < _M; m_bef++)
			{
				input_MidLayer[mk][m_aft] += _weight.Mid[mk][m_bef][m_aft] * _output.Mid[mk][m_bef];
			}
			input_MidLayer[mk][m_aft] += _weight.Mid[mk][_M][m_aft] * 1; // bias
		}
		_output.Mid[mk + 1] = sigmoid(input_MidLayer[mk]);
	}

	//’†ŠÔ‘w‚©‚ço—Í‘w‚Ö
	vector<double> input_LastLayer(_C, 0);
	for (int c = 0; c < _C; c++)
	{
		for (int m = 0; m < _M; m++)
		{
			input_LastLayer[c] += _weight.Last[m][c] * _output.Mid[_Mk - 1][m];
		}
		input_LastLayer[c] += _weight.Last[_M][c] * 1; // bias
	}
	_output.Last = sigmoid(input_LastLayer);
}

//---------------------------
// Back propagation
//---------------------------
void NN::backward(const vector<double>& delta) {
	double dLdY;
	auto dLdO = make_v<double>(_Mk, _M);

	//o—Í‘w‚©‚ç’†ŠÔ‘w‚Ö
	for (int c = 0; c < _C; c++)
	{
		dLdY = 2 * delta[c];	//2(Y - t)
		for (int m = 0; m < _M; m++)
		{			
			_weight.Last[m][c] -= _epsilon * _output.Mid[_Mk - 1][m] * d_sigmaoid(_output.Last[c]) * dLdY;  // update weight
			dLdO[_Mk - 1][m] += _weight.Last[m][c] * d_sigmaoid(_output.Last[c]) * dLdY;
		}
		_weight.Last[_M][c] -= _epsilon * 1 * d_sigmaoid(_output.Last[c]) * dLdY; // update bias weight
	}

	//’†ŠÔ‘w‚©‚ç’†ŠÔ‘w‚Ö
	for (int mk = _Mk - 1; mk > 0; mk--)
	{
		for (int m_aft = 0; m_aft < _M; m_aft++)
		{
			for (int m_bef = 0; m_bef < _M; m_bef++)
			{
				_weight.Mid[mk - 1][m_bef][m_aft] -= _epsilon * _output.Mid[mk - 1][m_bef] * d_sigmaoid(_output.Mid[mk - 1][m_aft]) * dLdO[mk][m_aft]; // update weight
				dLdO[mk - 1][m_bef] += _weight.Mid[mk - 1][m_bef][m_aft] * d_sigmaoid(_output.Mid[mk - 1][m_aft]) * dLdO[mk][m_aft];
			}
			_weight.Mid[mk - 1][_M][m_aft] -= _epsilon * 1 * d_sigmaoid(_output.Mid[mk - 1][m_aft]) * dLdO[mk][m_aft]; // update bias weight
		}
	}

	//’†ŠÔ‘w‚©‚ç“ü—Í‘w‚Ö
	for (int m = 0; m < _M; m++)
	{
		for (int d = 0; d < _D; d++)
		{
			_weight.First[d][m] -= _epsilon * _output.First[d] * d_sigmaoid(_output.Mid[0][m]) * dLdO[0][m];  // update weight
		}
		_weight.First[_D][m] -= _epsilon * 1 * d_sigmaoid(_output.Mid[0][m]) * dLdO[0][m]; // update bias weight
	}

}

void NN::Learning() {

}
