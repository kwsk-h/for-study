#include "LLGMN.h"

LLGMN::LLGMN(int D, int K, int M, double epsilon, double beta, double time, double smp_time)	//コンストラクタ
{
	_D = D;								//入力次元
	_H = 1 + D * (D + 3) / 2;		//非線形変換後の次元
	_K = K;								//クラス
	_M = M;								//コンポーネント数
	_epsilon = epsilon;					//学習率
	_weight = make_v<double>(_K, _M, _H); //重み用　weight[K][M][H]
	//TA
	_beta = beta;							//学習パラメータβ
	_time = time;							//学習時間
	_smp_time = smp_time;			//サンプリング時間⊿t[s]
}

LLGMN::~LLGMN()	//デストラクタ
{

}

void LLGMN::input_conversion(const vector<double>& input_data, vector<double>& converted_input)
{//非線形変換(/1 sample)
	int num = 0;
	converted_input[num] = 1;
	for (int i = 0; i < _D; i++)
	{
		num++;
		converted_input[num] = input_data[i];
	}
	for (int i = 0; i < _D; i++)
	{
		for (int j = i; j < _D; j++)
		{
			num++;
			converted_input[num] = input_data[i] * input_data[j];
		}
	}
}

void LLGMN::set_weight(void)
{
	srand(time(NULL));
	for (int i = 0; i < _K; i++)
	{
		for (int j = 0; j < _M; j++)
		{
			for (int k = 0; k < _H; k++)
			{
				_weight[i][j][k] = ((double)rand() / RAND_MAX) * 2 - 1;
			}
		}
	}
	//重みが0のところを0にする
	for (int k = 0; k < _H; k++)
	{
		_weight[_K - 1][_M - 1][k] = 0;
	}
}

void LLGMN::forward(vector<double>& input_data, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O)
{//順方向計算（/1 sample）
	vector<double> converted_input(_H); //１層目入力用（非線形変換用）
	input_conversion(input_data, converted_input);//非線形変換
	vector<vector<double>> I(_K, vector<double>(_M, 0)); //２層目入力用
	double I_sum = 0;
	for (int i = 0; i < _H; i++)
	{//1層目の出力
		O[i] = converted_input[i];
	}

	for (int i = 0; i < _K; i++)
	{//2層目の入力
		for (int j = 0; j < _M; j++)
		{
			for (int k = 0; k < _H; k++)
			{
				I[i][j] += O[k] * _weight[i][j][k];
			}
		}
	}

	for (int i = 0; i < _K; i++)
	{//2層目の出力
		for (int j = 0; j < _M; j++)
		{
			I_sum += exp(I[i][j]);
		}
	}
	for (int i = 0; i < _K; i++) {
		for (int j = 0; j < _M; j++) {
			O_2[i][j] = exp(I[i][j]) / I_sum;
		}
	}

	for (int i = 0; i < _K; i++)
	{//3層目
		for (int j = 0; j < _M; j++)
		{
			Y[i] += O_2[i][j];
		}
	}
}

//重み更新の関数
void LLGMN::backward_online(vector<double>input_label, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O, vector<vector<vector<double>>>& grad)
{//勾配部分（微分のところ）の計算
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				grad[i][j][k] = (Y[i] - input_label[i]) * (O_2[i][j] / Y[i]) * O[k];
			}
		}
	}
}
void LLGMN::backward_batch(vector<vector<double>>input_label, vector<vector<vector<double>>>& O_2, vector<vector<double>>& Y, vector<vector<double>>& O, vector<vector<vector<double>>>& grad)
{//勾配部分（微分のところ）の計算
	int size = input_label.size();
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				for (int n = 0; n < size; n++) {
					grad[i][j][k] += (Y[n][i] - input_label[n][i]) * (O_2[n][i][j] / Y[n][i]) * O[n][k] / size;
				}
			}
		}
	}
}

void LLGMN::weight_update(vector<vector<vector<double>>>& grad)
{//重みを更新
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				_weight[i][j][k] -= _epsilon * grad[i][j][k];
			}
		}
	}
}


void LLGMN::learn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label)
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
		J = 0;
		fill_v(O, 0);
		fill_v(O_2, 0);
		fill_v(Y, 0);

		for (int n = 0; n < size; n++)
		{
			//初期化
			fill_v(grad, 0);
			
			//層計算
			forward(input_data[n], O_2[n], Y[n], O[n]);

			//評価関数
			for (int i = 0; i < _K; i++)
			{
				J += -input_label[n][i] * log(Y[n][i]);
			}

			//重み更新
			backward_online(input_label[n], O_2[n], Y[n], O[n], grad);
			weight_update(grad);
		}

		count++;
		//Accuracy
		CM.setPram(Y, input_label);
		cout << fixed << setprecision(5) << count << "\t" << J << "\t" << CM.getAccuracy() << endl;
	}
	cout << endl;
}

void LLGMN::learn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label)
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
	cout << "count" <<  "\tJ" << "\tAccuracy" << endl;
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
				J += -input_label[n][i] * log(Y[n][i]);
			}
		}

		//重み更新
		backward_batch(input_label, O_2, Y, O, grad);
		weight_update(grad);

		count++;
		//Accuracy
		CM.setPram(Y, input_label);
		cout << fixed << setprecision(5) << count << "\t" << J << "\t" << CM.getAccuracy() << endl;
	}
	cout << endl;
}

vector<vector<double>> LLGMN::test(vector<vector<double>>& test_data, vector<vector<double>>& test_label)
{
	int count = 0;
	int jadge;
	int n, i;
	int size = test_data.size();
	double J = 0; //評価関数
	auto O_2 = make_v<double>(size, _K, _M);
	auto Y = make_v<double>(size, _K);
	auto O = make_v<double>(size, _H);
	//初期化
	fill_v(O_2, 0);
	fill_v(Y, 0);

	for (n = 0; n < size; n++)
	{
		forward(test_data[n], O_2[n], Y[n], O[n]);//層計算
	}

	//評価関数
	for (n = 0; n < size; n++)
	{
		for (i = 0; i < _K; i++)
		{
			J += -test_label[n][i] * log(Y[n][i]);
			//cout << Y[n][i] << "\t";
		}
		//cout << endl;
	}
	cout << "J = " << J << endl;
	//判別結果
	//Accuracy
	CM.setPram(Y, test_label);
	cout << "識別率 = " << CM.getAccuracy() << "\n" << endl;

	return Y;
}