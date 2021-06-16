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
	int _D = 2;//入力次元
	int _H = 1 + _D * (_D + 3) / 2;//非線形変換後の次元
	int _K = 4;//クラス
	int _M = 2;//コンポーネント数
	double _epsilon = 0.1;//学習率
	vector<vector<vector<double>>> _weight; //重み用　weight[K][M][H]
	void input_conversion(const vector<double>& input_data, vector<double>& converted_input); //非線形変換用
	void set_weight(vector<vector<vector<double>>>& weight); //重みの初期値設定用
	void forward(vector<double>& input_data,vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O); //順方向計算用
	void backward_online(vector<double>input_label, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O, vector<vector<vector<double>>>& grad); //逆方向計算，逐次学習
	void backward_batch(vector<vector<double>>input_label, vector<vector<vector<double>>>& O_2, vector<vector<double>>& Y, vector<vector<double>>& O, vector<vector<vector<double>>>& grad); //逆方向計算，一括学習
	void weight_update(vector<vector<vector<double>>>& grad); //逆方向計算，重み更新部分用
	void learn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //逐次学習用
	void learn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //一括学習用
	void test(vector<vector<double>>& test_data, vector<vector<double>>& test_label); //テスト用
	void fileset(string filename, vector<vector<double>>& datas, string data_type); //ファイル読み込み用
};


//多次元配列確保及び初期化用マクロ
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
//マクロここまで

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
			converted_input[num] = input_data[i]* input_data[j];
		}
	}
}

void LLGMN::set_weight(vector<vector<vector<double>>> &weight)
{
	srand(time(NULL));
	for (int i = 0; i < _K; i++) 
	{
		for (int j = 0; j < _M; j++) 
		{
			for (int k = 0; k < _H; k++) 
			{
				weight[i][j][k] = ((double)rand() / RAND_MAX) * 2 - 1;
			}
		}
	}
	//重みが0のところを0にする
	for (int k = 0; k < _H; k++) 
	{
		weight[_K - 1][_M - 1][k] = 0;
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
		for (int  j = 0; j < _M; j++) {
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
	//重みを更新
	weight_update(grad);
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
	//重みを更新
	weight_update(grad);
}

void LLGMN::weight_update(vector<vector<vector<double>>>& grad)
{//重みを更新
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				_weight[i][j][k] = _weight[i][j][k] - _epsilon * grad[i][j][k];
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

	while (count < 1000)
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
			forward(input_data[n],  O_2[n],  Y[n],  O[n]);

			//評価関数
			for (int i = 0; i < _K; i++)
			{
				J += -input_label[n][i] * log(Y[n][i]);
			}
			
			//重み更新
			backward_online(input_label[n], O_2[n],  Y[n],  O[n],  grad);
		}

		count++;
		cout << "count = " << count << "\tJ = " << J << endl;
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
	while (count < 1000)
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
			forward(input_data[n],O_2[n], Y[n], O[n]);
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

		count++;
		cout << "count = " << count << "\tJ = " << J << endl;
	}
	cout << endl;
}

void LLGMN::test(vector<vector<double>>& test_data, vector<vector<double>>& test_label)
{
	int count = 0;
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
			cout << Y[n][i] << "\t";
		}
		cout << endl;
	}

	for (n = 0;n < 200;n++) {
		if ((Y[n][0] > Y[n][1]) && (Y[n][0] > Y[n][2]) && (Y[n][0] > Y[n][3])) { count++; }
	}
	for (n = 200;n < 400;n++) {
		if ((Y[n][1] > Y[n][0]) && (Y[n][1] > Y[n][2]) && (Y[n][1] > Y[n][3])) { count++; }
	}
	for (n = 400;n < 600;n++) {
		if ((Y[n][2] > Y[n][0]) && (Y[n][2] > Y[n][1]) && (Y[n][2] > Y[n][3])) { count++; }
	}
	for (n = 600;n < 800;n++) {
		if ((Y[n][3] > Y[n][0]) && (Y[n][3] > Y[n][1]) && (Y[n][3] > Y[n][2])) { count++; }
	}
	cout << "識別率 = " << (double)count / (double)size << endl;
}


void LLGMN::fileset(string filename, vector<vector<double>>& datas, string data_type)
{
	ifstream file(filename);
	if (file.fail()) {	// ファイルオープンに失敗したらそこで終了
		cerr << "cannot open the file - '" << filename << "'" << endl;
		exit(1);
	}
	while (!file.eof())
	{//↓↓↓↓getlinesのほうが入力次元によらなくてよさそう↓↓↓↓
		vector<double> inner;
		double ch1, ch2, ch3, ch4;
		char split;//「,」の空読み用
		if (data_type == "data")
		{
			file >> ch1 >> split >> ch2;
			inner.push_back(ch1);
			inner.push_back(ch2);
		}
		if (data_type == "lavel")
		{
			file >> ch1 >> split >> ch2 >> split >> ch3 >> split >> ch4;
			inner.push_back(ch1);
			inner.push_back(ch2);
			inner.push_back(ch3);
			inner.push_back(ch4);
		}
		datas.push_back(inner);
	}
}

int main(void)
{
	cout << "LLGMN" << endl;
	LLGMN llgmn;
	int flag;

	vector<vector<double>> input_datas;
	vector<vector<double>> input_labels;
	vector<vector<double>> test_datas;
	vector<vector<double>> test_labels;

	//file読み込み
	cout << "file読み込み" << endl;
	llgmn.fileset("data/lea_sig.csv", input_datas, "data");
	llgmn.fileset("data/lea_T_sig.csv", input_labels, "lavel");
	llgmn.fileset("data/dis_sig.csv", test_datas, "data");
	llgmn.fileset("data/dis_T_sig.csv", test_labels, "lavel");

	//パラメータ設定
	cout << "パラメータ設定" << endl;
	/*cout << "入力次元(D) : ";
	cin >> llgmn._D;
	cout << "クラス数(K) : ";
	cin >> llgmn._K;
	cout << "コンポーネント数(M) : ";
	cin >> llgmn._M;
	cout << "学習率(ε) : ";
	cin >> llgmn._epsilon;*/

	//重みweight初期生成
	auto weight = make_v<double>(llgmn._K, llgmn._M, llgmn._H);
	llgmn.set_weight(weight);
	llgmn._weight = weight;
	while (true)
	{
		cout << "0:online or 1:batch ? : ";
		cin >> flag;
		//学習
		if (!flag)
		{
			cout << "逐次学習　教師データ" << endl;
			llgmn.learn_online(input_datas, input_labels);
			break;
		}
		else if (flag)
		{
			cout << "一括学習　教師データ" << endl;
			llgmn.learn_batch(input_datas, input_labels);
			break;
		}
		else
		{
			cerr << "0 or 1" << endl;
		}
	}	

	//テスト
	llgmn.test(test_datas, test_labels);

	return 0;
}