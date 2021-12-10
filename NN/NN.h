#pragma once

#include<stdio.h>
#include<iostream>
#include<vector>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include <fstream>
#include <string>
#include <random>

using namespace std;

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

class NN
{
public:
	int _D;				//入力次元
	int _C;				//クラス数
	int _M;				//中間層の要素数
	int _Mk;			//中間層の層数
	double _epsilon;	//学習率
	int _batch_size; //batchサイズの指定

	//各層の重みや各層の出力に利用する構造体
	template <typename T>
	struct LAYER {
		vector<T> First;				//入力層から中間層へ
		vector<vector<T>> Mid;	//中間層から中間層へ
		vector<T> Last;				//中間層から出力層へ
	};

	LAYER<vector<double>> _weight;	//各層の重み
	LAYER<double> _output;					//各層の出力

	//Constructor
	NN(int D, int C, int M, int Mk, double epsilon, int batch_size);//Parameter setting
	//Destructor
	~NN();

	//重みの初期値設定
	void setWeight();

	//vector sum
	double sigma(const vector<double>& V);

	//return 1/(1+exp(-s))
	vector<double> sigmoid(vector<double>& s);

	//return y(1-y)
	double d_sigmaoid(double y);

	//前向き計算　input_data：入力データ
	void forward(const vector<double> input_data);

	//誤差逆伝搬 重み更新　delta：誤差
	void backward(const vector<double>& delta);

	//学習
	void Learning();
};