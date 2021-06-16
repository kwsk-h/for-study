#pragma once
/*
LLGMN program

逐次学習，一括学習実装

―――――初期設定―――――
入力次元数 D					　2
クラス数 K						　4
コンポーネント数 M	　2
学習率 ε								0.1
――――――――――――――
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
	int _D = 2;//入力次元
	int _H = 1 + _D * (_D + 3) / 2;//非線形変換後の次元
	int _K = 4;//クラス
	int _M = 2;//コンポーネント数
	double _epsilon = 0.1;//学習率
	vector<vector<vector<double>>> _weight; //重み用　weight[K][M][H]
	void input_conversion(const vector<double>& input_data, vector<double>& converted_input); //非線形変換用
	void set_weight(vector<vector<vector<double>>>& weight); //重みの初期値設定用
	void forward(vector<double>& input_data, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O); //順方向計算用
	void backward_online(vector<double>input_label, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O, vector<vector<vector<double>>>& grad); //逆方向計算，逐次学習
	void backward_batch(vector<vector<double>>input_label, vector<vector<vector<double>>>& O_2, vector<vector<double>>& Y, vector<vector<double>>& O, vector<vector<vector<double>>>& grad); //逆方向計算，一括学習
	void weight_update(vector<vector<vector<double>>>& grad); //逆方向計算，重み更新部分用
	void learn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //逐次学習用
	void learn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //一括学習用
	void test(vector<vector<double>>& test_data, vector<vector<double>>& test_label); //テスト用
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