#include "LLGMN.h"
#include "file.h"
#include <conio.h>

int main(void)
{
	int flag;
	bool ta = 1;
	//LL
	int D;				//入力次元
	int K;				//クラス
	int M;				//コンポーネント数
	double epsilon;	//学習率
	//TA
	double beta;		//学習パラメータβ
	double time;		//学習時間
	double eta;		// 評価関数の初期値による大局的な学習率の決定
	double gamma;	// 評価関数の変動による局所的な学習率の調整
	double smp;		//サンプリング時間⊿t[s]

	cout << "0:LLGMN or 1:TA_LLGMN ? : ";
	cin >> ta;
	/*
	cout << "0:self set param or 1:auto set param ? : ";
	cin >> flag;
	while(!flag)
	{
		cout << "パラメータ設定" << endl;
		cout << "コンポーネント数(M) (def 2) : " << M << endl;
		cout << "学習率(ε) (def 0.1) : " << epsilon << endl;
		cout<<"学習パラメータ(β) (def 0.7) : " << beta << endl;
		cout << "学習時間(t) (def 1.0) : " << time << endl;
		cout << "サンプリング時間(⊿t[s]) (def 0.001) : " << smp << endl;
	}*/

	
	/* ------data読み込み------ */
	vector<vector<double>> input_datas;
	vector<vector<double>> input_labels;

	//file読み込み
	//fileset("data/lea_sig.csv", input_datas);
	//fileset("data/lea_T_sig.csv", input_labels);
	fileset("data/input1.txt", input_datas);
	fileset("data/label1.txt", input_labels);
	cout << "教師データ読み込みOK" << endl;
	//for(auto X : input_datas) for (auto x : X) cout << x << endl;

	D = input_datas[0].size();//入力次元
	K = input_labels[0].size();//クラス
	LLGMN LL(D, K);

	/* ------学習------ */
	while (true)
	{
		cout << "0:online or 1:batch ? : ";
		cin >> flag;
		cout << "--------------------------------------" << endl;
		if (!ta) cout << "LLGMN" << endl;
		else cout << "TA_LLGMN" << endl;
		cout << "--------------------------------------" << endl;
		//学習
		if (!flag)
		{
			cout << "逐次学習　教師データ" << endl;
			if (!ta) LL.learn_online(input_datas, input_labels);
			else LL.talearn_online(input_datas, input_labels);
			break;
		}
		else if (flag)
		{
			cout << "一括学習　教師データ" << endl;
			if (!ta) LL.learn_batch(input_datas, input_labels);
			else LL.talearn_batch(input_datas, input_labels);
			break;
		}
		else
		{
			cerr << "0 or 1" << endl;
		}
	}	
	/* ------学習ここまで------ */

	/* ------テスト------ */
	vector<string> testdataset = { "data/input2.txt" ,"data/input3.txt" ,"data/input4.txt" };
	vector<string> testlabelset = { "data/label2.txt" ,"data/label3.txt" ,"data/label4.txt" };
	for (int n = 0; n < 3; n++)
	{
		vector<vector<double>> test_datas;
		vector<vector<double>> test_labels;
		//fileset("data/dis_sig.csv", test_datas);
		//fileset("data/dis_T_sig.csv", test_labels);
		cout << testdataset[n] << endl;
		fileset(testdataset[n], test_datas);
		fileset(testlabelset[n], test_labels);
		cout << "テストデータ読み込みOK" << endl;

		//テスト
		vector<vector<double>> output_datas;
		output_datas = LL.test(test_datas, test_labels);
		string sname = "data/out"+ to_string(n+2)+".csv";
		//filewrite(sname, output_datas);
		LL.CM.resultSave(sname);
	}
	return 0;
}