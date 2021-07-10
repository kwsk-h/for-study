#include "LLGMN.h"
#include "file.h"

int main(void)
{
	int flag;
	int ta = 1;
	cout << "0:LLGMN or 1:TA_LLGMN ? : ";
	cin >> ta;
	LLGMN LL;
	
	
	/* ------学習------ */

	vector<vector<double>> input_datas;
	vector<vector<double>> input_labels;

	//file読み込み
	fileset(/*"data/input1.txt"*/"data/lea_sig.csv", input_datas);
	fileset(/*"data/label1.txt"*/"data/lea_T_sig.csv", input_labels);
	cout << "教師データ読み込みOK" << endl;

	//パラメータ設定
	/*
	cout << "パラメータ設定" << endl;
	cout << "入力次元(D) : ";
	cin >> LL._D;
	cout << "クラス数(K) : ";
	cin >> LL._K;
	cout << "コンポーネント数(M) : ";
	cin >> LL._M;
	cout << "学習率(ε) : ";
	cin >> LL._epsilon;
	*/

	while (true)
	{
		cout << "0:online or 1:batch ? : ";
		cin >> flag;
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
	for (int n = 0; n < 1; n++)
	{
		vector<vector<double>> test_datas;
		vector<vector<double>> test_labels;
		//cout << testdataset[n] << endl;
		fileset(/*testdataset[n]*/"data/dis_sig.csv", test_datas);
		fileset(/*testlabelset[n]*/"data/dis_T_sig.csv", test_labels);
		cout << "テストデータ読み込みOK" << endl;

		//テスト
		LL.test(test_datas, test_labels);
	}
	return 0;
}