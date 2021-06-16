#include "LLGMN.h"
#include "file.h"

int main(void)
{
	cout << "LLGMN" << endl;
	LLGMN llgmn;
	int flag;
	
	/* ------学習------ */

	vector<vector<double>> input_datas;
	vector<vector<double>> input_labels;

	//file読み込み
	fileset("data/input1.txt"/*"data/lea_sig.csv"*/, input_datas);
	fileset("data/label1.txt"/*"data/lea_T_sig.csv"*/, input_labels);
	cout << "教師データ読み込みOK" << endl;

	//パラメータ設定
	/*
	cout << "パラメータ設定" << endl;
	cout << "入力次元(D) : ";
	cin >> llgmn._D;
	cout << "クラス数(K) : ";
	cin >> llgmn._K;
	cout << "コンポーネント数(M) : ";
	cin >> llgmn._M;
	cout << "学習率(ε) : ";
	cin >> llgmn._epsilon;
	*/
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
	/* ------学習ここまで------ */

	/* ------テスト------ */
	vector<string> testdataset = { "data/input2.txt" ,"data/input3.txt" ,"data/input4.txt" };
	vector<string> testlabelset = { "data/label2.txt" ,"data/label3.txt" ,"data/label4.txt" };
	for (int n = 0; n < 3; n++)
	{
		vector<vector<double>> test_datas;
		vector<vector<double>> test_labels;
		cerr << testdataset[n] << endl;
		fileset(testdataset[n]/*"data/dis_sig.csv"*/, test_datas);
		fileset(testlabelset[n]/*"data/dis_T_sig.csv"*/, test_labels);
		cout << "テストデータ読み込みOK" << "\n" << endl;

		//テスト
		llgmn.test(test_datas, test_labels);
	}
	return 0;
}