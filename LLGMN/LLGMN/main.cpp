#include "LLGMN.h"
#include "file.h"
#include <conio.h>

int main(void)
{
	int flag;
	bool ta = 1;
	//LL
	int D;				//���͎���
	int K;				//�N���X
	int M;				//�R���|�[�l���g��
	double epsilon;	//�w�K��
	//TA
	double beta;		//�w�K�p�����[�^��
	double time;		//�w�K����
	double eta;		// �]���֐��̏����l�ɂ���ǓI�Ȋw�K���̌���
	double gamma;	// �]���֐��̕ϓ��ɂ��Ǐ��I�Ȋw�K���̒���
	double smp;		//�T���v�����O���ԇ�t[s]

	cout << "0:LLGMN or 1:TA_LLGMN ? : ";
	cin >> ta;
	/*
	cout << "0:self set param or 1:auto set param ? : ";
	cin >> flag;
	while(!flag)
	{
		cout << "�p�����[�^�ݒ�" << endl;
		cout << "�R���|�[�l���g��(M) (def 2) : " << M << endl;
		cout << "�w�K��(��) (def 0.1) : " << epsilon << endl;
		cout<<"�w�K�p�����[�^(��) (def 0.7) : " << beta << endl;
		cout << "�w�K����(t) (def 1.0) : " << time << endl;
		cout << "�T���v�����O����(��t[s]) (def 0.001) : " << smp << endl;
	}*/

	
	/* ------data�ǂݍ���------ */
	vector<vector<double>> input_datas;
	vector<vector<double>> input_labels;

	//file�ǂݍ���
	//fileset("data/lea_sig.csv", input_datas);
	//fileset("data/lea_T_sig.csv", input_labels);
	fileset("data/input1.txt", input_datas);
	fileset("data/label1.txt", input_labels);
	cout << "���t�f�[�^�ǂݍ���OK" << endl;
	//for(auto X : input_datas) for (auto x : X) cout << x << endl;

	D = input_datas[0].size();//���͎���
	K = input_labels[0].size();//�N���X
	LLGMN LL(D, K);

	/* ------�w�K------ */
	while (true)
	{
		cout << "0:online or 1:batch ? : ";
		cin >> flag;
		cout << "--------------------------------------" << endl;
		if (!ta) cout << "LLGMN" << endl;
		else cout << "TA_LLGMN" << endl;
		cout << "--------------------------------------" << endl;
		//�w�K
		if (!flag)
		{
			cout << "�����w�K�@���t�f�[�^" << endl;
			if (!ta) LL.learn_online(input_datas, input_labels);
			else LL.talearn_online(input_datas, input_labels);
			break;
		}
		else if (flag)
		{
			cout << "�ꊇ�w�K�@���t�f�[�^" << endl;
			if (!ta) LL.learn_batch(input_datas, input_labels);
			else LL.talearn_batch(input_datas, input_labels);
			break;
		}
		else
		{
			cerr << "0 or 1" << endl;
		}
	}	
	/* ------�w�K�����܂�------ */

	/* ------�e�X�g------ */
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
		cout << "�e�X�g�f�[�^�ǂݍ���OK" << endl;

		//�e�X�g
		vector<vector<double>> output_datas;
		output_datas = LL.test(test_datas, test_labels);
		string sname = "data/out"+ to_string(n+2)+".csv";
		//filewrite(sname, output_datas);
		LL.CM.resultSave(sname);
	}
	return 0;
}