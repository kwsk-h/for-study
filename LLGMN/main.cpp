#include "LLGMN.h"
#include "file.h"

int main(void)
{
	cout << "LLGMN" << endl;
	LLGMN llgmn;
	int flag;

	vector<vector<double>> input_datas;
	vector<vector<double>> input_labels;
	vector<vector<double>> test_datas;
	vector<vector<double>> test_labels;

	//file�ǂݍ���
	fileset("data/lea_sig.csv", input_datas, "data");
	fileset("data/lea_T_sig.csv", input_labels, "lavel");
	fileset("data/dis_sig.csv", test_datas, "data");
	fileset("data/dis_T_sig.csv", test_labels, "lavel");
	cout << "file�ǂݍ���OK" << endl;

	//�p�����[�^�ݒ�
	cout << "�p�����[�^�ݒ�" << endl;
	/*cout << "���͎���(D) : ";
	cin >> llgmn._D;
	cout << "�N���X��(K) : ";
	cin >> llgmn._K;
	cout << "�R���|�[�l���g��(M) : ";
	cin >> llgmn._M;
	cout << "�w�K��(��) : ";
	cin >> llgmn._epsilon;*/

	//�d��weight��������
	auto weight = make_v<double>(llgmn._K, llgmn._M, llgmn._H);
	llgmn.set_weight(weight);
	llgmn._weight = weight;
	while (true)
	{
		cout << "0:online or 1:batch ? : ";
		cin >> flag;
		//�w�K
		if (!flag)
		{
			cout << "�����w�K�@���t�f�[�^" << endl;
			llgmn.learn_online(input_datas, input_labels);
			break;
		}
		else if (flag)
		{
			cout << "�ꊇ�w�K�@���t�f�[�^" << endl;
			llgmn.learn_batch(input_datas, input_labels);
			break;
		}
		else
		{
			cerr << "0 or 1" << endl;
		}
	}	

	//�e�X�g
	llgmn.test(test_datas, test_labels);

	return 0;
}