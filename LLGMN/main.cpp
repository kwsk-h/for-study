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
	int _D = 2;//���͎���
	int _H = 1 + _D * (_D + 3) / 2;//����`�ϊ���̎���
	int _K = 4;//�N���X
	int _M = 2;//�R���|�[�l���g��
	double _epsilon = 0.1;//�w�K��
	vector<vector<vector<double>>> _weight; //�d�ݗp�@weight[K][M][H]
	void input_conversion(const vector<double>& input_data, vector<double>& converted_input); //����`�ϊ��p
	void set_weight(vector<vector<vector<double>>>& weight); //�d�݂̏����l�ݒ�p
	void forward(vector<double>& input_data,vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O); //�������v�Z�p
	void backward_online(vector<double>input_label, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O, vector<vector<vector<double>>>& grad); //�t�����v�Z�C�����w�K
	void backward_batch(vector<vector<double>>input_label, vector<vector<vector<double>>>& O_2, vector<vector<double>>& Y, vector<vector<double>>& O, vector<vector<vector<double>>>& grad); //�t�����v�Z�C�ꊇ�w�K
	void weight_update(vector<vector<vector<double>>>& grad); //�t�����v�Z�C�d�ݍX�V�����p
	void learn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�����w�K�p
	void learn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label); //�ꊇ�w�K�p
	void test(vector<vector<double>>& test_data, vector<vector<double>>& test_label); //�e�X�g�p
	void fileset(string filename, vector<vector<double>>& datas, string data_type); //�t�@�C���ǂݍ��ݗp
};


//�������z��m�ۋy�я������p�}�N��
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
//�}�N�������܂�

void LLGMN::input_conversion(const vector<double>& input_data, vector<double>& converted_input)
{//����`�ϊ�(/1 sample)
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
	//�d�݂�0�̂Ƃ����0�ɂ���
	for (int k = 0; k < _H; k++) 
	{
		weight[_K - 1][_M - 1][k] = 0;
	}
}

void LLGMN::forward(vector<double>& input_data, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O)
{//�������v�Z�i/1 sample�j
	vector<double> converted_input(_H); //�P�w�ړ��͗p�i����`�ϊ��p�j
	input_conversion(input_data, converted_input);//����`�ϊ�
	vector<vector<double>> I(_K, vector<double>(_M, 0)); //�Q�w�ړ��͗p
	double I_sum = 0;
	for (int i = 0; i < _H; i++)
	{//1�w�ڂ̏o��
		O[i] = converted_input[i];
	}
	
	for (int i = 0; i < _K; i++) 
	{//2�w�ڂ̓���
		for (int j = 0; j < _M; j++) 
		{
			for (int k = 0; k < _H; k++) 
			{
				I[i][j] += O[k] * _weight[i][j][k];
			}
		}
	}

	for (int i = 0; i < _K; i++)
	{//2�w�ڂ̏o��
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
	{//3�w��
		for (int j = 0; j < _M; j++)
		{
			Y[i] += O_2[i][j];
		}
	}
}

//�d�ݍX�V�̊֐�
void LLGMN::backward_online(vector<double>input_label, vector<vector<double>>& O_2, vector<double>& Y, vector<double>& O, vector<vector<vector<double>>>& grad)
{//���z�����i�����̂Ƃ���j�̌v�Z
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				grad[i][j][k] = (Y[i] - input_label[i]) * (O_2[i][j] / Y[i]) * O[k];
			}
		}
	}
	//�d�݂��X�V
	weight_update(grad);
}
void LLGMN::backward_batch(vector<vector<double>>input_label, vector<vector<vector<double>>>& O_2, vector<vector<double>>& Y, vector<vector<double>>& O, vector<vector<vector<double>>>& grad)
{//���z�����i�����̂Ƃ���j�̌v�Z
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
	//�d�݂��X�V
	weight_update(grad);
}

void LLGMN::weight_update(vector<vector<vector<double>>>& grad)
{//�d�݂��X�V
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
	double J = 0;//�]���֐�
	auto O = make_v<double>(size, _H); //�P�w�ڏo�͗p
	auto O_2 = make_v<double>(size, _K, _M); //�Q�w�ڏo�͗p
	auto Y = make_v<double>(size, _K); //�R�w�ڏo�͗p
	auto grad = make_v<double>(_K, _M, _H); //�d�ݍX�V �������p

	while (count < 1000)
	{
		//������
		J = 0;
		fill_v(O, 0);
		fill_v(O_2, 0);
		fill_v(Y, 0);

		for (int n = 0; n < size; n++)
		{
			//������
			fill_v(grad, 0);

			//�w�v�Z
			forward(input_data[n],  O_2[n],  Y[n],  O[n]);

			//�]���֐�
			for (int i = 0; i < _K; i++)
			{
				J += -input_label[n][i] * log(Y[n][i]);
			}
			
			//�d�ݍX�V
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
	double J = 0; //�]���֐�
	auto O = make_v<double>(size, _H); //�P�w�ڏo�͗p
	auto O_2 = make_v<double>(size, _K, _M); //�Q�w�ڏo�͗p
	auto Y = make_v<double>(size, _K); //�R�w�ڏo�͗p	
	auto grad = make_v<double>(_K, _M, _H); //�d�ݍX�V �������p
	while (count < 1000)
	{
		//������
		J = 0;
		fill_v(O, 0);
		fill_v(O_2, 0);
		fill_v(Y, 0);
		fill_v(grad, 0);

		for (n = 0; n < size; n++)
		{
			//�w�v�Z
			forward(input_data[n],O_2[n], Y[n], O[n]);
			//cout << Y[n][0] <<", "<< Y[n][1] << ", " << Y[n][2] << ", " << Y[n][3] << endl;
		}

		//�]���֐�
		for (n = 0; n < size; n++)
		{
			for (i = 0; i < _K; i++)
			{
				J += -input_label[n][i] * log(Y[n][i]);
			}
		}

		//�d�ݍX�V
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
	double J = 0; //�]���֐�
	auto O_2 = make_v<double>(size, _K, _M);
	auto Y = make_v<double>(size, _K);
	auto O = make_v<double>(size, _H);
	//������
	fill_v(O_2, 0);
	fill_v(Y, 0);

	for (n = 0; n < size; n++)
	{
		forward(test_data[n], O_2[n], Y[n], O[n]);//�w�v�Z
	}

	//�]���֐�
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
	cout << "���ʗ� = " << (double)count / (double)size << endl;
}


void LLGMN::fileset(string filename, vector<vector<double>>& datas, string data_type)
{
	ifstream file(filename);
	if (file.fail()) {	// �t�@�C���I�[�v���Ɏ��s�����炻���ŏI��
		cerr << "cannot open the file - '" << filename << "'" << endl;
		exit(1);
	}
	while (!file.eof())
	{//��������getlines�̂ق������͎����ɂ��Ȃ��Ă悳������������
		vector<double> inner;
		double ch1, ch2, ch3, ch4;
		char split;//�u,�v�̋�ǂݗp
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

	//file�ǂݍ���
	cout << "file�ǂݍ���" << endl;
	llgmn.fileset("data/lea_sig.csv", input_datas, "data");
	llgmn.fileset("data/lea_T_sig.csv", input_labels, "lavel");
	llgmn.fileset("data/dis_sig.csv", test_datas, "data");
	llgmn.fileset("data/dis_T_sig.csv", test_labels, "lavel");

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