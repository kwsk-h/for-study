#include "LLGMN.h"


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
			converted_input[num] = input_data[i] * input_data[j];
		}
	}
}

void LLGMN::set_weight(vector<vector<vector<double>>>& weight)
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
		for (int j = 0; j < _M; j++) {
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
			forward(input_data[n], O_2[n], Y[n], O[n]);

			//�]���֐�
			for (int i = 0; i < _K; i++)
			{
				J += -input_label[n][i] * log(Y[n][i]);
			}

			//�d�ݍX�V
			backward_online(input_label[n], O_2[n], Y[n], O[n], grad);
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
			forward(input_data[n], O_2[n], Y[n], O[n]);
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
	int jadge;
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
			//cout << Y[n][i] << "\t";
		}
		//cout << endl;
	}
	cout << "J = " << J << endl;

	//���ʌ���
	for (n = 0; n < size; n++) {
		//�ŏI�w�o�͂��ő�̃N���X
		if ((Y[n][0] > Y[n][1]) && (Y[n][0] > Y[n][2]) && (Y[n][0] > Y[n][3])) { jadge = 0; }	
		else if ((Y[n][1] > Y[n][0]) && (Y[n][1] > Y[n][2]) && (Y[n][1] > Y[n][3])) { jadge = 1; }
		else if ((Y[n][2] > Y[n][0]) && (Y[n][2] > Y[n][1]) && (Y[n][2] > Y[n][3])) { jadge = 2; }
		else if ((Y[n][3] > Y[n][0]) && (Y[n][3] > Y[n][1]) && (Y[n][3] > Y[n][2])) { jadge = 3; }

		//�e�X�g���x���ƈ�v�����
		if (test_label[n][jadge] == 1) { count++; }
	}
	cout << "���ʗ� = " << (double)count / (double)size << endl;
}
