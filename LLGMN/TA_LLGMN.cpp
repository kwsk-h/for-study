#include "LLGMN.h"

//�d�ݍX�V�̊֐�
void LLGMN::w_update(vector<vector<vector<double>>>& grad)
{
	double d_weight = 0.0;		 //���݂̏d�ݕω���
	double d_pre_weight = 0.0; //�ꎞ���O�̏d�ݕω���
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				d_weight = (-1.0) * _eta * _gamma * grad[i][j][k];
				_weight[i][j][k] += (_smp_time / 2.0) * (d_weight + d_pre_weight);
				d_pre_weight = d_weight;
			}
		}
	}
}

void LLGMN::gamma_update(double J, vector<vector<vector<double>>>& grad)
{
	double sum_grad = 0.0;
	for (int k = 0; k < _H; k++) {
		for (int i = 0; i < _K; i++) {
			for (int j = 0; j < _M; j++) {
				if (_finite(pow(grad[i][j][k], 2)) != 0) sum_grad += pow(grad[i][j][k], 2);
			}
		}
	}
	_gamma = pow(J, _beta) / sum_grad;
}

void LLGMN::set_eta(double J0)
{
	_eta = pow(J0, (1.0 - _beta)) / (double)(_time * (1.0 - _beta));
}

void LLGMN::talearn_online(vector<vector<double>>& input_data, vector<vector<double>>& input_label) // �Ȃ񂩂�������
{
	int count = 0;
	int size = input_data.size();
	double J = 0;//�]���֐�
	auto O = make_v<double>(size, _H); //�P�w�ڏo�͗p
	auto O_2 = make_v<double>(size, _K, _M); //�Q�w�ڏo�͗p
	auto Y = make_v<double>(size, _K); //�R�w�ڏo�͗p
	auto grad = make_v<double>(_K, _M, _H); //�d�ݍX�V �������p
	set_weight();//�d��weight��������
	cout << "count" << "\tJ" << endl;
	while (count < (_time / _smp_time))
	{
		//������
		fill_v(O, 0);
		fill_v(O_2, 0);
		fill_v(Y, 0);
		
		for (int n = 0; n < size; n++)
		{
			//������
			J = 0;
			fill_v(grad, 0);
			//�w�v�Z
			forward(input_data[n], O_2[n], Y[n], O[n]);

			//�]���֐�
			for (int i = 0; i < _K; i++)
			{
				J += -input_label[n][i] * log(Y[n][i]);
			}
			//�d�ݍX�V
			if (count == 0 && n == 0) set_eta(J);
			backward_online(input_label[n], O_2[n], Y[n], O[n], grad); //���z�v�Z��
			gamma_update(J, grad); 
			w_update(grad);
		}
		count++;
		cout << count << "\t" << J << endl;
		if (J < 1e-5) break;
	}	
	cout << endl;
}

void LLGMN::talearn_batch(vector<vector<double>>& input_data, vector<vector<double>>& input_label)
{	
	int count = 0;
	int n, i;
	int size = input_data.size();
	double J = 0; //�]���֐�
	auto O = make_v<double>(size, _H); //�P�w�ڏo�͗p
	auto O_2 = make_v<double>(size, _K, _M); //�Q�w�ڏo�͗p
	auto Y = make_v<double>(size, _K); //�R�w�ڏo�͗p	
	auto grad = make_v<double>(_K, _M, _H); //�d�ݍX�V �������p
	set_weight();//�d��weight��������
	cout << "count" << "\tJ" << endl;
	while (count < (_time / _smp_time))
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
				J += -input_label[n][i] * log(Y[n][i]) / size;
			}
		}
		//�d�ݍX�V
		if (count == 0) set_eta(J);
		backward_batch(input_label, O_2, Y, O, grad); //���z�v�Z��
		//cout << _eta << " " << _gamma << endl;
		gamma_update(J, grad);
		w_update(grad);

		count++;
		cout << count << "\t" << J << endl;
	}
	cout << endl;
}