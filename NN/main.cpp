#include "NN.h"
#include "file.h"

int main(void)
{
	cout << "aaaa" << endl;
	int D = 4;					//���͎���
	int K = 4;					//�N���X��
	int M = 4;					//���ԑw�̗v�f��
	int Mk = 4;				//���ԑw�̑w��
	double epsilon = 0.1;	//�w�K��
	int batch_size = 1;		//batch�T�C�Y�̎w��(1��online, N��batch, n��mini batch)

	NN nn(D, K, M, Mk, epsilon, batch_size);

	return 0;
}