#include "NN.h"
#include "file.h"

int main(void)
{
	cout << "--------------------" << endl;
	cout << "\tNN\t" << endl;
	cout << "--------------------" << endl;
	int D = 4;					//���͎���
	int C = 4;					//�N���X��
	int M = 4;					//���ԑw�̗v�f��
	int Mk = 4;				//���ԑw�̑w��
	double epsilon = 0.1;	//�w�K��
	int batch_size = 1;		//batch�T�C�Y�̎w��(1��online, N��batch, n��mini batch)

	NN nn(D, C, M, Mk, epsilon, batch_size);
	//for (auto& V : nn._weight.First) for (auto& v : V) {
	//	cout << v << " ";
	//}cout << endl;
	//for (auto& VV : nn._weight.Mid) for (auto& V : VV) for (auto& v : V) {
	//	cout << v << " ";
	//}cout << endl;
	//for (auto& V : nn._weight.Last) for (auto& v : V) {
	//	cout << v << " ";
	//}cout << endl;
	vector<double> vec = { 1,1,1,1 };
	vector<double> label = { 1,0,0,0 };
	vector<double> delta(C);
	nn.forward(vec);
	for (int n = 0; n < 200; n++) {
		cout << endl;
		for (int i = 0; i < nn._C; i++) {
			cout << nn._output.Last[i] << endl;
			delta[i] = nn._output.Last[i] - label[i];
		}
		nn.backward(delta); cout << endl;
		nn.forward(vec);
		for (auto x : nn._output.Last) cout << x << endl;
	}
	return 0;
}