#include "NN.h"
#include "file.h"

int main(void)
{
	cout << "--------------------" << endl;
	cout << "\tNN\t" << endl;
	cout << "--------------------" << endl;
	int D = 4;					//“ü—ÍŽŸŒ³
	int C = 4;					//ƒNƒ‰ƒX”
	int M = 4;					//’†ŠÔ‘w‚Ì—v‘f”
	int Mk = 4;				//’†ŠÔ‘w‚Ì‘w”
	double epsilon = 0.1;	//ŠwK—¦
	int batch_size = 1;		//batchƒTƒCƒY‚ÌŽw’è(1©online, N©batch, n©mini batch)

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