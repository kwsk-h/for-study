#include "NN.h"
#include "file.h"

int main(void)
{
	cout << "aaaa" << endl;
	int D = 4;					//“ü—ÍŽŸŒ³
	int K = 4;					//ƒNƒ‰ƒX”
	int M = 4;					//’†ŠÔ‘w‚Ì—v‘f”
	int Mk = 4;				//’†ŠÔ‘w‚Ì‘w”
	double epsilon = 0.1;	//ŠwK—¦
	int batch_size = 1;		//batchƒTƒCƒY‚ÌŽw’è(1©online, N©batch, n©mini batch)

	NN nn(D, K, M, Mk, epsilon, batch_size);

	return 0;
}