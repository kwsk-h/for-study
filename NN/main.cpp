#include "NN.h"
#include "../INCLUDE/file.h"

int main(void)
{
	cout << "--------------------" << endl;
	cout << "\tNN\t" << endl;
	cout << "--------------------" << endl;

	vector<vector<double>> input_datas;
	vector<vector<double>> input_labels;
	//dataset name
	vector<vector<string>> datasets
		= { { "data\\input1.txt","data\\label1.txt" },
		{ "data\\input2.txt","data\\label2.txt" },
		{ "data\\input3.txt","data\\label3.txt" },
		{ "data\\input4.txt","data\\label4.txt" }
	};	
	//select dataset
	int dataset_num = 1;
	cout<<"-----read files-----" << endl;
	/*fileset<double>(datasets[dataset_num - 1][0], input_datas);
	fileset<double>(datasets[dataset_num - 1][1], input_labels);	*/
	input_datas = { {0,0,0,},{1,0,1},{1,1,1},{1,1,0},{1,0,0},{0,0,1} };
	input_labels = { {0},{0},{1},{0},{1},{1} };
	cout << "------complete------" << endl;

	int D = input_datas[0].size();					//“ü—ÍŸŒ³
	int C = input_labels[0].size();					//ƒNƒ‰ƒX”
	int M = 3;												//’†ŠÔ‘w‚Ì—v‘f”
	int Mk =  3;											//’†ŠÔ‘w‚Ì‘w”
	double epsilon = 0.05;								//ŠwK—¦
	int batch_size = 3;			//batchƒTƒCƒY‚Ìw’è(1©online, N©batch, n©mini batch)

	NN nn(D, C, M, Mk, epsilon, batch_size);
	cout << endl;
	nn.Learning(input_datas, input_labels);
	cout << endl;
	nn.check(input_datas, input_labels);

	return 0;
}