#include "confusionmatrix.h"

//--------------------------------
//Constructor
// 
// out[sample][class] ：事後確率
// label[sample][class] ：正解ラベル
//--------------------------------
ConfusionMatrix::ConfusionMatrix()
{

}
ConfusionMatrix::ConfusionMatrix(const vector<vector<double>> out, const vector<vector<double>> label)
{
	setPram(out, label);
}

//--------------------------------
//settting init Pram（Accessor：setter）
// 
// out[sample][class] ：事後確率
// label[sample][class] ：正解ラベル
//--------------------------------
void ConfusionMatrix::setPram(const vector<vector<double>> out, const vector<vector<double>> label)
{
	OUT = out;
	Label = label;
	C = out[0].size();
	N = out.size();
	confusionMatrix = make_v<double>(C, C);	fill_v(confusionMatrix, 0);
	Accuracy = 0;
	Recall = make_v<double>(C);	fill_v(Recall, 0);
	Precision = make_v<double>(C);	fill_v(Recall, 0);
	Fmeasure = make_v<double>(C);	fill_v(Recall, 0);

	calcConfusionMatrix();
	calcAccuracy();
	calcRecalls();
	calcPrecisions();
	calcFmeasure();
}

//--------------------------------
//calcConfusionMatrix
//getConfusionMatrix（Accessor：getter）
// 
// return confusionMatrix[真クラス][予測クラス] ：混合行列
//--------------------------------
void ConfusionMatrix::calcConfusionMatrix()
{
	for (int i = 0; i < N; i++) {
		//事後確率最大が予測クラス
		vector<double>::iterator piter = max_element(OUT[i].begin(), OUT[i].end()); //最大値のイテレータ取得
		size_t predict = distance(OUT[i].begin(), piter);//最大値	となるクラスを取得

		//真クラスを取得
		vector<double>::iterator citer = max_element(Label[i].begin(), Label[i].end()); //最大値のイテレータ取得
		size_t correct = distance(Label[i].begin(), citer);//最大値	となるクラスを取得

		//混合行列に配置
		confusionMatrix[correct][predict] += 1;
	}
}
vector<vector<double>> ConfusionMatrix::getConfusionMatrix()
{
	if (confusionMatrix.empty())
	{
		confusionMatrix = make_v<double>(1, 1);
		fill_v(confusionMatrix, 0);
		cout << "Not initialize" << endl;
	}
	return confusionMatrix;
}

//--------------------------------
//calcAccuracy（micro Average）
//getAccuracy（Accessor：getter）
// 
// return confusionMatrix[真クラス][予測クラス] ：混合行列
//--------------------------------
void ConfusionMatrix::calcAccuracy()
{
	int count = 0;	//正解数
	for (int i = 0; i < C; i++) {
		count += confusionMatrix[i][i];
	}
	Accuracy = (double)count / (double)N;
}
double ConfusionMatrix::getAccuracy(){	return Accuracy;}

//--------------------------------
//calcRecalls (all class)
//getRecall（Accessor：getter）
// 
// input classNum：指定クラス
// return Precision
//--------------------------------
void ConfusionMatrix::calcRecalls()
{
	for (int classNum = 0; classNum < C; classNum++) {
		int count = 0;	//正解数
		int total = 0;		//総数
		for (int j = 0; j < C; j++) {
			if (j == classNum) count += confusionMatrix[classNum][j];
		}
		total = accumulate(confusionMatrix[classNum].begin(), confusionMatrix[classNum].end(), 0.0);
		Recall[classNum] = (double)count / (double)total;
	}
}
double ConfusionMatrix::getRecall(int classNum) { return Recall[classNum]; }
vector<double> ConfusionMatrix::getRecall() { return Recall; }

//--------------------------------
//calcPrecisions (all class)
//getPrecision（Accessor：getter）
// 
// input classNum：指定クラス
// return Precision
//--------------------------------
void ConfusionMatrix::calcPrecisions()
{
	for (int classNum = 0; classNum < C; classNum++) {
		int count = 0;	//正解数
		int total = 0;		//総数
		for (int i = 0; i < C; i++) {
			if (i == classNum) count += confusionMatrix[i][classNum];
			total += confusionMatrix[i][classNum];
		}
		Precision[classNum] = (double)count / (double)total;
	}
}
double ConfusionMatrix::getPrecision(int classNum) { return Precision[classNum]; }
vector<double> ConfusionMatrix::getPrecision() { return Precision; }

//--------------------------------
//calcFmeasure (all class)
//getFmeasure（Accessor：getter）
// 
// input classNum：指定クラス
// return Precision
//--------------------------------
void ConfusionMatrix::calcFmeasure()
{
	for (int classNum = 0; classNum < C; classNum++) {
		Fmeasure[classNum] = 2 * Recall[classNum] * Precision[classNum] / (Recall[classNum] + Precision[classNum]);
	}
}
double ConfusionMatrix::getFmeasure(int classNum) { return Fmeasure[classNum]; }
vector<double> ConfusionMatrix::getFmeasure() { return Fmeasure; }

//--------------------------------
// result save
// 
// Confusion Matrix
// Accuracy
// class1 : Recall Precision F-measure
// class2 : Recall Precision F-measure
// ...
//--------------------------------
void ConfusionMatrix::resultSave(string savename)
{
	ofstream ofs(savename);
	if (ofs.fail()) {	// ファイルオープンに失敗したらそこで終了
		cerr << "cannot open the file - '" << savename << "'" << endl;
		exit(1);
	}

	string sep;
	int fnd = savename.find(".csv");
	if (fnd == std::string::npos) {
		//not csv
		sep = "\t";
	}
	else {
		sep = ",";
	}

	// confusion matrix
	ofs << "Confusion Matrix" << endl;
	for (auto data : confusionMatrix) {
		for (auto x : data) {
			ofs << x << sep;
		}
		ofs << endl;
	}ofs << endl;

	// Accuracy
	ofs << "Accuracy" << endl;
	ofs << Accuracy << endl << endl;

	//Recall Precision F-measure
	ofs << "Class" << sep << "Recall" << sep << "Precision" << sep << "F-measure" << endl;
	for (int classNum = 0; classNum < C; classNum++) {
		ofs << classNum + 1 << sep << Recall[classNum] << sep << Precision[classNum] << sep << Fmeasure[classNum] << endl;
	}ofs << endl;

	ofs.close();
}