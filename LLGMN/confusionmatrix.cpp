#include "confusionmatrix.h"

//--------------------------------
//Constructor
// 
// out[sample][class] �F����m��
// label[sample][class] �F�������x��
//--------------------------------
ConfusionMatrix::ConfusionMatrix()
{

}
ConfusionMatrix::ConfusionMatrix(const vector<vector<double>> out, const vector<vector<double>> label)
{
	setPram(out, label);
}

//--------------------------------
//settting init Pram�iAccessor�Fsetter�j
// 
// out[sample][class] �F����m��
// label[sample][class] �F�������x��
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
//getConfusionMatrix�iAccessor�Fgetter�j
// 
// return confusionMatrix[�^�N���X][�\���N���X] �F�����s��
//--------------------------------
void ConfusionMatrix::calcConfusionMatrix()
{
	for (int i = 0; i < N; i++) {
		//����m���ő傪�\���N���X
		vector<double>::iterator piter = max_element(OUT[i].begin(), OUT[i].end()); //�ő�l�̃C�e���[�^�擾
		size_t predict = distance(OUT[i].begin(), piter);//�ő�l	�ƂȂ�N���X���擾

		//�^�N���X���擾
		vector<double>::iterator citer = max_element(Label[i].begin(), Label[i].end()); //�ő�l�̃C�e���[�^�擾
		size_t correct = distance(Label[i].begin(), citer);//�ő�l	�ƂȂ�N���X���擾

		//�����s��ɔz�u
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
//calcAccuracy�imicro Average�j
//getAccuracy�iAccessor�Fgetter�j
// 
// return confusionMatrix[�^�N���X][�\���N���X] �F�����s��
//--------------------------------
void ConfusionMatrix::calcAccuracy()
{
	int count = 0;	//����
	for (int i = 0; i < C; i++) {
		count += confusionMatrix[i][i];
	}
	Accuracy = (double)count / (double)N;
}
double ConfusionMatrix::getAccuracy(){	return Accuracy;}

//--------------------------------
//calcRecalls (all class)
//getRecall�iAccessor�Fgetter�j
// 
// input classNum�F�w��N���X
// return Precision
//--------------------------------
void ConfusionMatrix::calcRecalls()
{
	for (int classNum = 0; classNum < C; classNum++) {
		int count = 0;	//����
		int total = 0;		//����
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
//getPrecision�iAccessor�Fgetter�j
// 
// input classNum�F�w��N���X
// return Precision
//--------------------------------
void ConfusionMatrix::calcPrecisions()
{
	for (int classNum = 0; classNum < C; classNum++) {
		int count = 0;	//����
		int total = 0;		//����
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
//getFmeasure�iAccessor�Fgetter�j
// 
// input classNum�F�w��N���X
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
	if (ofs.fail()) {	// �t�@�C���I�[�v���Ɏ��s�����炻���ŏI��
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