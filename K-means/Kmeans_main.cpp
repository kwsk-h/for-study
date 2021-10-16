#include <stdio.h>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>

#define GNUPLOT_PATH "D:/gnuplot/bin/gnuplot.exe -persist" //gnuplot.exe ������p�X

using namespace std;

void fileset(string filename, vector<vector<double>>& datas)
{
    ifstream ifs_file(filename);
    if (ifs_file.fail()) {	// �t�@�C���I�[�v���Ɏ��s�����炻���ŏI��
        cerr << "cannot open the file - '" << filename << "'" << endl;
        exit(1);
    }
    string line;
    while (getline(ifs_file, line)) { // 1�s�ǂ��
        replace(line.begin(), line.end(), ',', ' '); //�J���}��؂���󔒋�؂��
        istringstream iss(line);
        vector<double> inner;
        double data;
        while (iss >> data) { // 1���؂蕪����
            inner.push_back(data);
        }
        datas.push_back(inner);
    }
}

int main() {
    vector<string> dataFiles;
    vector<string> useData = { "subB_20211001"};
    vector<vector<double>> datas;

    string path = "./data/";
    for (const auto& file : filesystem::directory_iterator(path)) {        
        dataFiles.push_back(file.path().string());
    }

    //std::ofstream ofs("./pltsetting.plt");
    //ofs << "set datafile separator \",\"" << endl;
    //string plotdata;
    for (int i = 0; i < dataFiles.size(); i++) {
        for (const auto& key : useData) {
            if (dataFiles[i].find(key) != string::npos) {
                cout << dataFiles[i] << endl;
                fileset(dataFiles[i], datas);                
                break;
            }
        }
    }
    /*for (int i = 0; i < datas.size(); i++) {
        for (int j = 0; j < datas[i].size(); j++) {

        }
    }*/


    FILE* gp;
    // gnuplot �̋N���R�}���h
    if ((gp = _popen(GNUPLOT_PATH, "w")) == NULL) { // gnuplot ���p�C�v�ŋN��
        fprintf(stderr, "�t�@�C����������܂��� %s.", GNUPLOT_PATH);
        exit(EXIT_FAILURE);
    }    
    /***************************************
    gnuplot �փR�}���h�𑗂�D�����������R�}���h���C
    fprintf(gp, "�R�}���h\n");
    �̂悤�ɋL�q����D
    ***************************************/
     //fprintf(gp, "load \"pltsetting.plt\"\n");
    fprintf(gp, "plot '-'\n");
    for (int i = 0; i < datas.size(); i++) {
        fprintf(gp, "%f\t%f\n", datas[i][0], datas[i][1]);
    }
    fprintf(gp, "e\n");
     fflush(gp); // �o�b�t�@�Ɋi�[����Ă���f�[�^��f���o���i�K�{�j
     system("pause");
     fprintf(gp, "exit\n"); // gnuplot �̏I��
     _pclose(gp);


    return 0;
}