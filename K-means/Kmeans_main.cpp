#include <stdio.h>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>

#define GNUPLOT_PATH "D:/gnuplot/bin/gnuplot.exe -persist" //gnuplot.exe があるパス

using namespace std;

void fileset(string filename, vector<vector<double>>& datas)
{
    ifstream ifs_file(filename);
    if (ifs_file.fail()) {	// ファイルオープンに失敗したらそこで終了
        cerr << "cannot open the file - '" << filename << "'" << endl;
        exit(1);
    }
    string line;
    while (getline(ifs_file, line)) { // 1行読んで
        replace(line.begin(), line.end(), ',', ' '); //カンマ区切りを空白区切りに
        istringstream iss(line);
        vector<double> inner;
        double data;
        while (iss >> data) { // 1個ずつ切り分ける
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
    // gnuplot の起動コマンド
    if ((gp = _popen(GNUPLOT_PATH, "w")) == NULL) { // gnuplot をパイプで起動
        fprintf(stderr, "ファイルが見つかりません %s.", GNUPLOT_PATH);
        exit(EXIT_FAILURE);
    }    
    /***************************************
    gnuplot へコマンドを送る．動かしたいコマンドを，
    fprintf(gp, "コマンド\n");
    のように記述する．
    ***************************************/
     //fprintf(gp, "load \"pltsetting.plt\"\n");
    fprintf(gp, "plot '-'\n");
    for (int i = 0; i < datas.size(); i++) {
        fprintf(gp, "%f\t%f\n", datas[i][0], datas[i][1]);
    }
    fprintf(gp, "e\n");
     fflush(gp); // バッファに格納されているデータを吐き出す（必須）
     system("pause");
     fprintf(gp, "exit\n"); // gnuplot の終了
     _pclose(gp);


    return 0;
}