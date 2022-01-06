#pragma once

#include<stdio.h>
#include<vector>
#include<string>
#include<iostream>
#include<fstream>
#include<sstream>
#include<algorithm>
using namespace std;

void fileset(string filename, vector<vector<double>>& datas);//ファイル読み込み用
void filewrite(string filename, vector<vector<double>>& datas);//ファイル書き出し用

void fileset(string filename, vector<vector<double>>& datas)
{
	ifstream ifs_file(filename);
	if (ifs_file.fail()) {	// ファイルオープンに失敗したらそこで終了
		cerr << "cannot open the file - '" << filename << "'" << endl;
		exit(1);
	}
	string line;
	while(getline(ifs_file, line)) { // 1行読んで
		replace(line.begin(), line.end(), ',', ' '); //カンマ区切りを空白区切りに
		istringstream iss(line);
		vector<double> inner;
		double data;
		while (iss >> data) { // 1個ずつ切り分ける
			inner.push_back(data);
		}
		datas.push_back(inner);
	}
	ifs_file.close();
}


void filewrite(string filename, vector<vector<double>>& datas) 
{
	string sep;
	int fnd = filename.find(".csv");
	if (fnd == std::string::npos) sep = "\t";//not csv
	else sep = ",";
	ofstream ofs(filename);
	if (ofs.fail()) {	// ファイルオープンに失敗したらそこで終了
		cerr << "cannot open the file - '" << filename << "'" << endl;
		exit(1);
	}
	for (auto data : datas) {
		for (auto x : data) {
			if (x == data[0]) ofs << x;
			else ofs << sep << x;
		}
		ofs << endl;
	}
	ofs.close();
}