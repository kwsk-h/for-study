#pragma once

#include<stdio.h>
#include<vector>
#include<string>
#include<iostream>
#include<fstream>
#include<sstream>
#include<algorithm>
using namespace std;

void fileset(string filename, vector<vector<double>>& datas);//�t�@�C���ǂݍ��ݗp
void filewrite(string filename, vector<vector<double>>& datas);//�t�@�C�������o���p

void fileset(string filename, vector<vector<double>>& datas)
{
	ifstream ifs_file(filename);
	if (ifs_file.fail()) {	// �t�@�C���I�[�v���Ɏ��s�����炻���ŏI��
		cerr << "cannot open the file - '" << filename << "'" << endl;
		exit(1);
	}
	string line;
	while(getline(ifs_file, line)) { // 1�s�ǂ��
		replace(line.begin(), line.end(), ',', ' '); //�J���}��؂���󔒋�؂��
		istringstream iss(line);
		vector<double> inner;
		double data;
		while (iss >> data) { // 1���؂蕪����
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
	if (ofs.fail()) {	// �t�@�C���I�[�v���Ɏ��s�����炻���ŏI��
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