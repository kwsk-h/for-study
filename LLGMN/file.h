#pragma once

#include<stdio.h>
#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
using namespace std;

void fileset(string filename, vector<vector<double>>& datas);//�t�@�C���ǂݍ��ݗp

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
}