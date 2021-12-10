#pragma once

#include<stdio.h>
#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
using namespace std;

void fileset(string filename, vector<vector<double>>& datas);//�t�@�C���ǂݍ��ݗp
void filewrite(string filename, vector<vector<double>>& datas);//�t�@�C�������o���p

template<typename T>
void fileset(string filename, vector<vector<T>>& datas)
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
		vector<T> inner;
		T data;
		while (iss >> data) { // 1���؂蕪����
			inner.push_back(data);
		}
		datas.push_back(inner);
	}
}

template<typename T>
void filewrite(string filename, vector<vector<T>>& datas) 
{
	ofstream ofs(filename);
	if (ofs.fail()) {	// �t�@�C���I�[�v���Ɏ��s�����炻���ŏI��
		cerr << "cannot open the file - '" << filename << "'" << endl;
		exit(1);
	}
	for (auto data : datas) {
		for (auto x : data) {
			if (x == data[0]) ofs << x;
			else ofs << "\t" << x;
		}
		ofs << endl;
	}
}