#pragma once

#include<stdio.h>
#include<iostream>
#include<vector>
#include<fstream>
#include<string>
using namespace std;

void fileset(string filename, vector<vector<double>>& datas, string data_type); //ファイル読み込み用

void fileset(string filename, vector<vector<double>>& datas, string data_type)
{
	ifstream file(filename);
	if (file.fail()) {	// ファイルオープンに失敗したらそこで終了
		cerr << "cannot open the file - '" << filename << "'" << endl;
		exit(1);
	}
	while (!file.eof())
	{
		vector<double> inner;
		double ch1, ch2, ch3, ch4;
		char split;//「,」の空読み用
		if (data_type == "data")
		{
			file >> ch1 >> split >> ch2;
			inner.push_back(ch1);
			inner.push_back(ch2);
		}
		if (data_type == "lavel")
		{
			file >> ch1 >> split >> ch2 >> split >> ch3 >> split >> ch4;
			inner.push_back(ch1);
			inner.push_back(ch2);
			inner.push_back(ch3);
			inner.push_back(ch4);
		}
		datas.push_back(inner);
	}
}