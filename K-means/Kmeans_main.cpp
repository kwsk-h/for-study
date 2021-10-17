#include <stdio.h>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

#define GNUPLOT_PATH "D:/gnuplot/bin/gnuplot.exe -persist" //gnuplot.exe ������p�X
#define  K 3
using namespace std;

struct _subData
{
    string cluster = "black";
    double x;
    double y;
};

void fileset(string filename, vector<_subData>& datas);
void init_cog(vector<_subData> datas, vector<_subData>& cog_axis);
void clustering(vector<_subData>& datas, vector<_subData> cog_axis);
void update_cog(vector<_subData> datas, vector<_subData>& cog_axis);

int main() {
    vector<string> dataFiles;
    vector<string> useData = { "sub"};
    vector<_subData> datas;
    vector<_subData> cog_axis(K);
    vector<_subData> bef_cog(K);
    string path = "./data/";

    for (const auto& file : filesystem::directory_iterator(path)) {        
        dataFiles.push_back(file.path().string());
    }
    for (int i = 0; i < dataFiles.size(); i++) {
        for (const auto& key : useData) {
            if (dataFiles[i].find(key) != string::npos) {
                cout << dataFiles[i] << endl;
                fileset(dataFiles[i], datas);//1�t�H���_�ɑS�f�[�^���i�[
                break;
            }
        }
    }

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
    fprintf(gp, "set terminal gif animate optimize delay 20\n");//gnuplot�̌��ʕۑ��ݒ�// size 480,360
    fprintf(gp, "set output 'result/bef.gif'\n");//�ۑ���
    fprintf(gp, "set palette model RGB rgbformulae 35,13,10\n");
    fprintf(gp, "set nokey\n");// �}���\�����Ȃ�
    fprintf(gp, "set nocolorbox\n");// �J���[�o�[�\�����Ȃ�
    fprintf(gp, "set xrange [-5:10]\n");
    fprintf(gp, "set yrange [-5:10]\n");

    init_cog(datas, cog_axis);//�����l
    for (int k = 0; k < 20; k++) {
        clustering(datas, cog_axis);//�N���X�^�����O        
        fprintf(gp, "set multiplot\n"); // �}���`�v���b�g���[�h
        for (int i = 0; i < datas.size(); i++) {
            fprintf(gp, "plot '-' lc rgb \"%s\"\n", datas[i].cluster.c_str());
            fprintf(gp, "%f\t%f\n", datas[i].x, datas[i].y);
            fprintf(gp, "e\n");
        }
        for (int c = 0; c < K; c++) {
            fprintf(gp, "plot '-' pt 7 lc rgb \"%s\"\n", cog_axis[c].cluster.c_str());//lc palette frac n/10.0
            fprintf(gp, "%f\t%f\n", cog_axis[c].x, cog_axis[c].y);
            fprintf(gp, "e\n");
        }
        fprintf(gp, "set nomultiplot\n"); // �}���`�v���b�g���[�h�I��
        for (int c = 0; c < K; c++) {
            bef_cog[c].x = cog_axis[c].x;
            bef_cog[c].y = cog_axis[c].y;
        }
        update_cog(datas, cog_axis);//�d�S�̍X�V
        //int flg = 0;
        //for (int c = 0; c < K; c++) {
        //    if ((bef_cog[c].x != cog_axis[c].x) && (bef_cog[c].y != cog_axis[c].y)) { break; }
        //    flg++;
        //}
        //if (flg != 0) { break; }
    }
    fprintf(gp, "set terminal windows\n");//�ۑ��o�͐ݒ�
    fprintf(gp, "set output\n");//�ۑ�
    fflush(gp); // �o�b�t�@�Ɋi�[����Ă���f�[�^��f���o���i�K�{�j
    //system("pause");
    fprintf(gp, "exit\n"); // gnuplot �̏I��
    _pclose(gp);
    return 0;
}

//�t�@�C���ǂݍ���
void fileset(string filename, vector<_subData>& datas) {
    ifstream ifs_file(filename);
    if (ifs_file.fail()) {	// �t�@�C���I�[�v���Ɏ��s�����炻���ŏI��
        cerr << "cannot open the file - '" << filename << "'" << endl;
        exit(1);
    }
    string line;
    while (getline(ifs_file, line)) { // 1�s�ǂ��
        replace(line.begin(), line.end(), ',', ' '); //�J���}��؂���󔒋�؂��
        istringstream iss(line);
        struct  _subData subdata;
        iss >> subdata.x >> subdata.y;
        datas.push_back(subdata);
    }
}

//�d�S�̏����l
void init_cog(vector<_subData> datas, vector<_subData>& cog_axis) {
    random_device rnd;     // �񌈒�I�ȗ���������
    mt19937 mt(rnd());            // �����Z���k�E�c�C�X�^��32�r�b�g�ŁA�����͏����V�[�h
    uniform_int_distribution<> randint(0, datas.size());  // [0, 1500] �͈͂̈�l����
    string color = "black";
    for (int i = 0; i < K; i++) {
        if (i == 0) color = "red";
        else if (i == 1) color = "green";
        else if (i == 2) color = "blue";
        cog_axis[i].cluster = color;
        cog_axis[i].x = datas[randint(mt)].x;//(double)i/5.0-3; //
        cog_axis[i].y =datas[randint(mt)].y; //(double)i/5.0-3; // //15.0 * ((double)randint(mt) / (double)datas.size()) - 5.0;
    }
}

//�d�S����f�[�^�̃N���X�^�����O
void clustering(vector<_subData>& datas, vector<_subData> cog_axis) {
    for (auto& data : datas) {
        double min_dis = 1e+5;
        for (const auto cog : cog_axis) {
            double dis = sqrt(pow((cog.x - data.x), 2) + pow((cog.y - data.y), 2));
            if (dis <= min_dis) {
                min_dis = dis;
                data.cluster = cog.cluster;
            }
        }
    }
}

//�e�N���X�^�̃f�[�^����d�S���X�V
void update_cog(vector<_subData> datas, vector<_subData>& cog_axis) {
    vector<_subData>RGB = { { "red",0.0,0.0 }, { "green",0.0,0.0 }, { "blue",0.0,0.0 } };
    vector<int> rgb(K, 0);
    for (const auto data : datas) {
        for (int i = 0; i < K; i++) {
            if (data.cluster == RGB[i].cluster) { RGB[i].x += data.x; RGB[i].y += data.y; rgb[i]++; break; }
        }
    }
    for (int i = 0; i < K; i++) {
        if (rgb[i] != 0) { cog_axis[i].x = RGB[i].x / rgb[i]; cog_axis[i].y = RGB[i].y / rgb[i]; }
    }
}