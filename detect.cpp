#include <iostream>
#include <fstream>
#include <sstream>
#include <istream>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>
#include "detect_util.hpp"
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
  if (strcmp(argv[1], "-t") == 0) {
    vector<vector<float> > features;
    vector<float> labels;
    string line;
    ifstream fp(argv[2], ifstream::in);
    while (getline(fp, line)) {
      vector<float> tmp;
      extract_line(line, tmp);
      features.push_back(tmp);
      labels.push_back(1);
    }
    ifstream fn(argv[3], ifstream::in);
    while (getline(fn, line)) {
      vector<float> tmp;
      extract_line(line, tmp);
      features.push_back(tmp);
      labels.push_back(0);
    }
    train(features, labels, argv[4]);
  } else if (strcmp(argv[1], "-p") == 0) {
    cout << predict_hog(argv[2], argv[3]) << endl;
  }
  
  return 0;
}

