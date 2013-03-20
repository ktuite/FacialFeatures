#include <iostream>
#include <fstream>
#include <sstream>
#include <istream>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>
#include "face_feature_detect.hpp"
using namespace std;
using namespace cv;

int main(int argc, char** argv) {

  string face_feature_config = "data/config_canonical.txt";

  FaceFeature ff(face_feature_config);

  if (strcmp(argv[1], "-t") == 0) {
    ff.train_data(argv[2], argv[3], argv[4]);
  } else if (strcmp(argv[1], "-p") == 0) {
    Mat image = imread(argv[3], 1);
    if (argc == 5) {
      vector<Point2f> detectedPoints;
      detectedPoints.clear();
      ifstream fi(argv[4], ifstream::in);
      string line, x, y;
      for (int i = 0; i < 10; i++) {
	getline(fi, line);
	stringstream linestream(line);
	getline(linestream, x, ' ');
	getline(linestream, y);
	detectedPoints.push_back(Point2f(atof(x.c_str()), atof(y.c_str())));
      }
      ff.align(image, detectedPoints);
      imwrite("test_aligned.jpg", image);
    }
    double result = ff.predict_img(argv[2], image);
    cout << (int)result << endl;
  } else if (strcmp(argv[1], "-h") == 0) {
    ff.compute_hogimg(argv[2], argv[3]);
  }
  
  return 0;
}
