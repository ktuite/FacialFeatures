#include <iostream>
#include <fstream>
#include <sstream>
#include <istream>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>
#include <sys/time.h>
#include "detect_util.hpp"
using namespace std;
using namespace cv;

typedef unsigned long long timestamp_t;

static timestamp_t
get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

int main(int argc, char** argv) {
  srand(0);
  timestamp_t t0 = get_timestamp();
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
    timestamp_t t0_p = get_timestamp();
    cout << predict_hog(argv[2], argv[3]) << endl;
    timestamp_t t1_p = get_timestamp();
    double p_secs = (t1_p - t0_p) / 1000000.0L;
    //printf("PREDICTION time: %f s\n", p_secs);
  }
  timestamp_t t1 = get_timestamp();
  double secs = (t1 - t0) / 1000000.0L;

  //printf("MAIN time: %f s\n", secs);
  return 0;
}

