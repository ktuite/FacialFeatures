#include <iostream>
#include <fstream>
#include <sstream>
#include <istream>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>
#include <sys/time.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include "detect_util.hpp"
using namespace std;
using namespace cv;

namespace fs = boost::filesystem;

typedef unsigned long long timestamp_t;

static timestamp_t
get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

void printUsage(){
  printf("USAGE:\n");
  printf("\t ./FaceFeatureDetect -t <positive_urls> <negative_urls> <model_save_directory> \n");
  printf("\t ./FaceFeatureDetect -p <hog_path> <model_directory> \n");
  printf("\t ./FaceFeatureDetect -i <model_directory> [for interactive mode]\n\n");
  exit(0);
}

int main(int argc, char** argv) {
  if (argc < 2){
    printUsage();
    exit(0);
  }

  timestamp_t t0 = get_timestamp();
  if (strcmp(argv[1], "-t") == 0) {
    vector<vector<float> > features;
    vector<float> labels;
    string line;
    string hog_file;
    ifstream fp(argv[2], ifstream::in);
    while (getline(fp, hog_file)) {
      ifstream hp(hog_file.c_str(), ifstream::in);
      if (getline(hp, line)){
        cout << "file found " << endl;
        vector<float> tmp;
        extract_line(line, tmp);
        features.push_back(tmp);
        labels.push_back(1);
      }
      else {
        cout << "file not found: (pos)" << hog_file << endl;
      }
      hp.close();
    }
    ifstream fn(argv[3], ifstream::in);
    while (getline(fn, hog_file)) {
      ifstream hp(hog_file.c_str(), ifstream::in);
      if (getline(hp, line)){
        cout << "file found " << endl;
        vector<float> tmp;
        extract_line(line, tmp);
        features.push_back(tmp);
        labels.push_back(0);
      }
      else {
        cout << "file not found: (neg) " << hog_file << endl;
      }
      hp.close();
    }
    train(features, labels, argv[4]);
  } else if (strcmp(argv[1], "-p") == 0) {
    timestamp_t t0_p = get_timestamp();
    cout << predict_hog(argv[2], argv[3]) << endl;
    timestamp_t t1_p = get_timestamp();
    double p_secs = (t1_p - t0_p) / 1000000.0L;
    //printf("PREDICTION time: %f s\n", p_secs);
  }
  else if (strcmp(argv[1], "-i") == 0) {
    // load the model file once
    char* model_dir = argv[2];
    char model_file[256];
    sprintf(model_file, "%s/model.t", model_dir);
    svm_model * model = svm_load_model(model_file);
    string line;
    while (getline(cin, line)){
      const char *hog_file = line.erase(line.find_last_not_of(" \n\r\t")+1).c_str();
      //printf("THE LINE: --%s--\n", hog_file);
      timestamp_t t0_p = get_timestamp();

      cout << predict_hog_on_loaded_model(hog_file, model_dir, model) << endl;
      cout.flush();

      timestamp_t t1_p = get_timestamp();
      double p_secs = (t1_p - t0_p) / 1000000.0L;
      //printf("PREDICTION time: %f s\n", p_secs);

    }
  }
  else {
    printUsage();
  }
  timestamp_t t1 = get_timestamp();
  double secs = (t1 - t0) / 1000000.0L;

  //printf("MAIN time: %f s\n", secs);
  return 0;
}

