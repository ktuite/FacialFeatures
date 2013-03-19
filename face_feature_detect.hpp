#ifndef FACE_FEATURE_DETECT_HPP_
#define FACE_FEATURE_DETECT_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <istream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <boost/progress.hpp>
#include "svm.h"

using namespace std;
using namespace cv;

class FaceFeature {
public:
  FaceFeature(string configFile);

  void extract_glasses(const Mat & image, Mat & img_glasses);
  void align(Mat & image, const vector<Point2f> & detectedPoints);
  void loadConfigFile(string path);
  void train_data(string positive_urls, string negative_urls, string save_path);
  double predict_img(string model_url, const Mat& img);
  void compute_hogimg(string positive_urls, string negative_urls);

private:
  vector<Point2f> canonicalPoints;
  Rect roi;
  svm_model* model;
  string ms_file;
  vector<float> mean, stddev;
  void extract_line(const string& line, vector<float>& values);
  void getUrls(const string& path, vector<string>& urls);
  void compute_hog(vector<string>& urls, vector<vector<float> >& features);
  void compute_hog(const HOGDescriptor& hog, const Mat& img, vector<float>& feature, string url);
  void train(vector<vector<float> >& features, vector<float>& labels, string& save_path);
  void mean_stddev(vector<vector<float> >& features);
  Mat get_hogdescriptor_visu(Mat& origImg, vector<float> feature);
};

#endif // FACE_FEATURE_DETECT_HPP_
