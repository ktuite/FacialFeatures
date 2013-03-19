#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "face_feature_detect.hpp"
#include "svm.h"

#define PI 3.1415926536

using namespace std;
using namespace cv;

FaceFeature::FaceFeature(string configFile) {
  model = NULL;
  loadConfigFile(configFile);
}

void FaceFeature::extract_line(const string& line, vector<float>& values) {
  vector<string> tokens;
  istringstream iss(line);
  tokens.clear();
  copy(istream_iterator<string>(iss),
       istream_iterator<string>(),
       back_inserter<vector<string> >(tokens));
  values.clear();
  values.reserve(tokens.size());
  for (unsigned int i = 0; i < tokens.size(); i++)
    values.push_back(atof(tokens[i].c_str()));
}

void FaceFeature::loadConfigFile(string path) {
  canonicalPoints.clear();
  ifstream fi(path.c_str(), ifstream::in);
  string line, x, y;
  int num;
  getline(fi, line);
  getline(fi, line);
  num = atoi(line.c_str());
  getline(fi, line);
  for (int i = 0; i < num; i++) {
    getline(fi, line);
    stringstream linestream(line);
    getline(linestream, x, ' ');
    getline(linestream, y);
    canonicalPoints.push_back(Point2f(atof(x.c_str()), atof(y.c_str())));
  }
  getline(fi, line);
  getline(fi, line);
  stringstream line1(line);
  getline(line1, x, ' ');
  getline(line1, y);
  roi.x = atoi(x.c_str());
  roi.y = atoi(y.c_str());
  getline(fi, line);
  getline(fi, line);
  stringstream line2(line);
  getline(line2, x, ' ');
  getline(line2, y);
  roi.width = atoi(x.c_str());
  roi.height = atoi(y.c_str());

  getline(fi, line);
  getline(fi, line);
  ms_file = line;
  ifstream ms(line.c_str(), ifstream::in);
  mean.clear();
  stddev.clear();
  extract_line(line, mean);

  getline(ms, line);
  extract_line(line, stddev);
}

void FaceFeature::extract_glasses(const Mat & image, Mat & img_glasses) {
  if (image.data == NULL) {
    cerr << "The image is invalid to be warped" << endl;
    return;
  }

//  string img_save_path = "result" + url.substr(9, url.size() - 9);
//  imwrite(img_save_path.c_str(), warped);
  img_glasses = image(roi);
//  cout << "extracted glass area of " << url << endl;
}

void FaceFeature::align(Mat & image, const vector<Point2f> & detectedPoints) {
  if (image.data == NULL) {
    cerr << "The image is invalid to be aligned" << endl;
    return;
  }

  Point2f l((detectedPoints[0].x + detectedPoints[1].x) / 2.0,
            (detectedPoints[0].y + detectedPoints[1].y) / 2.0);
  Point2f r((detectedPoints[6].x + detectedPoints[7].x) / 2.0,
            (detectedPoints[6].y + detectedPoints[7].y) / 2.0);

  double theta = atan2(r.y - l.y, r.x - l.x);
  theta = theta / PI * 180;
  Mat rot_mat = getRotationMatrix2D(l, theta, 1.0);
  Mat dst;
  warpAffine(image, dst, rot_mat, image.size());
  double dist = sqrt((l.x - r.x) * (l.x - r.x) + (l.y - r.y) * (l.y - r.y));
  double delta = 40 / dist;
  resize(dst, dst, Size(dst.cols * delta, dst.rows * delta));
  image = dst(Rect(l.x * delta - 20, l.y * delta - 20, 80, 40));
}

void FaceFeature::compute_hog(vector<string> & urls, vector<vector<float> >& features) {
  HOGDescriptor hog(Size(80, 40), Size(16, 16), Size(8, 8), Size(2, 2),
		    9, -1, 0.2, true, 64);

  vector<string> new_urls;
  new_urls.clear();
  for (unsigned int i = 0; i < urls.size(); i++) {
    vector<float> featureVector;
    Mat img;
    img = imread(urls[i].c_str(), 1);
    if (img.channels() == 3)
      cvtColor(img, img, CV_BGR2GRAY);
    if (img.data == NULL)
      continue;

    compute_hog(hog, img, featureVector, urls[i]);
    if (!featureVector.empty()) {
      new_urls.push_back(urls[i]);
      features.push_back(featureVector);
    }
  }
  urls = new_urls;
}

void FaceFeature::compute_hog(const HOGDescriptor& hog, const Mat& img, vector<float>& feature, string url) {
  assert(img.cols == hog.winSize.width);
  assert(img.rows == hog.winSize.height);
  vector<Point> locations;
  hog.compute(img, feature, Size(4, 4), Size(0, 0), locations);
}

void FaceFeature::getUrls(const string& path, vector<string>& urls) {
  ifstream fi(path.c_str(), ifstream::in);
  string line;
  while (getline(fi, line))
    urls.push_back(line);
  fi.close();
}

void FaceFeature::mean_stddev(vector<vector<float> >& features) {
  unsigned int i, j;
  unsigned int num_features = features[0].size();
  cout << "size of features:  "  << num_features << endl;

  vector<float> tot;
  tot.reserve(num_features);
  mean.reserve(num_features);
  stddev.reserve(num_features);
  for (i = 0; i < num_features; i++) {
    tot.push_back(0);
    mean.push_back(0);
    stddev.push_back(0);
  }

  for (i = 0; i < num_features; i++)
    for (j = 0; j < features.size(); j++)
      tot[i] += features[j][i];

  for (i = 0; i < num_features; i++)
    mean[i] = tot[i] / features.size();

  cout << "mean computed" << endl;

  for (i = 0; i < num_features; i++) {
    float err = 0;
    for (j = 0; j < features.size(); j++)
      err += (mean[i] - features[j][i]) * (mean[i] - features[j][i]);
    err /= features.size();
    stddev[i] = sqrt(err);
  }

  cout << "stddev computed" << endl;

  for (i = 0; i < num_features; i++)
    for (j = 0; j < features.size(); j++)
      if (stddev[i] != 0)
	features[j][i] = (features[j][i] - mean[i]) / (2 * stddev[i]);
}

void FaceFeature::train(vector<vector<float> >& features, vector<float>& labels, string& save_path) {
  mean_stddev(features);
  svm_parameter param;
  param.svm_type = C_SVC;
  param.kernel_type = LINEAR;
  param.degree = 3;
  param.gamma = 0.5;
  param.coef0 = 0;
  param.nu = 0.5;
  param.cache_size = 100;
  param.C = 1;
  param.eps = 1e-3;
  param.p = 0.1;
  param.shrinking = 1;
  param.probability = 0;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;

  svm_problem prob;
  prob.l = (int)features.size();
  prob.y = new double[prob.l];
  unsigned int line_size = features[0].size();
  svm_node *x_space = new svm_node[(line_size + 1) * prob.l];
  prob.x = new svm_node *[prob.l];
  for (unsigned int i = 0; i < features.size(); i++) {
    for (unsigned int j = 0; j < line_size; j++) {
      x_space[(line_size + 1) * i + j].index = j + 1;
      x_space[(line_size + 1) * i + j].value = features[i][j];
    }
    x_space[(line_size + 1) * i + line_size].index = -1;
    prob.x[i] = &x_space[(line_size + 1) * i];
    prob.y[i] = labels[i];
  }

  cout << "svm_parameter and svm_problem are computed" << endl;

  model = svm_train(&prob, &param);
  cout << "nr_class " << model->nr_class << endl;
  cout << "total #SV " << model->l << endl;

  svm_save_model(save_path.c_str(), model);

  cout << "saved model" << endl;

  cout << ms_file << endl;
  
  fstream file;
  file.open(ms_file.c_str(), ios::out);
  file << mean[0];
  for (unsigned int i = 1; i < mean.size(); i++)
    file << mean[i] << " ";
  file << endl;

  file << stddev[0];
  for (unsigned int i = 1; i < stddev.size(); i++)
    file << stddev[i] << " ";
  file << endl;
  file.close();

  cout << "Finished Train" << endl;
}

void FaceFeature::train_data(string positive_urls, string negative_urls, string save_path) {
  vector<vector<float> > features;
  vector<float> labels;
  vector<string> images;

  getUrls(positive_urls, images);
  compute_hog(images, features);
  int p = (int)features.size();
  for (int i = 0; i < p; i++)
    labels.push_back(1);

  images.clear();
  getUrls(negative_urls, images);
  compute_hog(images, features);
  int n = (int)features.size() - p;
  for (int i = 0; i < n; i++)
    labels.push_back(-1);

  cout << "train : computed hog  " << positive_urls << "   " << negative_urls << endl;
  
  train(features, labels, save_path);
}

double FaceFeature::predict_img(string model_url, const Mat& img) {
  vector<float> feature;
  HOGDescriptor hog(Size(80, 40), Size(16, 16), Size(8, 8), Size(8, 8),
		    9, -1, 0.2, true, 64);
  compute_hog(hog, img, feature, "test");
  model = svm_load_model(model_url.c_str());
  for (unsigned int i = 0; i < feature.size(); i++) {
    if (stddev[i] != 0)
      feature[i] = (feature[i] - mean[i]) / (2 * stddev[i]);
  }
  svm_node* tmp = new svm_node[feature.size() + 1];
  for (unsigned int i = 0; i < feature.size(); i++) {
    tmp[i].index = (int)i + 1;
    tmp[i].value = (double)(feature[i]);
      
  }
  tmp[feature.size()].index = -1;
  return svm_predict(model, tmp);
}

void FaceFeature::compute_hogimg(string positive_urls, string negative_urls) {
  vector<vector<float> > features;
  vector<string> images;
  getUrls(positive_urls, images);
  compute_hog(images, features);
  for (unsigned int i = 0; i < images.size(); i++) {
    ostringstream buffer(ostringstream::out);
    buffer <<  "./data/hogimg/positive/" << i << ".jpg" ;
    Mat img = imread(images[i].c_str(), 1);
    Mat visu = get_hogdescriptor_visu(img, features[i]);
    imwrite(buffer.str().c_str(), visu);
  }

  cout << "finish positive images" << endl;
  
  images.clear();
  features.clear();
  getUrls(negative_urls, images);
  compute_hog(images, features);
  for (unsigned int i = 0; i < images.size(); i++) {
    ostringstream buffer(ostringstream::out);
    buffer << "./data/hogimg/negative/" << i << ".jpg" ;
    Mat img = imread(images[i].c_str(), 1);
    Mat visu = get_hogdescriptor_visu(img, features[i]);
    imwrite(buffer.str().c_str(), visu);
  }

  cout << "finish negative images" << endl;
}

// this function from here: http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
Mat FaceFeature::get_hogdescriptor_visu(Mat& origImg, vector<float> descriptorValues)
{   
  Mat color_origImg = origImg;
  //cvtColor(origImg, color_origImg, CV_GRAY2RGBA);
 
  float zoomFac = 3;
  Mat visu;
  resize(color_origImg, visu, Size(color_origImg.cols*zoomFac, color_origImg.rows*zoomFac));
  int cellSize        = 4;
  int gradientBinSize = 9;
  float radRangeForOneBin = M_PI/(float)gradientBinSize; // dividing 180° into 9 bins, how large (in rad) is one bin?
 
  // prepare data structure: 9 orientation / gradient strenghts for each cell
  int cells_in_x_dir = origImg.size.p[1] / cellSize;
  int cells_in_y_dir = origImg.size.p[0] / cellSize;
  float*** gradientStrengths = new float**[cells_in_y_dir];
  int** cellUpdateCounter   = new int*[cells_in_y_dir];
  for (int y=0; y<cells_in_y_dir; y++)
  {
    gradientStrengths[y] = new float*[cells_in_x_dir];
    cellUpdateCounter[y] = new int[cells_in_x_dir];
    for (int x=0; x<cells_in_x_dir; x++)
    {
      gradientStrengths[y][x] = new float[gradientBinSize];
      cellUpdateCounter[y][x] = 0;
 
      for (int bin=0; bin<gradientBinSize; bin++)
	gradientStrengths[y][x][bin] = 0.0;
    }
  }
 
  // nr of blocks = nr of cells - 1
  // since there is a new block on each cell (overlapping blocks!) but the last one
  int blocks_in_x_dir = cells_in_x_dir - 1;
  int blocks_in_y_dir = cells_in_y_dir - 1;
 
  // compute gradient strengths per cell
  int descriptorDataIdx = 0;
 
  for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
  {
    for (int blocky=0; blocky<blocks_in_y_dir; blocky++)            
    {
      // 4 cells per block ...
      for (int cellNr=0; cellNr<4; cellNr++)
      {
	// compute corresponding cell nr
	int cellx = blockx;
	int celly = blocky;
	if (cellNr==1) celly++;
	if (cellNr==2) cellx++;
	if (cellNr==3)
	{
	  cellx++;
	  celly++;
	}
 
	for (int bin=0; bin<gradientBinSize; bin++)
	{
	  float gradientStrength = descriptorValues[ descriptorDataIdx ];
	  descriptorDataIdx++;
 
	  gradientStrengths[celly][cellx][bin] += gradientStrength;
 
	} // for (all bins)
 
 
	// note: overlapping blocks lead to multiple updates of this sum!
	// we therefore keep track how often a cell was updated,
	// to compute average gradient strengths
	cellUpdateCounter[celly][cellx]++;
 
      } // for (all cells)
 
 
    } // for (all block x pos)
  } // for (all block y pos)
 
 
    // compute average gradient strengths
  for (int celly=0; celly<cells_in_y_dir; celly++)
  {
    for (int cellx=0; cellx<cells_in_x_dir; cellx++)
    {
 
      float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
      // compute average gradient strenghts for each gradient bin direction
      for (int bin=0; bin<gradientBinSize; bin++)
      {
	gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
      }
    }
  }
 
 
  // draw cells
  for (int celly=0; celly<cells_in_y_dir; celly++)
  {
    for (int cellx=0; cellx<cells_in_x_dir; cellx++)
    {
      int drawX = cellx * cellSize;
      int drawY = celly * cellSize;
 
      int mx = drawX + cellSize/2;
      int my = drawY + cellSize/2;
 
      rectangle(visu, Point(drawX*zoomFac,drawY*zoomFac), Point((drawX+cellSize)*zoomFac,(drawY+cellSize)*zoomFac), CV_RGB(100,100,100), 1);

//      if (cellx >= 1 && cellx <= 3 && celly >= 1 && celly <= 3)
//	continue;
//      if (cellx >= 6 && cellx <= 8 && celly >= 1 && celly <= 3)
//	continue;
      // draw in each cell all 9 gradient strengths
      for (int bin=0; bin<gradientBinSize; bin++)
      {
	float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
	// no line to draw?
	if (currentGradStrength==0) {
	  continue;
	}
 
	float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
	float dirVecX = cos( currRad + M_PI/2);
	float dirVecY = sin( currRad + M_PI/2);
	float maxVecLen = cellSize/2;
	float scale = 2.5; // just a visualization scale, to see the lines better
 
	// compute line coordinates
	float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
	float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
	float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
	float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

	// draw gradient visualization
	line(visu, Point(x1*zoomFac,y1*zoomFac), Point(x2*zoomFac,y2*zoomFac), CV_RGB(0,255,0), 1);
 
      } // for (all bins)
 
    } // for (cellx)
  } // for (celly)
 
 
    // don't forget to free memory allocated by helper data structures!
  for (int y=0; y<cells_in_y_dir; y++)
  {
    for (int x=0; x<cells_in_x_dir; x++)
    {
      delete[] gradientStrengths[y][x];            
    }
    delete[] gradientStrengths[y];
    delete[] cellUpdateCounter[y];
  }
  delete[] gradientStrengths;
  delete[] cellUpdateCounter;
 
  return visu;
 
} // get_hogdescriptor_visu
