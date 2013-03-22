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
    
  } else if (strcmp(argv[1], "-p") == 0) {
    
  } else if (strcmp(argv[1], "-h") == 0) {
    compute_hogimg(argv[2], argv[3]);
  }
  
  return 0;
}
