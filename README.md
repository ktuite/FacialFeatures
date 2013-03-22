FacialFeatures
==============

1. run "make" to compile the codes <br />
2. there are two modes for FaceFeatureDetect:

   	 # Train new data <br />
   	 ./FaceFeatureDetect -t <positive_urls> <negative_urls> <model_save_directory>

	 # Predict a well aligned and cropped image <br />
	 ./FaceFeatureDetect -p  <hog_path> <model_directory>


I got two sample for running the second predict mode, feel free to try it: <br />
  ./FaceFeatureDetect -t ./positive_hog.txt ./negative_hog.txt ./glasses
  ./FaceFeatureDetect -p ./test.txt ./glasses

NOTE: <br />
	1. The train mode will not output anything. <br />
	2. The predict model will output a single number for the predicted label <br />

