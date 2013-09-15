FacialFeatures
==============

1. run "make" to compile the codes <br />
2. there are two modes for FaceFeatureDetect:

   	 # Train new data <br />
   	 ./FaceFeatureDetect -t <positive_urls> <negative_urls> <model_save_directory>

	 # Predict a well aligned and cropped image <br />
	 ./FaceFeatureDetect -p  <hog_path> <model_directory>


I got two sample for running the second predict mode, feel free to try it: <br />
	 ./FaceFeatureDetect -t data/exp1_pos.hog data/exp1_neg.hog data/exp1
	 ./FaceFeatureDetect -p data/pos.hog data/exp1 (outputs 1)
	 ./FaceFeatureDetect -p data/neg.hog data/exp1 (outputs 0)

NOTE: <br />
	1. The train mode will not output anything. <br />
	2. The predict model will output a single number for the predicted label <br />

