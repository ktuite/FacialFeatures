FacialFeatures
==============

1. run "make" to compile the codes <\br>
2. there are three modes for FaceFeatureDetect:

   	 # Train new data
   	 ./FaceFeatureDetect -t <positive_urls> <negative_urls> <model_save_path>

	 # Predict a well aligned and cropped image
	 ./FaceFeatureDetect -p <model_path> <image_path>

	 # Predict a image with its fiducial points
	 ./FaceFeatureDetect -p <model_path> <image_path> <points_path>


I got a sample for running the second predict mode for you, feel free to try it:
  ./FaceFeatureDetect -p ./data/model.t ./data/test.jpg ./data/points.txt

NOTE:
1. The train mode will not output anything. The images used to train must be already aligned and cropped <br />
2. The standard fiducial points with size 500 by 500 is stored in ./data/config_canonical.txt under "____all the canonical points on the face", we can change it appropriately
3. The file containing the mean and stddev information of trained data is stored in ./data/config_canonical.txt under "____the file that contains mean and stddev values for current svm model", we can change it appropriately
4. I have include in a pre-trained data stored in ./data/model.t

