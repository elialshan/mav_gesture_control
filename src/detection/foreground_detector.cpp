#include "foreground_detector.h"

ForegroundDetector::ForegroundDetector(int frame_height, int frame_width, int hbins, int sbins, float alpha):hbins_(hbins), sbins_(sbins), alpha_(alpha) {
	initialized_ = false;
	int open_size = 1, close_size = 8, close_size2 = 2, target_size = 4, shape = cv::MORPH_RECT;
	const float mu_scalar[] = { 0.0981f, 0.9651f, 0.9805f};

	background_model_ = cv::Mat::zeros(hbins_, sbins_, CV_32FC1);	

	for (int c = 0; c < 3; c++) {
		mu_.push_back(mu_scalar[c] * cv::Mat::ones(frame_height, frame_width, CV_64FC1));
	}

	close_erosion_elem_ = cv::getStructuringElement( shape,
                                       cv::Size( 2*close_size + 1, 2*close_size+1 ),
                                       cv::Point( close_size, close_size ) );

	close_dilation_elem_ = cv::getStructuringElement( shape,
                                       cv::Size( 2*close_size + 1, 2*close_size+1 ),
                                       cv::Point( close_size, close_size ) );
        
        close_dilation_elem2_ = cv::getStructuringElement( shape,
                                       cv::Size( 2*close_size2 + 1, 2*close_size2+1 ),
                                       cv::Point( close_size2, close_size2 ) );
        
	open_erosion_elem_ = cv::getStructuringElement( shape,
                                       cv::Size( 2*open_size + 1, 2*open_size+1 ),
                                       cv::Point( open_size, open_size ) );

	open_dilation_elem_ = cv::getStructuringElement( shape,
                                       cv::Size( 2*open_size + 1, 2*open_size+1 ),
                                       cv::Point( open_size, open_size ) );

	target_dilation_elem_ = cv::getStructuringElement( cv::MORPH_RECT,
                                    cv::Size( 2 * target_size + 1, 2 * target_size + 1 ),
                                    cv::Point( target_size, target_size ) );
        
        target_erosion_elem_ = cv::getStructuringElement( cv::MORPH_RECT,
                                    cv::Size( 1 * target_size + 1, 1 * target_size + 1 ),
                                    cv::Point( target_size, target_size ) );
}

void ForegroundDetector::GetHistogram(const cv::Mat &hue, const cv::Mat &sat, const cv::Mat &mask, int hbins, int sbins, cv::Mat *hist) {	
	int randSize = (int)(0.01 * hue.rows * hue.cols);
	cv::Mat_<int> randRow(1,randSize);
	cv::Mat_<int> randCol(1,randSize);
	double n;

	//build 2D histogram from random 1% of the image pixels
	randu(randRow, cv::Scalar(0), cv::Scalar(hue.rows));
	randu(randCol, cv::Scalar(0), cv::Scalar(hue.cols));
	for (int idx = 0; idx < randSize; idx++) {
		if (mask.at<uchar>(randRow(idx),randCol(idx)) == 0)
			hist->at<float>(hue.at<uchar>(randRow(idx),randCol(idx)), sat.at<uchar>(randRow(idx),randCol(idx)))++;		
	}

	//smooth and normalize the histogram
	GaussianBlur(*hist, *hist, cv::Size(5,5), 3, 3);
	n = norm(*hist, cv::NORM_L1);
	*hist /= n;	
}


void ForegroundDetector::GetOperatorMask(const cv::Mat &frame, const cv::Mat &mask, cv::Mat *fore) {	
	cv::Mat hist, hue, sat, hsv;		
	std::vector<cv::Mat> hsv_channels;	

	hist = cv::Mat::zeros(hbins_, sbins_, CV_32FC1);
	

	*fore = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1 );

	//transform the image to HSV space   
	cvtColor( frame, hsv, CV_BGR2HSV );
	split(hsv, hsv_channels);

	//extract hue and saturation
	hsv_channels[0].convertTo(hue,CV_32FC1);
	hue = (hbins_ - 1) * hue / 179;
	hue.convertTo(hue,CV_8UC1);
	hsv_channels[1].convertTo(sat,CV_32FC1);
	sat = (sbins_ - 1) * sat / 255;
	sat.convertTo(sat,CV_8UC1);

	//calcualte 2D histogram
	GetHistogram(hue, sat, mask, hbins_, sbins_, &hist);
	if (initialized_ == false) {		
		background_model_ = hist;
		initialized_ = true;
	}
	else
		background_model_ = alpha_ * hist + (1.0f - alpha_) * background_model_;

	{
		cv::Mat show_hist;
		double max;		

		background_model_.copyTo(show_hist);		
		max = norm(show_hist, cv::NORM_INF);
		show_hist = 255*(show_hist/max);
				
		show_hist.convertTo(show_hist, CV_8UC1);
		resize(show_hist, show_hist, cv::Size(200, 200));
		//imshow("background model", show_hist);		
	}
	
	//label rare pixels as foreground pixels
	for (int row = 0; row < fore->rows; row++) {
		for (int col = 0; col <fore->cols; col++) {
			uchar h = hue.at<uchar>(row,col);					
			uchar s = sat.at<uchar>(row,col);					
			if (background_model_.at<float>(h, s) < FOREGROUND_THRESHHOLD) {
				fore->at<char>(row,col) = (char)255;
			}				
		}
	}

	//filter isolated pixels and inhance connected regions
	// image open
	erode(*fore, *fore, open_erosion_elem_);
	//dilate(*fore, *fore, open_dilation_elem_);
	// image close
	dilate(*fore, *fore, close_dilation_elem_);        
	//erode(*fore, *fore, close_dilation_elem2_);
	//imshow("Operator Mask Clean", *fore);
	
}

void ForegroundDetector::GetTargetMask(const cv::Mat &frame, cv::Mat *target_mask) {
	std::vector<cv::Mat> frame_channels;

	split(frame, frame_channels);
	*target_mask = 255*((frame_channels[0] < (0.45*255)) & (frame_channels[1] > (0.97*255)) & (frame_channels[2] > (0.97*255)));
	dilate(*target_mask, *target_mask, target_dilation_elem_);
        erode(*target_mask, *target_mask, target_erosion_elem_);
}