#ifndef FOREFROUND_DETECTOR_H
#define FOREFROUND_DETECTOR_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>

#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/gpu/gpu.hpp"


#define FOREGROUND_THRESHHOLD	(0.00003)

/**
 * @brief Foreground detector
 */

class ForegroundDetector {	
	std::vector<cv::Mat> mu_;
	int hbins_;
	int sbins_;
	cv::Mat open_erosion_elem_;
	cv::Mat open_dilation_elem_;
	cv::Mat close_erosion_elem_;
	cv::Mat close_dilation_elem_;
    cv::Mat close_dilation_elem2_;
	cv::Mat target_dilation_elem_;
    cv::Mat target_erosion_elem_;
	bool initialized_;
	cv::Mat background_model_;
	float alpha_;
	
	void Init(cv::Size size);
	void GetHistogram(const cv::Mat &hue, const cv::Mat &sat, const cv::Mat &mask, int hbins, int sbins, cv::Mat *hist);					  
public:
	/**
   * @brief Constructor
   *
   * @param frame_height frame height
   * @param frame_width frame width
   * @param hbins number of hue histogram bins
   * @param sbins number of saturation histogram bins
   * @param alpha background model update coefficient
   */
	ForegroundDetector(int frame_height, int frame_width, int hbins, int sbins, float alpha);

	/**
   * @brief Extract operator mask from frame by sampling hue and saturation probabilities
   *
   * @param frame input frame
   * @param mask input mask
   * @param fore output mask
   */	
	void GetOperatorMask(const cv::Mat &frame, const cv::Mat &mask, cv::Mat *fore);

	/**
   * @brief Extract target mask from frame by color threshold
   *
   * @param frame input frame
   * @param target_mask output mask
   */
	void GetTargetMask(const cv::Mat &frame, cv::Mat *target_mask);
};

#endif