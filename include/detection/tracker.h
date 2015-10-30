#ifndef _TRACKER_H_
#define _TRACKER_H_

#include <fstream>
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

#include "cvblob.h"
#include "foreground_detector.h"


#define DET_TH (0.5)


int	TrainOperatorClassifier(const char *data_path_char, 
	const char *classifier_path_char);

int	TestOperatorClassifier(const char *data_path_char, 
	const char *classifier_path_char);

int     EvaluateDetection(const char *input_video_path, 
        const char* models_path, const char *gt_path, 
        const char *output_path);

float	ClassifyObject(const cv::Mat &img, 
	const CvSVM &operatorClassifier);

void	FilterObjects(const cv::Mat &img, 
	const CvSVM &operatorClassifier, cvb::CvBlobs &blobs);

void TrackOperators(const cv::Mat &foreMat, 
	const cv::Mat &frameMat, 
	const CvSVM &operatorClassifier, 
	cvb::CvTracks *tracks);

void DetectBlobs(const cv::Mat &fore, 
	cvb::CvBlobs *blobs);

void TrackTarget(const cv::Mat &foreMat, 
	const cv::Mat &frameMat, 
	cvb::CvTracks *tracks);

void MarkTarget(ForegroundDetector &fd, 
	const cv::Mat &cur_frame, 
	cvb::CvTracks *tracks);

void RenderTracks(const cvb::CvTracks tracks, 
	unsigned int active_time_th, 
	cv::Scalar bbox_color,
	int bbox_size,
	cv::Mat *img);

void GetObjectsMask(const cv::Size mask_size, 
	const cvb::CvTracks tracks,
	cv::Mat *mask);

void EvaluateTracker(const cvb::CvTracks tracks, 
	const cv::Mat &frame,
	unsigned int active_time_th, 
	std::ifstream *op_track_strm,
	cv::Scalar bbox_color,
	int bbox_size,
	cv::Mat *marked_frame,
	int *tp, int *fp, int *fn);

#endif