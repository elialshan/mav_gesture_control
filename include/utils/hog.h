#include <cstdlib>
#include <iostream>
#include <string>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "dense_track.h"
#include "utils.h"

void	NormDesc(float *desc, int nbins);	
void	BuildHogMat(const cv::Mat& x_comp, const cv::Mat& y_comp, float* desc);
void	ExtractHog(const cv::Mat &org_img, int nx_blocks, int ny_blocks, int nbins, cv::Mat *frame_desc);