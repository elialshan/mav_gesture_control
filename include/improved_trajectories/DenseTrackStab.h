#ifndef DENSETRACKSTAB_H_
#define DENSETRACKSTAB_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <ctype.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/legacy/legacy.hpp>


extern int g_start_frame;
extern int g_end_frame;
extern int g_scale_num;
extern const float g_scale_stride;

// parameters for descriptors
extern int g_patch_size;
extern int g_nhog_bins;
extern int g_nxy_cell;
extern int g_nt_cell;
extern float g_epsilon;
extern const float g_min_flow;

// parameters for tracking
extern double g_quality;
extern int g_min_distance;
extern int g_init_gap;
extern int g_track_length;

// parameters for rejecting trajectory
extern const float g_min_var;
extern const float g_max_var;
extern const float g_max_dis;

/**
* @brief Bounding box data
*/
typedef struct {
	int x;       // top left corner
	int y;
	int width;
	int height;
}RectInfo;

/**
* @brief IDT trajecotry properties
*/
typedef struct {
    int length;  // length of the trajectory
    int gap;     // initialization gap for feature re-sampling 
}TrackInfo;

/**
* @brief IDT descriptor properties
*/
typedef struct {
    int nBins;   // number of bins for vector quantization
    bool isHof; 
    int nxCells; // number of cells in x direction
    int nyCells; 
    int ntCells;
    int dim;     // dimension of the descriptor
    int height;  // size of the block for computing the descriptor
    int width;
}DescInfo; 

/**
* @brief IDT integral histogram for the descriptors
*/
typedef struct {
    int height;
    int width;
    int nBins;
    int size;
    float* desc;
}DescMat;

/**
* @brief IDT integral histogram for the descriptors
*/
class Track
{
public:
    std::vector<cv::Point2f> point;
    std::vector<cv::Point2f> disp;
    std::vector<float> hog;
    std::vector<float> hof;
    std::vector<float> mbhX;
    std::vector<float> mbhY;
    int index;

	/**
   * @brief Constructor
   *
   * @param first_point trajectory start point
   * @param trackInfo IDT trajectory properties
   * @param hogInfo IDT HOG properties
   * @param hofInfo IDT HOF properties
   * @param mbhInfo IDT MBH properties
   */
    Track(const cv::Point2f& first_point, const TrackInfo& trackInfo, const DescInfo& hogInfo,
          const DescInfo& hofInfo, const DescInfo& mbhInfo)
        : point(trackInfo.length+1), disp(trackInfo.length), hog(hogInfo.dim*trackInfo.length),
          hof(hofInfo.dim*trackInfo.length), mbhX(mbhInfo.dim*trackInfo.length), mbhY(mbhInfo.dim*trackInfo.length)
    {
        index = 0;
        point[0] = first_point;
    }

	/**
   * @brief Add point to trajectory
   *
   * @param new_point trajectory new point
   */
    void addPoint(const cv::Point2f& new_point)
    {
        index++;
        point[index] = new_point;
    }
};

//Initialize.h
void InitTrackInfo(TrackInfo* trackInfo, int track_length, int init_gap);
void InitDescMat(int height, int width, int nBins, DescMat* descMat);
void ReleDescMat(DescMat* descMat);
void InitDescInfo(DescInfo* descInfo, int nBins, bool isHof, int size, int g_nxy_cell, int g_nt_cell);

//Descriptos.h
// get the rectangle for computing the descriptor
void GetRect(const cv::Point2f& point, RectInfo& rect, const int width, const int height, const DescInfo& descInfo);
// compute integral histograms for the whole image
void BuildDescMat(const cv::Mat& xComp, const cv::Mat& yComp, float* desc, const DescInfo& descInfo);
// get a descriptor from the integral histogram
void GetDesc(const DescMat* descMat, RectInfo& rect, DescInfo descInfo, std::vector<float>& desc, const int index);
// for HOG descriptor
void HogComp(const cv::Mat& img, float* desc, DescInfo& descInfo);
// for HOF descriptor
void HofComp(const cv::Mat& flow, float* desc, DescInfo& descInfo);
// for MBH descriptor
void MbhComp(const cv::Mat& flow, float* descX, float* descY, DescInfo& descInfo);



// check whether a trajectory is valid or not
bool IsValid(std::vector<cv::Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length);
bool IsCameraMotion(std::vector<cv::Point2f>& disp);
// detect new feature points in an image without overlapping to previous points
void DenseSample(const cv::Mat& grey, std::vector<cv::Point2f>& points, const double g_quality, const int g_min_distance);
void InitPry(const cv::Mat& frame, std::vector<float>& scales, std::vector<cv::Size>& sizes);
void BuildPry(const std::vector<cv::Size>& sizes, const int type, std::vector<cv::Mat>& grey_pyr);
void DrawTrack(const std::vector<cv::Point2f>& point, const int index, const float scale, cv::Mat& image);
void CollapsetDesc(std::vector<float>& inDesc, std::vector<float>& outDesc, DescInfo& descInfo, TrackInfo& trackInfo);
void ComputeMatch(const std::vector<cv::KeyPoint>& prev_kpts, const std::vector<cv::KeyPoint>& kpts,
	const cv::Mat& prev_desc, const cv::Mat& desc, std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& pts);
void MyWarpPerspective(cv::Mat& prev_src, cv::Mat& src, cv::Mat& dst, cv::Mat& M0, int flags = cv::INTER_LINEAR,
	int borderType = cv::BORDER_CONSTANT, const cv::Scalar& borderValue = cv::Scalar());
void MergeMatch(const std::vector<cv::Point2f>& prev_pts1, const std::vector<cv::Point2f>& pts1,	
	std::vector<cv::Point2f>& prev_pts_all, std::vector<cv::Point2f>& pts_all);
void MatchFromFlow(const cv::Mat& prev_grey, const cv::Mat& flow, std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& pts, const cv::Mat& mask);

namespace my
{

	// in-place median blur for optical flow
	void MedianBlurFlow(cv::Mat& flow, const int ksize);
	void FarnebackPolyExpPyr(const cv::Mat& img, std::vector<cv::Mat>& poly_exp_pyr,
		std::vector<float>& fscales, int poly_n, double poly_sigma);
	void calcOpticalFlowFarneback(std::vector<cv::Mat>& prev_poly_exp_pyr, std::vector<cv::Mat>& poly_exp_pyr,
		std::vector<cv::Mat>& flow_pyr, int winsize, int iterations);
}

#endif /*DENSETRACKSTAB_H_*/
