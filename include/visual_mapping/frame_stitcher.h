#ifndef _FRAME_STITCHER_H_
#define _FRAME_STITCHER_H_


#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

#define MIN_INLIERS			(40)
#define MIN_HESSIAN			(100)
#define GOOD_KP_NUM			(300)
#define MAX_RANSAC_ITER		(4000)

/**
* @brief Visual mapping frame stitcher
*/
class FrameStitcher
{
    bool						initialized_;
    // coordinates transformation of the current frame relative to the 
    // first frame 
    cv::Mat						acc_H_;
    // accumulated origin transformatin (due to map size modification)
    cv::Mat                     acc_center_H_;
    cv::Mat                     prv_gray_;
    std::vector<cv::Point2f>    prv_pts_;
    cv::Mat						weights_;
	

    void WarpPoints(const cv::Mat& H, 
		const std::vector<cv::Point2f> &pts_in, 
        std::vector<cv::Point2f> *pts_out);

    void WarpKPoints(const cv::Mat& H, 
		const std::vector<cv::KeyPoint> &kpts_in, 
        std::vector<cv::KeyPoint> *kpts_out);

    int FindTransformRANSAC(const std::vector<cv::Point2f> &src_pts, 
		const std::vector<cv::Point2f> &dst_pts,
		int iteration_num,
        double ransac_reproj_th,		
		cv::Mat *M);

	int FindTransformIntrinsic(const std::vector<cv::Point2f> &src_pts, 		
		const cv::Mat &H,
		int iteration_num,
        double ransac_reproj_th,		
		cv::Mat *M);

    void MatchFrames(const cv::Mat &gray,
        std::vector<cv::Point2f> *prv_matching_pts, 
        std::vector<cv::Point2f> *cur_matching_pts);

    void FindVisualMapDim(const cv::Mat &H, 
		const cv::Size &frame_size, 
        const cv::Size &panorama_size, 
		double *minCols, 
		double *maxCols, 
        double *minRows, 
		double *maxRows);

	
public:
	FrameStitcher();

	/**
   * @brief Stitch new frame into a the visual map
   *
   * @param frame new frame
   * @param feature_mask features within the masked area will be ignored
   * @param visual_map current visual map and the output location for the new visual map
   * @param framed_visual_map new visual map with current frame framed
   */
	void Stitch(const cv::Mat &frame, 
		const cv::Mat &feature_mask, 
        cv::Mat *visual_map, 
		cv::Mat *framed_visual_map);
};

#endif