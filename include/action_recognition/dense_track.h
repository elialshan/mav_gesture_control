#ifndef DENSE_TRACK_H
#define DENSE_TRACK_H

#include "DenseTrackStab.h"
#include "opencv2/gpu/gpu.hpp"

typedef enum
{
    TRJ = 0,//trajectory
    HOF,	//histogram of optical flow
    HOG,	//histogram of gradients
    MBH,	//motion boundary histogram	
    DESCRIPTORS_NUM
}DescType;

typedef enum
{
    ACTION0 = 0,	//DIRECTION
    ACTION1,		//LEFT_ROTATION
    ACTION2,		//RIGHT_ROTATION
    BACKGROUND,
    ACTIONS_NUM
}ActionType;


#define ANGLE_RESOLUTION		(20.0f)
#define OP_RECT_SIZE			(100)
#define TAR_RECT_SIZE			(40)
#define BOW_SIZE				(1000)
#define WINDOW_LENGTH			(40)
#define WINDOW_OVERLAP			(30)

/**
 * @brief Dense trajectories descriptors extractor
 */

class DenseTrack {
	friend class ActionClassifier;
	
	int								id_;
	int								frames_cnt_;
	TrackInfo						track_info_;
	DescInfo						hog_info_, hof_info_, mbh_info_;	
	std::vector<cv::Point2f>		prev_pts_flow_, pts_flow_;
	std::vector<cv::Point2f>		prev_pts_all_, pts_all_;
	cv::Mat							prev_grey_, grey_;
	std::vector<float>				fscales_;
	std::vector<cv::Size>			sizes_;
        std::vector<cv::Mat>			prev_poly_pyr_, poly_pyr_, poly_warp_pyr_;
        std::vector<cv::Mat>			prev_grey_pyr_;
        std::vector<cv::Mat>                    grey_pyr_;
        std::vector<cv::Mat>                    flow_pyr_;
        std::vector<cv::Mat>                    flow_warp_pyr_;
        std::vector<DescMat>                   hog_mat_;
        std::vector<DescMat>                   hof_mat_;
        std::vector<DescMat>                   mbh_mat_x_;
        std::vector<DescMat>                   mbh_mat_y_;
	
	std::vector<std::list<Track> >          xy_scale_tracks_;
	bool					initialized_;	

	void Init(const cv::Mat &frame);
public:
	static	int						unique_id_;
	DenseTrack();
        ~DenseTrack();
	/**
   * @brief Extract dense trajectories descriptors from a video frame
   *
   * @param frame input frame
   * @param descriptor_set set of frame descriptors (trajectories, HOG, HOF and MBH BoW)
   * @param tracks_image (optional) image with trajectories visualization
   */
	int Extract(const cv::Mat &frame, cv::Mat descriptor_set[], cv::Mat *tracks_image = NULL);
};

#endif /*DENSE_TRACK_H*/
