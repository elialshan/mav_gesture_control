#ifndef ORIENTATION_H
#define ORIENTATION_H
#include "utils.h"
#include "action_classifiers.h"
#include "hog.h"

/**
 * @brief Action orientation descriptors extractor and classifier
 */

class OrientationEstimator {
private:
	typedef enum{
		DOUBLE = 0,
		SINGLE,
		TYPES_NUM
	}EstimatorType;

	static CvSVM			sin_estimator_[TYPES_NUM];
	static CvSVM			cos_estimator_[TYPES_NUM];
	static bool				statics_loaded_;	
	
	const static int		time_win_len_;
	const static int		time_win_overlap_;
	const static int		nbins_;
	const static int		nx_blocks_;
	const static int		ny_blocks_;
	const static int		ndims_;
	
	CyclicQueue<cv::Mat>	frame_desc_q_;
	cv::Mat					video_desc_;

	float					cur_angle_;
	int						cur_state_;

	void	Reset();

public:
 /**
   * @brief Load single and double handed direction estimators
   *
   * @param single_estimators_dir_char single estimators directory
   * @param double_estimators_dir_char double estimators directory
   */
	static void LoadStaticMembers(const char *single_estimators_dir_char, const char *double_estimators_dir_char);

	OrientationEstimator();

  /**
   * @brief Extract orientation video descriptors from a video frame
   *
   * @param frame input frame
   * @param frame_desc frame descriptor
   */
	int		Extract(const cv::Mat &frame, cv::Mat *frame_desc);	

	/**
   * @brief Classify frame based on previously extracted descriptors
   * @param frame input frame
   * @param label action type
   * @param angle result angle
   */
	void	Evaluate(const cv::Mat &frame, int label, float *angle);

};

#endif //ORIENTATION_H