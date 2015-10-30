#ifndef ACTION_CLASSIFIERS_H
#define ACTION_CLASSIFIERS_H

#include "dense_track.h"
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
* @brief Action descriptors extractor and classifier
*/
class ActionClassifier {
private:
	static cv::Mat						codebooks_[DESCRIPTORS_NUM];
	static cv::SVM						base_classifiers_[DESCRIPTORS_NUM][ACTIONS_NUM];	
	static cv::SVM						combined_classifier_;
	static bool							statics_loaded_;
	
	DenseTrack							dense_track_;			
	cv::Mat								video_desc_[DESCRIPTORS_NUM];	
	std::vector<CyclicQueue<cv::Mat> >	frame_desc_q_;
	float								cur_label_;
	bool								initialized_;
	
	void			Init(const cv::Mat &frame);	
public:	
	static std::string	desc_names_[DESCRIPTORS_NUM];
	static std::string	actions_names_[ACTIONS_NUM];	

   /**
   * @brief Load codebooks and classifier (those members are shard by all instances)
   *
   * @param codebooks_path_char codebooks path
   * @param classifiers_path_char classifiers path
   */
	static void		LoadStaticMembers(const char *codebooks_path_char, const char *classifiers_path_char);

	/**
   * @brief Convert set of raw trajectory descriptors to BoW format
   *
   * @param trajectories
   * @param codebook 
   * @param frame_desc BoW
   */
	static void		BuildFrameBoW(const cv::Mat &trajectories, const cv::Mat &codebook, cv::Mat *frame_desc);

   /**
   * @brief Accumulate frame descriptors along a time window
   *
   * @param frame_desc frame BoW descriptor
   * @param frame_desc_q cyclic queue of previous frame descriptors
   * @param video_desc accumulator of previous frame descriptors
   * @param new_video_desc if a time window was completed will contain the new video descriptor
   */
	static int		BuildVideoBow(const cv::Mat &frame_desc, CyclicQueue<cv::Mat> *frame_desc_q, cv::Mat *video_desc, cv::Mat *new_video_desc);

   /**
   * @brief Stack base descriptors
   *
   * @param base_desc set of video descriptors (trajectories, HOG, HOF and MBH BoW)
   * @param base_classifiers base descriptors classifiers
   * @param combined_desc combined descriptor
   */
	static void		CombineDesc(const cv::Mat base_desc[DESCRIPTORS_NUM], const cv::SVM base_classifiers[DESCRIPTORS_NUM][ACTIONS_NUM], cv::Mat *combined_desc);
	
	ActionClassifier();

	/**
   * @brief Extract action frame and video descriptors from a video frame
   *
   * @param frame input frame
   * @param frame_desc set of frame descriptors (trajectories, HOG, HOF and MBH BoW)
   * @param base_video_desc set of video descriptors (trajectories, HOG, HOF and MBH BoW)
   *		video descriptor is created by accumulating frame descriptors along a time window
   */
	int		Extract(const cv::Mat &frame, cv::Mat frame_desc[], cv::Mat base_video_desc[DESCRIPTORS_NUM]);  

	/**
   * @brief Classify frame based on previously extracted descriptors
   * @param frame input frame
   * @param label pointer to output label
   */
	void	ClassifyFrame(const cv::Mat &frame, float *label);

};


#endif //ACTION_CLASSIFIERS_H