#ifndef _OPERATORS_H_
#define _OPERATORS_H_

#define _USE_MATH_DEFINES
#include <math.h>

#include "cvblob.h"
#include "action_classifiers.h"
#include "orientation.h"

/**
 * @brief Single operator data
 */

class Operator {
	friend class Operators;
	cvb::CvID				id_;
	ActionClassifier		ac_;
	OrientationEstimator	oe_;
    float					prev_label_; 
    float					cur_label_;
    float					action_angle_;
    int                     consistency_cnt_;


public:/**
   * @brief Constructor
   *
   * @param id track id associated to the operator   
   */
	Operator(cvb::CvID id);

	/**
   * @brief get operator id (track id)
   */
	cvb::CvID	get_id() {return id_;}

	/**
   * @brief get operator's current action type
   */
    int			get_label() { return (int)cur_label_;}

	/**
   * @brief get operator's action direction (radians)
   */
    float		get_angle() { return action_angle_;}

  
   	/**
   * @brief Process new frame
   *
   * @param frame new frame
   */
    void		Process(const cv::Mat &frame);
};


/*********************************************************************/

typedef std::pair<cvb::CvTrack *, cv::Mat> OperatorROI;

/**
 * @brief Complete operators data
 */
class Operators {	
    
	typedef std::map<cvb::CvID, Operator *> OperatorsMap;
	typedef std::pair<cvb::CvID, Operator *> CvIDOperator;
	cvb::CvTracks			tracks_;
	OperatorsMap		ops_;
	std::vector<int>	track_id_queue_;

public:
	/**
   * @brief get all operators tracks data
   */
	cvb::CvTracks* get_tracks() { return &tracks_;}

	/**
   * @brief get operator's current action type
   *
   * @param id operator id
   */
	int get_label(cvb::CvID id) { return ops_[id]->get_label(); }

    /**
   * @brief get operator's action direction (radians)
   *
   * @param id operator id
   */    
    float get_angle(cvb::CvID id) { return ops_[id]->get_angle(); }
    
    
    Operator* get_operator(cvb::CvID id) { return ops_[id]; }

	/**
   * @brief Update operators locations, delete lost and add new   
   */    
	void UpdateOperators();

	/**
   * @brief Store operator image for debug/training
   *
   * @param frame operator image
   * @param output_path_char output path
   */    
	void StoreROI(const cv::Mat &frame, const char* output_path_char);

	/**
   * @brief Extract ROI from a aframe
   *
   * @param frame image
   * @param roi vector of ROI's
   * @param max_operators maximum number of operators to detect
   */  
	void ExtractROI(const cv::Mat &frame, std::vector<OperatorROI> *roi, int max_operators);     
        
};
#endif //_OPERATORS_H_