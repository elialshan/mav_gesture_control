#include "orientation.h"

CvSVM OrientationEstimator::sin_estimator_[2];
CvSVM OrientationEstimator::cos_estimator_[2];
bool OrientationEstimator::statics_loaded_ = false;
const int OrientationEstimator::time_win_len_ = 20;
const int OrientationEstimator::time_win_overlap_ = 10;
const int OrientationEstimator::nbins_ = 16;
const int OrientationEstimator::nx_blocks_ = 5;
const int OrientationEstimator::ny_blocks_ = 5;
const int OrientationEstimator::ndims_ = nbins_*nx_blocks_*ny_blocks_;

void OrientationEstimator::LoadStaticMembers(const char *single_estimators_dir_char, const char *double_estimators_dir_char) {	
	fs::path estimator_path;

	estimator_path = fs::path(double_estimators_dir_char)/fs::path("sin_estimator.model");
	CHECK(fs::exists(estimator_path));
	sin_estimator_[DOUBLE].load(estimator_path.string().c_str());
	estimator_path = fs::path(double_estimators_dir_char)/fs::path("cos_estimator.model");
	CHECK(fs::exists(estimator_path));
	cos_estimator_[DOUBLE].load(estimator_path.string().c_str());
	estimator_path = fs::path(single_estimators_dir_char)/fs::path("sin_estimator.model");
	CHECK(fs::exists(estimator_path));
	sin_estimator_[SINGLE].load(estimator_path.string().c_str());
	estimator_path = fs::path(single_estimators_dir_char)/fs::path("cos_estimator.model");
	CHECK(fs::exists(estimator_path));
	cos_estimator_[SINGLE].load(estimator_path.string().c_str());
	statics_loaded_ = true;
}

OrientationEstimator::OrientationEstimator() {	
	video_desc_ = cv::Mat::zeros(1, ndims_, CV_32FC1);	
	frame_desc_q_ = CyclicQueue<cv::Mat>(time_win_len_);

	cur_angle_ = -1.0f;
	cur_state_ = BACKGROUND;
}

int OrientationEstimator::Extract(const cv::Mat &frame, cv::Mat *res) {	
	cv::Mat frame_desc;

	ExtractHog(frame, OrientationEstimator::nx_blocks_, OrientationEstimator::ny_blocks_, OrientationEstimator::nbins_, &frame_desc);	
	frame_desc_q_.enqueue(frame_desc);					
	video_desc_ += frame_desc;

	if (frame_desc_q_.full()) {
		video_desc_.copyTo((*res));

		for (int i=0; i< (time_win_len_ - time_win_overlap_); i++)	{
			
			frame_desc_q_.dequeue(&frame_desc);
			video_desc_ -= frame_desc;
		}	
		return 0;
	}	

	return -1;
}

void OrientationEstimator::Reset() {
	frame_desc_q_.reset();
	video_desc_.setTo(cv::Scalar(0.0));
}

void OrientationEstimator::Evaluate(const cv::Mat &frame, int label, float *angle) {
	cv::Mat video_desc;
	float sin, cos;
	EstimatorType type;

	if ((cur_state_ != BACKGROUND) && (label == BACKGROUND)){
		cur_angle_ = -1.0;		
		Reset();
	}
	cur_state_ = label;

	if (Extract(frame, &video_desc) == 0) {
		switch (label) {		
		case ACTION0:
			type = DOUBLE;			
			break;
		case ACTION1:
		case ACTION2:
			type = SINGLE;
			break;
		default:
			*angle = cur_angle_;
			return;
		}
		sin = std::max<float>(-1.0f, std::min<float>(1.0, sin_estimator_[type].predict(video_desc)));
		cos = std::max<float>(-1.0f, std::min<float>(1.0, cos_estimator_[type].predict(video_desc)));
		cur_angle_ = cv::fastAtan2(sin, cos);
	}
	*angle = cur_angle_;
}

