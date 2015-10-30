#include "action_classifiers.h"

//private static members
cv::Mat	ActionClassifier::codebooks_[DESCRIPTORS_NUM];
cv::SVM	ActionClassifier::base_classifiers_[DESCRIPTORS_NUM][ACTIONS_NUM];	
cv::SVM	ActionClassifier::combined_classifier_;
bool ActionClassifier::statics_loaded_ = false;

//public static members
std::string ActionClassifier::desc_names_[DESCRIPTORS_NUM] = {"TRJ", "HOF", "HOG", "MBH"};
//std::string ActionClassifier::actions_names_[ACTIONS_NUM] = {"direction", "lrotation", "rrotation", "background"};
std::string ActionClassifier::actions_names_[ACTIONS_NUM] = {"action0", "action1", "action2", "background"};

ActionClassifier::ActionClassifier() {	
	frame_desc_q_ = std::vector<CyclicQueue<cv::Mat> >(DESCRIPTORS_NUM, CyclicQueue<cv::Mat>(WINDOW_LENGTH));
	//set descriptors accumulator to zero
	for (int desc_idx=0; desc_idx<DESCRIPTORS_NUM; desc_idx++)
		video_desc_[desc_idx] = cv::Mat::zeros(BOW_SIZE, 1, CV_32FC1);
	cur_label_ = BACKGROUND;
	initialized_ = false;
}

void ActionClassifier::LoadStaticMembers(const char *codebooks_path_char, const char *classifiers_path_char) {
	const fs::path cb_dir(codebooks_path_char), classifiers_dir(classifiers_path_char);
	
	//load codebooks and classifiers
	for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
		MatReader mr;
		fs::path cb_path = cb_dir/fs::path("cb_" + std::string(desc_names_[desc_type]) + ".dat");

		CHECK(fs::exists(cb_path));		
		CHECK(mr.Init(cb_path.string()) == 0);
		mr.Read(&codebooks_[desc_type]);
		for (int action_type = 0; action_type < ACTIONS_NUM; action_type++) {
			fs::path classifier_path;
			classifier_path = classifiers_dir/fs::path(std::string(desc_names_[desc_type]) + "_" + 
				std::string(actions_names_[action_type]) + "_classifier.model");
			CHECK(fs::exists(classifier_path));
			base_classifiers_[desc_type][action_type].load(classifier_path.string().c_str());
		}
	}
	combined_classifier_.load((classifiers_dir/fs::path("combined_classifier.model")).string().c_str());
	statics_loaded_ = true;
}

void ActionClassifier::Init(const cv::Mat &frame) {	
	CHECK(statics_loaded_);	
	//initialize dense trajectories extractor
	dense_track_.Init(frame);	
	initialized_ = true;
}

int	ActionClassifier::Extract(const cv::Mat &frame, cv::Mat frame_desc[DESCRIPTORS_NUM], cv::Mat video_desc[DESCRIPTORS_NUM]) {	
	cv::Mat descriptors_set[DESCRIPTORS_NUM];
	int rc = -1;

	if (initialized_ == false) {
		Init(frame);
		return -1;
	}
	
	//extract dense trajectories 
	if (dense_track_.Extract(frame, descriptors_set) > 0) {
		cv::Mat combined_desc;
		for (int desc_idx=0; desc_idx<DESCRIPTORS_NUM; desc_idx++) {
			//convert each descriptor type to BoW
			BuildFrameBoW(descriptors_set[desc_idx], codebooks_[desc_idx], &frame_desc[desc_idx]);			
			//add the frame descriptor to video descriptor										
			if (BuildVideoBow(frame_desc[desc_idx], &frame_desc_q_[desc_idx], &video_desc_[desc_idx], &video_desc[desc_idx]) == 0) {				
				rc = 0;										
			}
		}
	}
	return rc;
}

void ActionClassifier::ClassifyFrame(const cv::Mat &frame, float *label) {
	cv::Mat base_desc[DESCRIPTORS_NUM], frame_desc[DESCRIPTORS_NUM];

	if (Extract(frame, frame_desc, base_desc) == 0) {
		cv::Mat combined_desc;
		CombineDesc(base_desc, base_classifiers_, &combined_desc);
		cur_label_ = combined_classifier_.predict(combined_desc);		
	}
	
	*label = cur_label_;
}

void ActionClassifier::BuildFrameBoW(const cv::Mat &trajectories, const cv::Mat &codebook, cv::Mat *frame_desc) {
	cv::BFMatcher			matcher(cv::NORM_L2, false);
	std::vector<cv::DMatch>	matches;
	std::vector<float>		desc(BOW_SIZE, 0.0f);
	//build a histogram of codebook words
	matcher.match(trajectories, codebook, matches);
	for (int i=0; i < (int)matches.size(); i++)
		desc[matches[i].trainIdx]++;	

	//convert the vector to a matrix and create new copy of it
	frame_desc->push_back(cv::Mat(desc));
}


int ActionClassifier::BuildVideoBow(const cv::Mat &frame_desc, CyclicQueue<cv::Mat> *frame_desc_q, cv::Mat *video_desc, cv::Mat *new_video_desc) {
	// add the new descriptor to both the queue and the accumulator
	frame_desc_q->enqueue(frame_desc);					
	(*video_desc) += frame_desc;

	// when the time window is complete copy the current accumulator to the output then remove the oldest descriptors from both the queue and the accumulator
	if (frame_desc_q->full()) {
		(*video_desc).copyTo((*new_video_desc));

		for (int i=0; i<(WINDOW_LENGTH - WINDOW_OVERLAP); i++) {
			cv::Mat desc;
			frame_desc_q->dequeue(&desc);
			(*video_desc) -= desc;
		}	
		return 0;
	}
	return -1;
}

void ActionClassifier::CombineDesc(const cv::Mat base_desc[DESCRIPTORS_NUM], const cv::SVM base_classifiers[DESCRIPTORS_NUM][ACTIONS_NUM], cv::Mat *combined_desc) {
	cv::Mat desc;
	for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
		for (int action_type = 0; action_type < ACTIONS_NUM; action_type++) {							
			float score;
			score = base_classifiers[desc_type][action_type].predict(base_desc[desc_type], true);
			desc.push_back(score);
		}
	}
	cv::normalize(desc.reshape(1,1), *combined_desc);
}

