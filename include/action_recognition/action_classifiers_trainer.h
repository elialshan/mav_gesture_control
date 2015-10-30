#ifndef ACTION_CLASSIFIERS_TRAINER_H
#define ACTION_CLASSIFIERS_TRAINER_H

#include "dense_track.h"
#include "action_classifiers.h"
#include "utils.h"


// number of descriptors used for bag of word codebook learning
#define CB_MAX_DESCRIPTOR_NUM			(100000)
// probability of a rotation angle to be picked for codebook learning
#define CB_ANGLE_REJECTION_RATE		(50)
// probability of a descriptor to be picked for codebook learning
#define CB_DESCRIPTOR_REJECTION_RATE	(10)
// kmeans iterations number
#define CB_KMEAS_MAX_ITER				(15)

/**
   * @brief Store video trajectories descriptors with video rotation.
   *
   * @param video_path_char input video file path
   * @param desc_path_char path to output folder
   */
int StoreTrjDesc(const char *video_path_char, const char *desc_path_char);

/**
   * @brief Calculate bag of words codebook for a given descriptor
   *
   * @param desc_path_char path to trajectories descriptors folder
   * @param desc_type_char descriptor type (trajectory, HOG, HOF, MBH)
   * @param cb_path_char path to store the codebook
   */
int CalcCodebooks(const char *desc_path_char, const char *desc_type_char, const char *cb_path_char);

/**
   * @brief Convert and store frame's trajectories descriptors to time windowed bag of words descriptors
   *
   * @param trj_desc_path_char path to trajectories descriptors folder
   * @param cb_path_char path to codebook folder
   * @param video_name_char input video name and rotation
   * @param action_desc_path_char output folder path   
   */
int StoreActionDesc(const char *trj_desc_path_char, const char *cb_path_char, const char *video_name_char, const char *action_desc_path_char);

/**
   * @brief Split data set by a given ratio (used to split training data to base and combined sets)
   *
   * @param base_path_char original descriptors folder
   * @param split_path_char new descriptors folder
   * @param split_raio split ratio
   */
int SplitDataSet(const char *base_path_char, const char *split_path_char, float split_ratio);

/**
	* @brief Train base action classifiers for a specific descriptors type
	*
	* @param train_data_path_char train data path
	* @param test_data_path_char test data path
	* @param desc_type_char descriptor type
	* @param output_path_char path to store the classifier
	*/
int TrainBaseActionClassifiers(const char *train_data_path_char, const char *test_data_path_char, const char *desc_type_char, const char *output_path_char);

/**
	* @brief Test all base classifiers
	*
	* @param classifiers_path_char classifiers path
	* @param test_data_path_char test data path
	*/
void TestAllBaseClassifiers(const char *classifiers_path_char, const char *test_data_path_char);

/**
	* @brief Test base classifier
	*
	* @param classifiers_path_char classifiers path
	* @param test_data_path_char test data path
	* @param desc_type descriptor type
	* @param action_type action type
	* @param report_file report file file descriptor
	*/
int TestBaseActionClassifier(const char *classifiers_path_char, const char *test_data_path_char, const int desc_type,  const int action_type, FILE *report_file);

/**
	* @brief Train combined classifier
	*
	* @param train_data_path_char train data path
	* @param test_data_path_char test data path
	* @param output_path_char path to store the classifier
	*/
int TrainCombinedActionClassifier(const char *train_data_path_char, const char *test_data_path_char, const char *classifiers_path_char);

/**
	* @brief Test multi-class classifier
	*
	* @param SVM classifier under test
	* @param data test data
	* @param labels ground truth
	* @param report_file report file file descriptor
	*/
int TestMultiClassClassifier(const cv::SVM &SVM, const cv::Mat &data, const cv::Mat labels, FILE *report_file);
/**
	* @brief Test combined classifier
	*
	* @param classifiers_path_char classifiers path
	* @param test_data_path_char test data path
	* @param report_file report file file descriptor
	*/
int TestCombinedActionClassifier(const char *classifiers_path_char, const char *test_data_path_char, FILE *report_file);

#endif//ACTION_CLASSIFIERS_TRAINER_H