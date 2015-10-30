#ifndef ORIENTATION_TRAINER_H
#define ORIENTATION_TRAINER_H

#include "orientation.h"

/**
   * @brief Store action orientation descriptors with video rotation.
   *
   * @param input_path_str input video file path
   * @param output_path_str path to output folder
   */
int StoreOrientationDesc(std::string input_path_str, std::string output_path_str);

/**
	* @brief Train action orientation estimator
	*
	* @param train_data_dir_str train data path
	* @param test_data_dir_str test data path
	* @param output_dir_str path to store the estimator
	*/
int TrainOrientationEstimator(std::string train_data_dir_str, std::string test_data_dir_str, std::string output_dir_str);
int TrainOrientationCosEstimator(std::string train_data_dir_str, std::string test_data_dir_str, std::string output_dir_str);
int TrainOrientationSinEstimator(std::string train_data_dir_str, std::string test_data_dir_str, std::string output_dir_str);

/**
	* @brief Test action orientation estimator
	*
	* @param test_dir_str test data path
	* @param estimators_dir_str estimators path
	*/
void TestOrientationEstimator(std::string test_dir_str, std::string estimators_dir_str);

#endif //ORIENTATION_TRAINER_H