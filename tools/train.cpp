#include "utils.h"
#include "tracker.h"
#include "action_classifiers_trainer.h"
#include "orientation_trainer.h"




int main(int argc, char** argv) {
	CHECK(argc > 2);
	if (argv[1] == std::string("train_op_class")) {		
		CHECK(argc == 4);		
		TrainOperatorClassifier((fs::path(argv[2])/"train").string().c_str(), argv[3]);	
		TestOperatorClassifier((fs::path(argv[2])/"test").string().c_str(), argv[3]);	
		return 0;
	}
        if (argv[1] == std::string("evaluate_detection")) {		
            char *input_video_path, *models_path, *gt_path, *output_path;
            CHECK(argc == 6);	
            input_video_path = argv[2];
            models_path = argv[3];
            gt_path = argv[4];
            output_path = argv[5];
            return EvaluateDetection(input_video_path, models_path, gt_path, output_path);
	}
	if (argv[1] == std::string("extract_trj")) {		
		CHECK(argc == 4);
		return StoreTrjDesc(argv[2], argv[3]);		
	}
	else if (argv[1] == std::string("calc_cb")) {		
		CHECK(argc == 5);
		return CalcCodebooks(argv[2], argv[3], argv[4]);		
	}	
	else if (argv[1] == std::string("extract_action_desc")) {
		CHECK(argc == 6);
		return StoreActionDesc(argv[2], argv[3], argv[4], argv[5]);		
	}
	else if (argv[1] == std::string("split_data_set")) {
		CHECK(argc == 4);
		return SplitDataSet(argv[2], argv[3], 0.2f);		
	}
	else if (argv[1] == std::string("train_base_action")) {
		CHECK(argc == 6);
		ActionClassifier::actions_names_[ACTION0] = "direction";
		ActionClassifier::actions_names_[ACTION1] = "lrotation";
		ActionClassifier::actions_names_[ACTION2] = "rrotation"; 
		return TrainBaseActionClassifiers(argv[2], argv[3], argv[4], argv[5]);	
	}
	else if (argv[1] == std::string("train_combined_action")) {
		CHECK(argc == 5);
		ActionClassifier::actions_names_[ACTION0] = "direction";
		ActionClassifier::actions_names_[ACTION1] = "lrotation";
		ActionClassifier::actions_names_[ACTION2] = "rrotation"; 
		return TrainCombinedActionClassifier(argv[2], argv[3], argv[4]);
	}
	else if (argv[1] == std::string("extract_orientation")) {
		CHECK(argc == 4);
		return StoreOrientationDesc(std::string(argv[2]), std::string(argv[3]));
	}
	else if (argv[1] == std::string("train_orientation_estimator")) {
		CHECK(argc == 5);		
        return TrainOrientationEstimator(std::string(argv[2]), std::string(argv[3]), std::string(argv[4]));
	}
        else if (argv[1] == std::string("test_orientation_estimator")) {
		CHECK(argc == 4);
		TestOrientationEstimator(std::string(argv[2]), std::string(argv[3]));
        return 0;
	}	
	else
		LOGE() << "Undefined command " << argv[1] << std::endl;

	return -1;
}



