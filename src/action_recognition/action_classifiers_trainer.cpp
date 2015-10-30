#include "action_classifiers_trainer.h"

int StoreTrjDesc(const char *video_path_char, const char *desc_path_char) {
	fs::path	output_dir;	
	fs::path	video_path(video_path_char);
	fs::path	video_basename = fs::basename(video_path);

	CHECK(fs::exists(video_path));
	// open descriptors files
	for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
		std::string output_file;
		output_dir = fs::path(desc_path_char)/fs::path(ActionClassifier::desc_names_[desc_type]);

		// if doesn't exist create a folder for the descriptors
		try {
			if (fs::is_directory(output_dir) == false)
				fs::create_directories(output_dir);
		} catch(boost::filesystem::filesystem_error e) { 
			LOGF() << "Failed to create dense trajectories descriptors folder: " << output_dir << std::endl;
			return -1;
		}
	}

	// for every angle extract trajectories descriptors from the video and store each frame's 
	// descriptors in a matrix format (row per trajectory) in a file
	for (float angle = 0.0f; angle < 360.0f; angle = angle + ANGLE_RESOLUTION) {
		DenseTrack			dense_track;
		cv::Mat				frame;
		VideoHandle			video_handle;
		int					frame_num = 0, desc_cnt = 0;
		std::ostringstream	angle_strm;
		MatWriter			mw[DESCRIPTORS_NUM];
		int					skip_cnt = 0;

		angle_strm << std::setfill('0') << std::setw(3) << (int)angle;
		// skip if output already exists
		for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
			fs::path output_file;			

			output_dir = fs::path(desc_path_char)/fs::path(ActionClassifier::desc_names_[desc_type]);			
			output_file = output_dir/fs::path(video_basename.string() + "_rotation_" + angle_strm.str() + ".dat");

			if (fs::exists(output_file))
				skip_cnt++;
		}	
		
		if (skip_cnt == DESCRIPTORS_NUM) {
			LOGD() << video_basename.string() << "trajectories with " << angle_strm.str() << " rotation already exists. Skipping" << std::endl;
			continue;
		}

		// initialize serializers for every descriptor		
		CHECK(video_handle.Init(video_path_char, angle, OP_RECT_SIZE) == 0);
		LOGI() << "Extracting descriptors from " << video_path << " with " << angle << " rotation" << std::endl;

		for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
			fs::path output_file;			

			output_dir = fs::path(desc_path_char)/fs::path(ActionClassifier::desc_names_[desc_type]);			
			output_file = output_dir/fs::path(video_basename.string() + "_rotation_" + angle_strm.str() + ".dat");
			LOGI() << "Creating " << output_file.string() << std::endl;
			CHECK(mw[desc_type].Init(output_file.string()) == 0);
		}		

		// extract and serialize frames
		while (video_handle.GetFrame(&frame) == 0) {			
			cv::Mat descriptor_set[DESCRIPTORS_NUM];			

			desc_cnt += dense_track.Extract(frame, descriptor_set);
			for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++)
				mw[desc_type].Write(descriptor_set[desc_type]);			

			if ((frame_num % 20) == 0) {
				std::stringstream log_msg;
				log_msg << frame_num << " frames processed, " << desc_cnt << " descriptors stored";
				LOGD() << log_msg.str() << std::endl;
			}
			frame_num++;
		}//video frames		
	}//video rotation angles	

	return 0;
}

int CalcCodebooks(const char *desc_path_char, const char *desc_type_char, const char *cb_path_char) {
	cv::Mat					desc_set, labels_set, centers;			
	int						desc_dim;	
	fs::path				cb_path, desc_path;
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*dat");
	DescType				cb_desc_type = DESCRIPTORS_NUM;
	bool					valid_desc_type = false;
	RandGen					rand_gen;
	MatWriter				mw;	

	// decode codebook type
	desc_path = fs::path(desc_path_char)/fs::path(desc_type_char);
	CHECK(fs::exists(desc_path));
	for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
		if (std::string(desc_type_char) == ActionClassifier::desc_names_[desc_type]) {
			cb_desc_type = static_cast<DescType>(desc_type);
			valid_desc_type = true;
			break;
		}
	}
	CHECK(valid_desc_type);
	cb_path = fs::path(cb_path_char);
	// if doesn't exist create a folder for codebooks
	try {
		if (fs::is_directory(cb_path) == false)
			fs::create_directories(cb_path);
	} catch(boost::filesystem::filesystem_error e) { 
		LOGF() << "Failed to create codebooks folder: " << cb_path << std::endl;
		return -1;
	}	
	cb_path = cb_path/fs::path("cb_" + std::string(ActionClassifier::desc_names_[cb_desc_type]) + ".dat");

	if (fs::exists(cb_path)) {
		LOGI() << std::string(desc_type_char) << "code-book already exists." << std::endl;
		return 0;
	}

	// iterate through different actions at different rotations until sufficient number of descriptors is collected
	LOGI() << "Collecting descriptors for " << desc_type_char << " codebook" << std::endl;	
	while (desc_set.rows < CB_MAX_DESCRIPTOR_NUM) {
		for(fs::directory_iterator dir_iter(desc_path) ; (dir_iter != eod) && (desc_set.rows < CB_MAX_DESCRIPTOR_NUM); ++dir_iter) {			
			if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {
				MatReader mr;
				cv::Mat frame_desc;
				int desc_cnt = 0;

				if (rand_gen.RandInt(100) > CB_ANGLE_REJECTION_RATE)
					continue;			
				
				LOGI() << "Extracting descriptors from " << fs::basename(dir_iter->path()) << std::endl;				
                if (mr.Init(dir_iter->path().string()) != 0) {
                    LOGW() << "Failed to open " << fs::basename(dir_iter->path()) << std::endl;
                    continue;
                }
                                    
				while ((mr.Read(&frame_desc) == 0) && (desc_set.rows < CB_MAX_DESCRIPTOR_NUM)){
					for (int desc_idx = 0; desc_idx < frame_desc.rows; desc_idx++) {
						if (rand_gen.RandInt(100) > CB_DESCRIPTOR_REJECTION_RATE)
							continue;
						desc_set.push_back(frame_desc.row(desc_idx));
						desc_cnt ++;						

						if (desc_set.rows == CB_MAX_DESCRIPTOR_NUM)	{
							break;
						}
					}
				}
				LOGI() << desc_cnt << " descriptors added, total descriptors num: " << desc_set.rows << std::endl;
			}
		}
	}	

	// calculate codebook words
	desc_dim = desc_set.cols;
	LOGI() << "kmeans clustering, max iterations: " << CB_KMEAS_MAX_ITER << std::endl;
	kmeans(desc_set, BOW_SIZE, labels_set, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, CB_KMEAS_MAX_ITER, 0.01*desc_dim), 5, cv::KMEANS_PP_CENTERS, centers);
	LOGI() << "kmeans complete" << std::endl;

	CHECK(mw.Init(cb_path.string()) == 0);
	mw.Write(centers);

	return 0;
}

int StoreActionDesc(const char *trj_desc_path_char, const char *cb_path_char, const char *video_name_char, const char *action_desc_path_char) {
	for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
		fs::path				trj_desc_path, action_desc_path, cb_path;	
		cv::Mat					trj_desc, action_desc_acc, codebook;
		MatReader				mr;
		MatWriter				mw;
		CyclicQueue<cv::Mat>	frame_desc_q(WINDOW_LENGTH);
		int						frame_cnt = 0, desc_cnt = 0;
		
		action_desc_acc = cv::Mat::zeros(BOW_SIZE, 1, CV_32FC1);
		cb_path = fs::path(cb_path_char)/fs::path("cb_" + std::string(ActionClassifier::desc_names_[desc_type]) + ".dat");		
		CHECK(mr.Init(cb_path.string()) == 0);
		mr.Read(&codebook);
		mr.Deinit();

		trj_desc_path = fs::path(trj_desc_path_char)/fs::path(ActionClassifier::desc_names_[desc_type])/fs::path(video_name_char);
		CHECK(fs::exists(trj_desc_path));
		CHECK(mr.Init(trj_desc_path.string()) == 0);

		action_desc_path = fs::path(action_desc_path_char)/fs::path(ActionClassifier::desc_names_[desc_type]);
		// if doesn't exist create a folder for action descriptors
		try {
			if (fs::is_directory(action_desc_path) == false)
				fs::create_directories(action_desc_path);
		} catch(boost::filesystem::filesystem_error e) { 
			LOGF() << "Failed to create action descriptors folder: " << action_desc_path << std::endl;
			return -1;
		}
		action_desc_path = action_desc_path/fs::path(video_name_char);
		if (fs::exists(action_desc_path)) {
			LOGI() << ActionClassifier::desc_names_[desc_type] << " action descriptor " << 
				" for " << std::string(video_name_char) << " already exists. Skipping" << std::endl;
			continue;
		}
		CHECK(mw.Init(action_desc_path.string()) == 0);
		LOGI() << "Extracting action descriptors from " << fs::basename(action_desc_path) << std::endl;

		while (mr.Read(&trj_desc) == 0) {
			cv::Mat frame_desc, action_desc;

			if (frame_cnt >= g_track_length) {
				//build features histogram for current frame (frame descriptor)
				ActionClassifier::BuildFrameBoW(trj_desc, codebook, &frame_desc);			
				//add the frame descriptor to video descriptor 
				if (ActionClassifier::BuildVideoBow(frame_desc, &frame_desc_q, &action_desc_acc, &action_desc) == 0) {
					mw.Write(action_desc);
					desc_cnt++;
				}
			}
			frame_cnt++;
		}
		LOGI() << desc_cnt << " " << ActionClassifier::desc_names_[desc_type] << " descriptors stored" << std::endl;
	}

	return 0;
}

int SplitDataSet(const char *base_path_char, const char *split_path_char, float split_ratio) {
	RandGen rand_gen;
	fs::path				base_path(base_path_char), split_path(split_path_char), desc_base_path[DESCRIPTORS_NUM], desc_split_path[DESCRIPTORS_NUM];
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*dat");

	CHECK(fs::exists(base_path));
	// if doesn't exist create a split folder
	try {
		if (fs::is_directory(split_path) == false)
			fs::create_directories(split_path);
	} catch(boost::filesystem::filesystem_error e) { 
		LOGF() << "Failed to create split folder: " << split_path << std::endl;
		return -1;
	}

	for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
		desc_base_path[desc_type] = base_path/fs::path(ActionClassifier::desc_names_[desc_type]);
		desc_split_path[desc_type] = split_path/fs::path(ActionClassifier::desc_names_[desc_type]);
		CHECK(fs::exists(desc_base_path[desc_type]));
		// if doesn't exist create a split folder
		try {
			if (fs::is_directory(desc_split_path[desc_type]) == false)
				fs::create_directories(desc_split_path[desc_type]);
		} catch(boost::filesystem::filesystem_error e) { 
			LOGF() << "Failed to create split folder: " << desc_split_path[desc_type] << std::endl;
			return -1;
		}
	}
	
	for(fs::directory_iterator dir_iter(desc_base_path[TRJ]) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {
			// randomly split base folder, move ALL descriptors for a specific action
			if (rand_gen.RandFloat(1.0f) < split_ratio) {
				std::string filename = dir_iter->path().filename().string();

				for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
					fs::path org_path = desc_base_path[desc_type]/fs::path(filename);
					fs::path new_path = desc_split_path[desc_type]/fs::path(filename);
					LOGI() << "Moving " << org_path.string() << " to " << new_path.string() << std::endl;
					fs::rename(org_path, new_path);					
				}
			}
		}
	}
	
	return 0;
}

/**
   * @brief Read descriptors folder to memory
   *
   * @param data_path data path
   * @param data data in cv::Mat format
   * @param labels in cv::Mat format
   */
static void GetBaseDataSet(const fs::path &data_path, cv::Mat *data, cv::Mat *labels) {
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*dat");

	// collect all action descriptors	
	for(fs::directory_iterator dir_iter(data_path) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {
			MatReader	mr;
			std::string filename;
			cv::Mat		desc;
			float		label = -1;

			filename = fs::basename(dir_iter->path());
			for (int action_type = 0; action_type < ACTIONS_NUM; action_type++) {
				if (filename.find(ActionClassifier::actions_names_[action_type]) != std::string::npos) {
					label = (float)action_type;
					break;
				}
			}
			if (label < 0)
				continue;

			
			LOGI() << "Extracting descriptors from " << filename << std::endl;
			CHECK(mr.Init(dir_iter->path().string()) == 0);
			while (mr.Read(&desc) == 0) {				
				for (int desc_idx = 0; desc_idx < desc.cols; desc_idx++) {					
					data->push_back(desc.col(desc_idx).reshape(1,1));
					labels->push_back(label);					
				}
			}			
		}
	}
}

/**
   * @brief Read base descriptors and convert them to combined format
   *
   * @param data_path data path
   * @param classifiers_path path to base classifiers
   * @param data data in cv::Mat format
   * @param labels in cv::Mat format
   */
static void GetCombinedDataSet(const fs::path &data_path, const fs::path &classifiers_path, cv::Mat *data, cv::Mat *labels) {
	cv::Mat					base_desc_set[DESCRIPTORS_NUM], base_labels_set[DESCRIPTORS_NUM];
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*dat");
	cv::SVM					base_classifiers[DESCRIPTORS_NUM][ACTIONS_NUM], combined_classifier;	
	int						data_set_size = 0;

	for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
		fs::path desc_data_path, classifier_path;		
		
		desc_data_path = data_path/fs::path(ActionClassifier::desc_names_[desc_type]);
		CHECK(fs::exists(desc_data_path));	
		GetBaseDataSet(desc_data_path, &base_desc_set[desc_type], &base_labels_set[desc_type]);
		for (int action_type = 0; action_type < ACTIONS_NUM; action_type++) {
			classifier_path = classifiers_path/fs::path(std::string(ActionClassifier::desc_names_[desc_type]) + "_" + 
				std::string(ActionClassifier::actions_names_[action_type]) + "_classifier.model");
			base_classifiers[desc_type][action_type].load(classifier_path.string().c_str());
		}
	}

	data_set_size = base_desc_set[0].rows;
	for (int desc_idx = 0; desc_idx < data_set_size; desc_idx++) {
		cv::Mat base_desc[DESCRIPTORS_NUM], combined_desc;
		float combined_label;
		
		combined_label = base_labels_set[0].at<float>(desc_idx);
		for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
			base_desc[desc_type] = base_desc_set[desc_type].row(desc_idx);
		}
		ActionClassifier::CombineDesc(base_desc, base_classifiers, &combined_desc);
	
		data->push_back(combined_desc);
		labels->push_back(combined_label);
	}
}

int TrainBaseActionClassifiers(const char *train_data_path_char, const char *test_data_path_char, const char *desc_type_char, const char *output_path_char) {
	cv::Mat					train_data, train_labels;			
	fs::path				output_dir, train_data_path, test_data_path;
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*dat");
	DescType				class_desc_type = DESCRIPTORS_NUM;
	bool					valid_desc_type = false;
	CvSVM					multiclass_classifier; 
	CvSVMParams				svm_params;
	
	// decode descriptor type
	train_data_path = fs::path(train_data_path_char)/fs::path(desc_type_char);
	CHECK(fs::exists(train_data_path));
	test_data_path = fs::path(test_data_path_char)/fs::path(desc_type_char);
	CHECK(fs::exists(test_data_path));
	for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
		if (std::string(desc_type_char) == ActionClassifier::desc_names_[desc_type]) {
			class_desc_type = static_cast<DescType>(desc_type);//ActionClassifier::desc_types_[desc_type];
			valid_desc_type = true;
			break;
		}
	}
	CHECK(valid_desc_type);
	output_dir = fs::path(output_path_char);
	// if doesn't exist create a folder for classifiers
	try {
		if (fs::is_directory(output_dir) == false)
			fs::create_directories(output_dir);
	} catch(boost::filesystem::filesystem_error e) { 
		LOGF() << "Failed to create classifiers folder: " << output_dir << std::endl;
		return -1;
	}	
	
	svm_params.svm_type = cv::SVM::C_SVC;
	svm_params.kernel_type = cv::SVM::LINEAR;
	GetBaseDataSet(train_data_path, &train_data, &train_labels);	
	for (int action_type = 0; action_type < ACTIONS_NUM; action_type++) {		
		cv::Mat bin_labels;
		CvSVM SVM; 
		fs::path output_path, report_file_path;
		FILE* report_file;		
			
		output_path = output_dir/fs::path(std::string(ActionClassifier::desc_names_[class_desc_type]) + "_" + 
			std::string(ActionClassifier::actions_names_[action_type]) + "_classifier.model");	

		if (fs::exists(output_path)) {
			LOGI() << std::string(ActionClassifier::desc_names_[class_desc_type]) << 
				" classifier for " << std::string(ActionClassifier::actions_names_[action_type]) << "already exists. Skipping" << std::endl;
			continue;
		}

		for (int desc_idx = 0; desc_idx < train_labels.rows; desc_idx++) {			
			bin_labels.push_back((action_type == (int)train_labels.at<float>(desc_idx))? 1.0f: -1.0f);
		}			
		
		LOGI() << output_path.string() << std::endl;
		LOGI() << "Training " << desc_type_char << " " << ActionClassifier::actions_names_[action_type] << " classifier..." << std::endl;		
		SVM.train_auto(train_data, bin_labels, cv::Mat(), cv::Mat(), svm_params);				
		SVM.save(output_path.string().c_str());
		
		report_file_path = output_dir/fs::path(std::string(ActionClassifier::desc_names_[class_desc_type]) + "_" + 
			std::string(ActionClassifier::actions_names_[action_type]) + "_classifier_train_results.txt");
		report_file = fopen(report_file_path.string().c_str(), "w");
		TestBaseActionClassifier(output_path_char, train_data_path.string().c_str(), class_desc_type, action_type, report_file);			
		fclose(report_file);

		report_file_path = output_dir/fs::path(std::string(ActionClassifier::desc_names_[class_desc_type]) + "_" + 
			std::string(ActionClassifier::actions_names_[action_type]) + "_classifier_test_results.txt");
		report_file = fopen(report_file_path.string().c_str(), "w");
		TestBaseActionClassifier(output_path_char, test_data_path.string().c_str(), class_desc_type, action_type, report_file);
		fclose(report_file);		
	}

	return 0;
}

void TestAllBaseClassifiers(const char *classifiers_path_char, const char *test_data_path_char) {
	fs::path classifiers_path(classifiers_path_char);

	for (int desc_type = 0; desc_type < DESCRIPTORS_NUM; desc_type++) {
		for (int action_type=0; action_type < ACTIONS_NUM; action_type++) {			
			fs::path classifier_path, report_file_path;
			FILE* report_file;
			fs::path test_data_path(test_data_path_char);
			
			test_data_path = test_data_path/fs::path(ActionClassifier::desc_names_[desc_type]);

			classifier_path = classifiers_path/fs::path(std::string(ActionClassifier::desc_names_[desc_type]) + "_" + 
				std::string(ActionClassifier::actions_names_[action_type]) + "_classifier.model");
			
			report_file_path = classifiers_path/fs::path(std::string(ActionClassifier::desc_names_[desc_type]) + "_" + 
				std::string(ActionClassifier::actions_names_[action_type]) + "_classifier_test_results.txt");
			report_file = fopen(report_file_path.string().c_str(), "w");
			TestBaseActionClassifier(classifiers_path_char, test_data_path.string().c_str(), desc_type, action_type, report_file);
			fclose(report_file);
		}
	}
}

int TestBaseActionClassifier(const char *classifiers_path_char, const char *test_data_path_char, const int desc_type,  const int action_type, FILE *report_file) {
	fs::path test_data_path, classifiers_path;
	cv::Mat data, labels;
	int confusion_matrix[2][2] = {0};
	float recall, specificity, precision, accuracy;
	cv::SVM SVM;

	test_data_path = fs::path(test_data_path_char);	
	CHECK(fs::exists(test_data_path));	

	classifiers_path = fs::path(classifiers_path_char)/fs::path(std::string(ActionClassifier::desc_names_[desc_type]) + "_" + 
		std::string(ActionClassifier::actions_names_[action_type]) + "_classifier.model");	
	CHECK(fs::exists(classifiers_path));
	SVM.load(classifiers_path.string().c_str());

	GetBaseDataSet(test_data_path, &data, &labels);	
	fprintf(report_file, "Scores:\n");
	for (int desc_idx = 0; desc_idx < data.rows; desc_idx++) {
		float label;
		int true_idx, predicted_idx;

		true_idx = (labels.at<float>(desc_idx) == action_type)? 0 : 1;
		label = SVM.predict(data.row(desc_idx));												
		predicted_idx = (label == 1.0f) ? 0 : 1;
		confusion_matrix[predicted_idx][true_idx]++;		
	}
	fprintf(report_file, "\n");
	

	for (int row=0; row<2; row++)
	{
		for (int col=0; col<2; col++)
			printf("%d\t\t",confusion_matrix[row][col]);
		printf("\n");
	}
	
	recall = (float)confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0]);
	specificity = (float)confusion_matrix[1][1]/(confusion_matrix[0][1] + confusion_matrix[1][1]);
	precision = (float)confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1]);
	accuracy = (float)(confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1]);
	printf("Recall: %2.4f, Specificity: %2.4f, Precision: %2.4f, Accuracy: %2.4f\n\n\n",recall, specificity, precision, accuracy);	

	
	for (int row=0; row<2; row++)
	{
		for (int col=0; col<2; col++)
			fprintf(report_file,"%d\t\t",confusion_matrix[row][col]);
		fprintf(report_file,"\n");
	}	
	fprintf(report_file,"Recall: %2.4f, Specificity: %2.4f, Precision: %2.4f, Accuracy: %2.4f\n\n\n", recall, specificity, precision, accuracy);
	

	return 0;
}

int TrainCombinedActionClassifier(const char *train_data_path_char, const char *test_data_path_char, const char *classifiers_path_char) {
	cv::Mat		combined_desc_set, combined_labels_set;
	fs::path	classifiers_path, train_data_path, test_data_path, output_path, report_file_path;	
	cv::SVM		combined_classifier;
	CvSVMParams	svm_params;
	FILE		*report_file;	

	classifiers_path = fs::path(classifiers_path_char);
	CHECK(fs::exists(classifiers_path));
	train_data_path = fs::path(train_data_path_char);
	CHECK(fs::exists(train_data_path));

	GetCombinedDataSet(train_data_path, classifiers_path, &combined_desc_set, &combined_labels_set);

	output_path = classifiers_path/fs::path("combined_classifier.model");
	if (fs::exists(output_path)) {
		LOGI() << "combined action classifier already exists. Skipping" << std::endl;
		return 0;
	}

	LOGI() << "Training combined classifier..." << std::endl;
	svm_params.svm_type = cv::SVM::C_SVC;
	svm_params.kernel_type = cv::SVM::LINEAR;

	combined_classifier.train_auto(combined_desc_set, combined_labels_set, cv::Mat(), cv::Mat(), svm_params);				
	combined_classifier.save(output_path.string().c_str());

	test_data_path = fs::path(test_data_path_char);
	CHECK(fs::exists(test_data_path));

	report_file_path = classifiers_path/fs::path("combined_multiclass_classifier_test_results.txt");
	report_file = fopen(report_file_path.string().c_str(), "w");

	TestCombinedActionClassifier(classifiers_path_char, test_data_path_char, report_file);
	fclose(report_file);

	return 0;
}

int TestMultiClassClassifier(const cv::SVM &SVM, const cv::Mat &data, const cv::Mat labels, FILE *report_file) {
	int confussion_matrix[ACTIONS_NUM][ACTIONS_NUM] = {0};
	float accuracy = 0.0f;

	for (int desc_idx = 0; desc_idx < data.rows; desc_idx++) {
		float label;
		int true_idx, predicted_idx;

		true_idx = (int)labels.at<float>(desc_idx);
		label = SVM.predict(data.row(desc_idx));		
		predicted_idx = (int)label;
		confussion_matrix[predicted_idx][true_idx]++;
		accuracy += (true_idx == predicted_idx)? 1: 0;
	}
	printf("============================================\n");
	fprintf(report_file, "============================================\n");
	
	accuracy /= data.rows;
	printf("accuracy: %2.2f\n", accuracy);
	fprintf(report_file, "accuracy: %2.2f\n", accuracy);

	for (int row = 0; row < ACTIONS_NUM; row++) {
		for (int col=0; col < ACTIONS_NUM; col++) {
			printf("%d\t\t",confussion_matrix[row][col]);
			fprintf(report_file,"%d\t\t",confussion_matrix[row][col]);
		}
		printf("\n");
		fprintf(report_file,"\n");
	}
	return 0;
}

int TestCombinedActionClassifier(const char *classifiers_path_char, const char *test_data_path_char, FILE *report_file) {
	fs::path test_data_path, classifiers_path;
	cv::Mat data, labels;
	cv::SVM SVM;

	test_data_path = fs::path(test_data_path_char);	
	CHECK(fs::exists(test_data_path));
	classifiers_path = fs::path(classifiers_path_char);
	CHECK(fs::exists(classifiers_path));
	SVM.load((classifiers_path/fs::path("combined_classifier.model")).string().c_str());

	GetCombinedDataSet(test_data_path, classifiers_path, &data, &labels);
	TestMultiClassClassifier(SVM, data, labels, report_file);

	return 0;
}
