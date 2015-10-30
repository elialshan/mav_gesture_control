#include "orientation_trainer.h"

static float extract_angle(const char *filename) {
	const char *angle_str_ptr;
	char angle_str[7] = {0};
	angle_str_ptr = strstr(filename, "angle_") + 6;
	memcpy(angle_str, angle_str_ptr, 6);	
	return (float)atof(angle_str);
}


static float extract_scale(char *filename) {
	char *scale_str_ptr, scale_str[7] = {0};
	scale_str_ptr = strstr(filename, "scale_") + 6;
	memcpy(scale_str, scale_str_ptr, 4);	
	return (float)atof(scale_str);
}

static float extract_photo(char *filename) {
	char *photo_str_ptr, photo_str[7] = {0};
	photo_str_ptr = strstr(filename, "photo_") + 6;
	memcpy(photo_str, photo_str_ptr, 4);	
	return (float)atof(photo_str);
}

int StoreOrientationDesc(std::string input_path_str, std::string output_dir_str) {
	fs::path input_path(input_path_str), output_dir(output_dir_str);
	
	CHECK(fs::exists(input_path));
	// if doesn't exist create a folder for the descriptors
	try {
		if (fs::is_directory(output_dir) == false)
			fs::create_directories(output_dir);
	} catch(boost::filesystem::filesystem_error e) { 
		LOGF() << "Failed to create dense trajectories descriptors folder: " << output_dir;
		return -1;
	}

	std::string input_fname = fs::basename(input_path);
			
	LOGI() << "Extracting orientation descriptors form " << input_fname << std::endl;			
	for (float angle = 0.0f; angle < 360.0f; angle = angle + ANGLE_RESOLUTION) {			
		fs::path				output_path;
		VideoHandle				video;
		cv::Mat					frame, video_descs;
		std::stringstream		ss;
		OrientationEstimator	oe;
		MatWriter				mw;

		ss << boost::format("%s_angle_%06.2f.dat") %input_fname.c_str() %angle;
		output_path = output_dir/fs::path(ss.str());
		if (fs::exists(output_path)) {
			LOGI() << boost::format("%s, with angle: %2.2f already exists") %input_fname.c_str() %angle;	
			continue;
		}

		CHECK(video.Init(input_path.string().c_str(), angle, OP_RECT_SIZE) == 0);
		LOGI() << boost::format("angle: %2.2f") %angle;

		while (video.GetFrame(&frame) == 0){
			cv::Mat video_desc;
			if (oe.Extract(frame, &video_desc) == 0)
				video_descs.push_back(video_desc);						
		}						

		if (video_descs.empty()) {
			LOGI() << boost::format("Skipping %s, with angle: %2.2f no descriptors available") %input_fname.c_str() %angle;	
			break;
		}
					
		mw.Init(output_path.string());
		mw.Write(video_descs);
	}

	return 0;
}


int TrainOrientationEstimator(std::string train_data_dir_str, std::string test_data_dir_str, std::string output_dir_str) {
	fs::path input_dir(train_data_dir_str), output_dir(output_dir_str), output_path;
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*dat");
	cv::Mat				train_data, sin_labels, cos_labels;
	cv::SVM				svm; 
	cv::SVMParams			svm_params;

	CHECK(fs::exists(input_dir));
	// if doesn't exist create a folder for the descriptors
	try {
		if (fs::is_directory(output_dir) == false)
			fs::create_directories(output_dir);
	} catch(boost::filesystem::filesystem_error e) { 
		LOGF() << "Failed to create folder: " << output_dir << std::endl;
		return -1;
	}

	for(fs::directory_iterator dir_iter(input_dir) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {
			fs::path input_path = dir_iter->path();
			std::string input_fname = fs::basename(input_path);
			float angle, sin_label, cos_label;
			cv::Mat  desc;
			MatReader	mr;
		
			LOGI() << "Extracting data from " << input_fname << std::endl;		
			angle = extract_angle(input_fname.c_str());
			if (((int)angle % 20) != 0)
				continue;
		
			angle += 90;
			sin_label = sin((3.14f/180)*angle);
			cos_label = cos((3.14f/180)*angle);
			CHECK(mr.Init(input_path.string()) == 0);

			while (mr.Read(&desc) == 0) {
				for (int desc_idx = 0; desc_idx < desc.rows; desc_idx++) {
					train_data.push_back(desc.row(desc_idx));
					sin_labels.push_back(sin_label);			
					cos_labels.push_back(cos_label);
				}
			}
		}
	}	
	
	
	svm_params.nu = 0.1;
	svm_params.svm_type = cv::SVM::NU_SVR;
	svm_params.kernel_type = cv::SVM::RBF;
        
	output_path = output_dir/fs::path("sin_estimator.model");
	if (fs::exists(output_path)) {
		LOGI() << "Sine estimator already exists. Skipping.";	
		return 0;
	}
	LOGI() << "Training sine estimator" << std::endl;
	svm.train_auto(train_data, sin_labels, cv::Mat(), cv::Mat(), svm_params, 5);	
	svm.save(output_path.string().c_str());
	LOGI() << "Done" << std::endl;
        
	output_path = output_dir/fs::path("cos_estimator.model");
	if (fs::exists(output_path)) {
		LOGI() << "Cosine estimator already exists. Skipping.";	
		return 0;
	}
	LOGI() << "Training cosine estimator" << std::endl;
	svm.train_auto(train_data, cos_labels, cv::Mat(), cv::Mat(), svm_params, 5);				
	svm.save(output_path.string().c_str());        
	LOGI() << "Done" << std::endl;

	TestOrientationEstimator(test_data_dir_str, output_dir.string());

	return 0;
}

int TrainOrientationCosEstimator(std::string train_data_dir_str, std::string test_data_dir_str, std::string output_dir_str) {
	fs::path input_dir(train_data_dir_str), output_dir(output_dir_str), output_path;
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*dat");
	cv::Mat					train_data, sin_labels, cos_labels;
	cv::SVM					cos_svm, sin_svm; 
	cv::SVMParams			cos_svm_params, sin_svm_params;

	CHECK(fs::exists(input_dir));
	// if doesn't exist create a folder for the descriptors
	try {
		if (fs::is_directory(output_dir) == false)
			fs::create_directories(output_dir);
	} catch(boost::filesystem::filesystem_error e) { 
		LOGF() << "Failed to create dense trajectories descriptors folder: " << output_dir << std::endl;
		return -1;
	}

	for(fs::directory_iterator dir_iter(input_dir) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {
			fs::path input_path = dir_iter->path();
			std::string input_fname = fs::basename(input_path);
			float angle, sin_label, cos_label;
			cv::Mat  desc;
			MatReader	mr;
		
			LOGI() << "Extracting data from " << input_fname << std::endl;		
			angle = extract_angle(input_fname.c_str());
			if (((int)angle % 20) != 0)
				continue;
		
			angle += 90;
			sin_label = sin((3.14f/180)*angle);
			cos_label = cos((3.14f/180)*angle);
			CHECK(mr.Init(input_path.string()) == 0);

			while (mr.Read(&desc) == 0) {
				for (int desc_idx = 0; desc_idx < desc.rows; desc_idx++) {
					train_data.push_back(desc.row(desc_idx));
					sin_labels.push_back(sin_label);			
					cos_labels.push_back(cos_label);
				}
			}
		}
	}	
	
	
	cos_svm_params.nu = 0.1;
	cos_svm_params.svm_type = cv::SVM::NU_SVR;
	cos_svm_params.kernel_type = cv::SVM::RBF;
	LOGI() << "Training cosine estimator" << std::endl;
	cos_svm.train_auto(train_data, cos_labels, cv::Mat(), cv::Mat(), cos_svm_params, 5);			
	output_path = output_dir/fs::path("cos_estimator.model");
	cos_svm.save(output_path.string().c_str());        
	LOGI() << "Done" << std::endl;

	return 0;
}

int TrainOrientationSinEstimator(std::string train_data_dir_str, std::string test_data_dir_str, std::string output_dir_str) {
	fs::path input_dir(train_data_dir_str), output_dir(output_dir_str), output_path;
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*dat");
	cv::Mat					train_data, sin_labels, cos_labels;
	cv::SVM					cos_svm, sin_svm; 
	cv::SVMParams			cos_svm_params, sin_svm_params;

	CHECK(fs::exists(input_dir));
	// if doesn't exist create a folder for the descriptors
	try {
		if (fs::is_directory(output_dir) == false)
			fs::create_directories(output_dir);
	} catch(boost::filesystem::filesystem_error e) { 
		LOGF() << "Failed to create dense trajectories descriptors folder: " << output_dir << std::endl;
		return -1;
	}

	for(fs::directory_iterator dir_iter(input_dir) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {
			fs::path input_path = dir_iter->path();
			std::string input_fname = fs::basename(input_path);
			float angle, sin_label, cos_label;
			cv::Mat  desc;
			MatReader	mr;
		
			LOGI() << "Extracting data from " << input_fname << std::endl;		
			angle = extract_angle(input_fname.c_str());
			if (((int)angle % 20) != 0)
				continue;
		
			angle += 90;
			sin_label = sin((3.14f/180)*angle);
			cos_label = cos((3.14f/180)*angle);
			CHECK(mr.Init(input_path.string()) == 0);

			while (mr.Read(&desc) == 0) {
				for (int desc_idx = 0; desc_idx < desc.rows; desc_idx++) {
					train_data.push_back(desc.row(desc_idx));
					sin_labels.push_back(sin_label);			
					cos_labels.push_back(cos_label);
				}
			}
		}
	}	
	
	
	cos_svm_params.nu = 0.1;
	cos_svm_params.svm_type = cv::SVM::NU_SVR;
	cos_svm_params.kernel_type = cv::SVM::RBF;	

    sin_svm_params.nu = 0.1;
	sin_svm_params.svm_type = cv::SVM::NU_SVR;
	sin_svm_params.kernel_type = cv::SVM::RBF;	
	LOGI() << "Training sine estimator" << std::endl;
	sin_svm.train_auto(train_data, sin_labels, cv::Mat(), cv::Mat(), sin_svm_params, 5);
	output_path = output_dir/fs::path("sin_estimator.model");	
	sin_svm.save(output_path.string().c_str());
	LOGI() << "Done" << std::endl;

	return 0;
}

void TestOrientationEstimator(std::string test_dir_str, std::string estimators_dir_str) {	
	fs::path					test_dir(test_dir_str), estimators_dir(estimators_dir_str), estimator_path;
	fs::directory_iterator		eod;
	boost::smatch				what;	
	boost::regex				filter(".*dat");
	cv::Mat						test_data, sin_labels, cos_labels, angles;
	std::vector<std::string>	filenames;
	MatReader					mr;
	FILE						*res_file;
	cv::SVM						sin_estimator, cos_estimator; 
	
	LOGI() << "Testing " << estimators_dir.parent_path() << std::endl;
	res_file = fopen((fs::path(estimators_dir)/fs::path("results.txt")).string().c_str(), "w");
	CHECK(res_file != NULL);
	CHECK(fs::exists(test_dir));
	estimator_path = estimators_dir/fs::path("sin_estimator.model");
	CHECK(fs::exists(estimator_path));
	sin_estimator.load(estimator_path.string().c_str());
	estimator_path = estimators_dir/fs::path("cos_estimator.model");
	CHECK(fs::exists(estimator_path));
	cos_estimator.load(estimator_path.string().c_str());
	for(fs::directory_iterator dir_iter(test_dir) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {			
			float angle, sin_label, cos_label;
			cv::Mat  desc;
			fs::path input_path = dir_iter->path();
			std::string input_fname = fs::basename(input_path);
		
			angle = extract_angle(input_fname.c_str());
			if (((int)angle % 20) != 0)
				continue;

			angle += 90;
			sin_label = sin((3.14f/180)*angle);
			cos_label = cos((3.14f/180)*angle);
			CHECK(mr.Init(input_path.string()) == 0);
			while (mr.Read(&desc) == 0) {
				for (int row = 0; row < desc.rows; row++) {
					filenames.push_back(input_fname);
					test_data.push_back(desc.row(row));
					sin_labels.push_back(sin_label);
					cos_labels.push_back(cos_label);
					angles.push_back(angle);
				}
			}
		}
	}

	float mse = 0.0, sin_mse = 0.0, cos_mse = 0.0;
	for (int row = 0; row < test_data.rows; row++) { 
		float predicted_sin, predicted_cos, true_sin, true_cos, predicted_angle, true_angle;
		float err, sin_err, cos_err;

		true_sin = *(float*)(sin_labels.row(row).data);
		predicted_sin = std::max<float>(-1.0f, std::min<float>(1.0, sin_estimator.predict(test_data.row(row))));
		sin_err = true_sin - predicted_sin;
		sin_mse += sin_err*sin_err;

		true_cos = *(float*)(cos_labels.row(row).data);
		predicted_cos =  std::max<float>(-1.0f, std::min<float>(1.0, cos_estimator.predict(test_data.row(row))));
		cos_err = true_cos - predicted_cos;
		cos_mse += cos_err*cos_err;

		predicted_angle = cv::fastAtan2(predicted_sin, predicted_cos);
		true_angle = *(float*)(angles.row(row).data);
		err = AngleDif(true_angle, predicted_angle);				
		mse += err*err;
		//fprintf(res_file, "%s: angle = %3.1f, matched angles = %3.1f, err = %3.1f\n", filenames[row].c_str(), true_angle, predicted_angle, err);
	}		
			
	sin_mse /= test_data.rows;
	sin_mse = sqrt(sin_mse);			
	cos_mse /= test_data.rows;
	cos_mse = sqrt(cos_mse);			
	mse /= test_data.rows;
	mse = sqrt(mse);	
	LOGI() << "sine MSE: " << sin_mse << ", cosine MSE: " << cos_mse << ", MSE: " << mse << std::endl;
	fprintf(res_file, "sine MSE: %2.2f, cosine MSE: %2.2f, MSE: %f\n", sin_mse, cos_mse, mse);
	fclose(res_file);
}



