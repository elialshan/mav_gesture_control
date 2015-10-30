#include "tracker.h"
#include "utils.h"
#include "hog.h"

#include "foreground_detector.h"
#include "frame_stitcher.h"
#include "operators.h"

#include <string>
#define HOG_NX_BLOCKS	(5)
#define HOG_NY_BLOCKS	(5)
#define HOG_NBINS		(16)


int EvaluateDetection(const char *input_video_path, const char* models_path, const char *gt_path, const char *output_path) {    
    cv::Mat         cur_frame;

    //Detection
    cvb::CvTracks   target_tracks;		
    CvSVM           operator_classifier;	
    Operators       operators;
    int             frame_num = 0, op_active_th = 5, tar_active_th = 5;

    
    fs::path        op_class_path((fs::path(models_path)/fs::path("operator_detector")/fs::path("operator_classifier.model")).string().c_str());
    fs::path        video_path(input_video_path);
    std::string     video_name = fs::basename(video_path);
    fs::path        op_gt_path((fs::path(gt_path)/fs::path(video_name  + std::string(".op.gt.csv"))));
    fs::path        tar_gt_path((fs::path(gt_path)/fs::path(video_name  + std::string(".tar.gt.csv"))));
    
    VideoHandle     video_handle;
  
    CHECK(fs::exists(op_class_path));
    CHECK(fs::exists(video_path));
    CHECK(fs::exists(op_gt_path));
    CHECK(fs::exists(tar_gt_path));
    CHECK(video_handle.Init(video_path.string().c_str(), 0.0f, 0) == 0);
    
    operator_classifier.load(op_class_path.string().c_str());	

    LOGI() << "Evaluation detection in " << video_name << std::endl;
    ForegroundDetector	fd(video_handle.get_frame_height(), video_handle.get_frame_widht(), 32, 32, 0.5f);
    
    std::ifstream op_track_strm(op_gt_path.string().c_str());	
    int op_tp = 0, op_fp = 0, op_fn = 0;
    std::ifstream tar_track_strm(tar_gt_path.string().c_str());	
    int tar_tp = 0, tar_fp = 0, tar_fn = 0;
    FILE* results_fd = fopen((fs::path(output_path)/fs::path(video_name + std::string("_track_res.txt"))).string().c_str(), "w");	

    
    while (video_handle.GetFrame(&cur_frame) == 0) {		
        cv::Mat                     marked_frame, operator_mask, target_mask, object_mask;
        std::vector<OperatorROI>    op_roi;		
        cvb::CvBlobs                blobs;

        cur_frame.copyTo(marked_frame);
        MarkTarget(fd, cur_frame, &target_tracks);

        GetObjectsMask(cur_frame.size(), *operators.get_tracks(), &object_mask);
        //calculate operator mask
        fd.GetOperatorMask(cur_frame, object_mask, &operator_mask);	
		
        //detect and track operators
        TrackOperators(operator_mask, cur_frame, operator_classifier, operators.get_tracks());
        operators.UpdateOperators();
        cvb::CvTracks tracks = *operators.get_tracks();	
        
        EvaluateTracker(tracks, cur_frame, op_active_th, &op_track_strm,
                CV_RGB(0, 255, 255), OP_RECT_SIZE, &marked_frame, 
                &op_tp, &op_fp, &op_fn);

        RenderTracks(target_tracks, tar_active_th, CV_RGB(255.,0,0), TAR_RECT_SIZE, &marked_frame);
        EvaluateTracker(target_tracks, cur_frame, 
                tar_active_th,	&tar_track_strm,
                CV_RGB(255, 255, 0), TAR_RECT_SIZE, 
                &marked_frame, &tar_tp, &tar_fp, &tar_fn);
        
	int total_op = op_tp + op_fp + op_fn;
        int total_tar = tar_tp + tar_fp + tar_fn;
        float op_accuracy, op_recall, op_precision, tar_accuracy, tar_recall, tar_precision;
        op_accuracy = 1.0f - ((float)(op_fp + op_fn))/total_op;
        op_recall = ((float)op_tp)/(op_tp + op_fn);
        op_precision = ((float)op_tp)/(op_tp + op_fp);
        tar_accuracy = 1.0f - ((float)(tar_fp + tar_fn))/total_tar;
        tar_recall = ((float)tar_tp)/(tar_tp + tar_fn);
        tar_precision = ((float)tar_tp)/(tar_tp + tar_fp);
	printf("%d Operator:\t Accuracy=%2.4f, Recall=%2.4f, Precision=%2.4f\n", frame_num, op_accuracy, op_recall, op_precision);
        printf("%d Target:\t Accuracy=%2.4f, Recall=%2.4f, Precision=%2.4f\n", frame_num, tar_accuracy, tar_recall, tar_precision);	

        frame_num++;
    }

    int total_op = op_tp + op_fp + op_fn;
    printf("===========================\n");
    fprintf(results_fd, "===========================\n");
    printf("Operator:\t TP=%d, FP=%d, FN=%d\n", op_tp, op_fp, op_fn);
    fprintf(results_fd, "Operator:\t TP=%d, FP=%d, FN=%d\n", op_tp, op_fp, op_fn);
    printf("Operator:\t Accuracy=%2.4f, Recall=%2.4f, Precision=%2.4f\n", 1.0 - ((float)(op_fp + op_fn))/total_op, ((float)op_tp)/(op_tp + op_fn), ((float)op_tp)/(op_tp + op_fp));
    fprintf(results_fd, "Operator:\t Accuracy=%2.4f, Recall=%2.4f, Precision=%2.4f\n", 1.0 - ((float)(op_fp + op_fn))/total_op, ((float)op_tp)/(op_tp + op_fn), ((float)op_tp)/(op_tp + op_fp));
    int total_tar = tar_tp + tar_fp + tar_fn;
    printf("Target:\t TP=%d, FP=%d, FN=%d\n", tar_tp, tar_fp, tar_fn);
    fprintf(results_fd, "Target:\t TP=%d, FP=%d, FN=%d\n", tar_tp, tar_fp, tar_fn);
    printf("Target:\t Accuracy=%2.4f, Recall=%2.4f, Precision=%2.4f\n", 1.0 - ((float)(tar_fp + tar_fn))/total_tar, ((float)tar_tp)/(tar_tp + tar_fn), ((float)tar_tp)/(tar_tp + tar_fp));
    fprintf(results_fd, "Target:\t Accuracy=%2.4f, Recall=%2.4f, Precision=%2.4f\n", 1.0 - ((float)(tar_fp + tar_fn))/total_tar, ((float)tar_tp)/(tar_tp + tar_fn), ((float)tar_tp)/(tar_tp + tar_fp));
    fclose(results_fd);

    return 0;
}


int TrainOperatorClassifier(const char *data_path_char, const char *classifier_path_char) {	
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*jpg");
	fs::path				data_path, classifier_dir, classifier_path;
	cv::Mat					data, labels;
	CvSVM					SVM; 
	CvSVMParams				svm_params;

	if (fs::exists(fs::path(classifier_path_char))) {
		LOGI() << "Operator classifier already exists. Skipping" << std::endl;
		return 0;
	}
	LOGI() << "Training operator classifier" << std::endl;
	classifier_path = fs::path(classifier_path_char);
	classifier_dir = classifier_path.parent_path();
	data_path = fs::path(data_path_char)/"pos";
	CHECK(fs::exists(data_path));	
	
	try {
		if (fs::is_directory(classifier_dir) == false)
			fs::create_directories(classifier_dir);
	} catch(boost::filesystem::filesystem_error e) { 
		LOGF() << "Failed to create operator classifier folder: " << classifier_dir << std::endl;
		return -1;
	}
	LOGD() << "Collecting training data" << std::endl;
	// collect all action descriptors	
	for(fs::directory_iterator dir_iter(data_path) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {			
			std::string filename;
			cv::Mat		desc_mat, img;						
			float		label = 1.0f;

			filename = fs::basename(dir_iter->path());
			img = cv::imread(dir_iter->path().string(), CV_LOAD_IMAGE_COLOR); 			
			ExtractHog(img, HOG_NX_BLOCKS, HOG_NY_BLOCKS, HOG_NBINS, &desc_mat);
			data.push_back(desc_mat);

			labels.push_back(label);
			LOGD() << data.rows << " objects collected" << std::endl;
		}
	}

	data_path = fs::path(data_path_char)/"neg";
	CHECK(fs::exists(data_path));		
	// collect all descriptors	
	for(fs::directory_iterator dir_iter(data_path) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {			
			std::string filename;
			cv::Mat		desc_mat, img;						
			float		label = -1.0f;

			filename = fs::basename(dir_iter->path());
			img = cv::imread(dir_iter->path().string(), CV_LOAD_IMAGE_COLOR); 
			ExtractHog(img, HOG_NX_BLOCKS, HOG_NY_BLOCKS, HOG_NBINS, &desc_mat);
			data.push_back(desc_mat);

			labels.push_back(label);
			LOGD() << data.rows << " objects collected" << std::endl;
		}
	}

	//SVM	
	svm_params.svm_type = cv::SVM::C_SVC;	
	svm_params.kernel_type = cv::SVM::LINEAR;		
		
	LOGD() << "Training ..." << std::endl;
	SVM.train_auto(data, labels, cv::Mat(), cv::Mat(), svm_params, 10);			
	LOGD() << "Training complete. Saving model to " << classifier_path.string().c_str() << std::endl;			
	SVM.save(classifier_path.string().c_str());	

	return 0;
}

int TestOperatorClassifier(const char *data_path_char, const char *classifier_path_char) {	
	fs::directory_iterator	eod;
	boost::smatch			what;	
	boost::regex			filter(".*jpg");
	fs::path				data_path, classifier_path, results_path;
	cv::Mat					data, labels;
	CvSVM					SVM; 
	float					sensitivity, specificity, precision, accuracy;
	int						confusion_matrix[2][2] = {0};
	char					line[256];
	FILE					*results_fd;
	
	classifier_path = fs::path(classifier_path_char);
	CHECK(fs::exists(classifier_path));

	LOGD() << "Loading model from " << classifier_path.string().c_str() << std::endl;
	SVM.load(classifier_path.string().c_str());


	LOGD() << "Collecting test data" << std::endl;
	data_path = fs::path(data_path_char)/"pos";
	CHECK(fs::exists(data_path));

	results_path = fs::path(classifier_path_char).parent_path()/"results.txt";
	// collect all descriptors	
	for(fs::directory_iterator dir_iter(data_path) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {			
			std::string filename;
			cv::Mat				desc_mat, img;			
			std::vector<float>	desc;
			float				label;

			filename = fs::basename(dir_iter->path());
			label = 1.0f;

			img = cv::imread(dir_iter->path().string(), CV_LOAD_IMAGE_COLOR); 
			ExtractHog(img, HOG_NX_BLOCKS, HOG_NY_BLOCKS, HOG_NBINS, &desc_mat);
			data.push_back(desc_mat);

			labels.push_back(label);
			LOGD() << data.rows << " objects collected" << std::endl;
		}
	}

	data_path = fs::path(data_path_char)/"neg";
	CHECK(fs::exists(data_path));
	// collect all action descriptors	
	for(fs::directory_iterator dir_iter(data_path) ; (dir_iter != eod); ++dir_iter) {			
		if ((fs::is_regular_file(*dir_iter)) && (boost::regex_match(dir_iter->path().string(), what, filter))) {			
			std::string filename;
			cv::Mat				desc_mat, img;			
			std::vector<float>	desc;
			float				label;

			filename = fs::basename(dir_iter->path());
			label = -1.0f;

			img = cv::imread(dir_iter->path().string(), CV_LOAD_IMAGE_COLOR); 
			ExtractHog(img, HOG_NX_BLOCKS, HOG_NY_BLOCKS, HOG_NBINS, &desc_mat);
			data.push_back(desc_mat);

			labels.push_back(label);
			LOGD() << data.rows << " objects collected" << std::endl;
		}
	}

	LOGD() << "Testing ..." << std::endl;
	for (int desc_idx = 0; desc_idx < data.rows; desc_idx++) {
		float label;
		int true_idx, predicted_idx;

		true_idx = (labels.at<float>(desc_idx) == 1.0f)? 0: 1;
		label = SVM.predict(data.row(desc_idx));								
		predicted_idx = (label == 1.0f) ? 0 : 1;
		confusion_matrix[predicted_idx][true_idx]++;		
		LOGD() << (desc_idx + 1) << "/" << data.rows << " complete" << std::endl;
	}	
	
	results_fd = fopen(results_path.string().c_str(), "w");
	LOGD() << "========================================" << std::endl;
	LOGD() << "Confusion matrix" << std::endl;
	fprintf(results_fd,"========================================\n");
	fprintf(results_fd, "Confusion matrix\n");
	for (int row=0; row<2; row++)
	{
		for (int col=0; col<2; col++) {
			printf("%d\t\t",confusion_matrix[row][col]);
			fprintf(results_fd, "%d\t\t",confusion_matrix[row][col]);
		}
		printf("\n");
		fprintf(results_fd, "\n");
	}
	
	sensitivity = (float)confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0]);
	specificity = (float)confusion_matrix[1][1]/(confusion_matrix[0][1] + confusion_matrix[1][1]);
	precision = (float)confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1]);
	accuracy = (float)(confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1]);
	sprintf(line, "Recall: %2.4f, Specificity: %2.4f, Precision: %2.4f, Accuracy: %2.4f\n\n\n",sensitivity, specificity, precision, accuracy);	
	LOGD() << std::string(line);
	fprintf(results_fd, "%s", line);

	return 0;
}


float ClassifyObject(const cv::Mat& img, 
	const CvSVM &operator_classifier)
{	
	cv::Mat desc_mat;
	ExtractHog(img, HOG_NX_BLOCKS, HOG_NY_BLOCKS, HOG_NBINS, &desc_mat);
	return operator_classifier.predict(desc_mat);
}


void FilterObjects(const cv::Mat &img, 
	const CvSVM &operator_classifier, 
	cvb::CvBlobs *blobs) {

    cvb::CvBlobs::iterator it=blobs->begin();
    while(it!=blobs->end())    {
		cv::Mat			object, temp;		
		cvb::CvBlob*	blob = (*it).second;
		float			label;
		cv::Rect		r1, r2;
		cv::Point		p1, p2;
		cv::Mat			tmp;				

		object = cv::Mat::zeros(OP_RECT_SIZE, OP_RECT_SIZE, img.type());
		p1 = cv::Point(std::max<int>(0, (int)blob->centroid.x - OP_RECT_SIZE/2), 
			std::max<int>(0, (int)blob->centroid.y - OP_RECT_SIZE/2));
		p2 = cv::Point(std::min<int>(img.cols, (int)blob->centroid.x + OP_RECT_SIZE/2), 
			std::min<int>(img.rows, (int)blob->centroid.y + OP_RECT_SIZE/2)); 
		r1 = cv::Rect(p1, p2);			
		tmp = img(r1);
		r2 = cv::Rect((OP_RECT_SIZE - tmp.cols)/2, (OP_RECT_SIZE - tmp.rows)/2, tmp.cols, tmp.rows);
		tmp.copyTo(object(r2));
		
		ExtractHog(object, HOG_NX_BLOCKS, HOG_NY_BLOCKS, HOG_NBINS, &blob->desc);
		label = operator_classifier.predict(blob->desc);

		if (label != 1)
		{
			cvb::cvReleaseBlob(blob);
			cvb::CvBlobs::iterator tmp=it;
			++it;
			blobs->erase(tmp);
		}
		else
		{			
			++it;				
		}
    }
  }
/*************************************************/

void FilterByAspectRatio(double min_ratio, 
	double max_ratio, 
	cvb::CvBlobs *blobs) {

	cvb::CvBlobs::iterator it=blobs->begin();
    while(it!=blobs->end()) {
        cvb::CvBlob *blob=(*it).second;
        double ratio = ((double)(blob->maxx - blob->minx))/(blob->maxy - blob->miny);

        if ((ratio < min_ratio) || (ratio > max_ratio)) {
            cvReleaseBlob(blob);
            cvb::CvBlobs::iterator tmp=it;
            ++it;
            blobs->erase(tmp);
        }
        else
            ++it;
    }
}

void TrackOperators(const cv::Mat &foreground_mat, 
	const cv::Mat &frame_mat, 
	const CvSVM &operator_classifier, 
	cvb::CvTracks *tracks) {	

	IplImage *fore = new IplImage(foreground_mat);
	IplImage *labelImg = cvCreateImage(cvGetSize(fore), IPL_DEPTH_LABEL, 1);
	cvb::CvBlobs blobs;

	//detect blobs
	cvLabel(fore, labelImg, blobs);

	cvFilterByArea(blobs, 500, 10000);
	FilterByAspectRatio(0.333, 3, &blobs);
	FilterObjects(frame_mat, operator_classifier, &blobs);
	
	cvUpdateTracks(blobs, (*tracks), 10., 15, 10);	
	
	cvReleaseBlobs(blobs);
	cvReleaseImage(&labelImg);
	delete fore;
}


void TrackTarget(const cv::Mat &foreground_mat, 
	const cv::Mat &frame_mat, 
	cvb::CvTracks *tracks) {

	IplImage *fore = new IplImage(foreground_mat);
	IplImage *labelImg = cvCreateImage(cvGetSize(fore), IPL_DEPTH_LABEL, 1);
	cvb::CvBlobs blobs;

	//detect blobs
	cvLabel(fore, labelImg, blobs);

	cvFilterByArea(blobs, 40, 500);
	FilterByAspectRatio(0.5, 2, &blobs);
	cvUpdateTracks(blobs, (*tracks), 10., 5, 10);	
	
	cvReleaseBlobs(blobs);
	cvReleaseImage(&labelImg);
	delete fore;
}

void MarkTarget(ForegroundDetector &fd, 
	const cv::Mat &cur_frame, 
	cvb::CvTracks *tracks) {

    cv::Mat target_mask;
    
    fd.GetTargetMask(cur_frame, &target_mask);
    TrackTarget(target_mask, cur_frame, tracks);   
}

void RenderTracks(const cvb::CvTracks tracks, 
	unsigned int active_time_th, 
	cv::Scalar bbox_color,
	int bbox_size,
	cv::Mat *img) {	

	cv::Scalar font_color = bbox_color;
	int bbox_width = 1, margin = bbox_size/4;
	bool show_counter = false;

	if (img->channels() != 3) {
		bbox_color = CV_RGB(255.,255.,255.);
		if (show_counter)
			font_color = CV_RGB(0.,0.,0.);
	}

	//for each track draw a bounding box and track id
	for (cvb::CvTracks::const_iterator it=tracks.begin(); it!=tracks.end(); ++it) {
		cvb::CvTrack *track = (*it).second;	
		int x = (int)track->centroid.x, y = (int)track->centroid.y;

		if ((track->lifetime > active_time_th) && (x > margin) && (x < img->cols - margin) && (y > margin) && (y < img->rows - margin)) {
			cv::Point p1(std::max<int>(0, x - bbox_size/2), std::max<int>(0, y - bbox_size/2)); 
			cv::Point p2(std::min<int>(img->cols, x + bbox_size/2), std::min<int>(img->rows, y + bbox_size/2));
			
			cv::rectangle(*img, p1, p2, bbox_color, bbox_width);			
			std::stringstream buffer;
			buffer << it->first;
			if (show_counter)
                            cv::putText(*img, buffer.str().c_str(), cv::Point(x, y), NULL, 0.9, font_color, bbox_width);			
		}
	}	
}

void GetObjectsMask(const cv::Size mask_size, 
	const cvb::CvTracks tracks,
	cv::Mat *mask) {
	cv::Scalar bbox_color(255.,255.,255.);

	*mask = cv::Mat::zeros(mask_size, CV_8UC1);
	for (cvb::CvTracks::const_iterator it=tracks.begin(); it!=tracks.end(); ++it) {
		cvb::CvTrack *track = (*it).second;	

		cv::Point p1(std::max<int>(0, (int)track->centroid.x - OP_RECT_SIZE/2), std::max<int>(0, (int)track->centroid.y - OP_RECT_SIZE/2)); 
		cv::Point p2(std::min<int>(mask_size.width, (int)track->centroid.x + OP_RECT_SIZE/2), std::min<int>(mask_size.height, (int)track->centroid.y + OP_RECT_SIZE/2));
			
		cv::rectangle(*mask, p1, p2, bbox_color, -1);		
	}	
}

void EvaluateTracker(const cvb::CvTracks tracks, 
	const cv::Mat &frame,
	unsigned int active_time_th, 
	std::ifstream *track_strm,
	cv::Scalar bbox_color,
	int bbox_size,	
	cv::Mat *marked_frame,
	int *tp, int *fp, int *fn) {

	std::string line, token;
	std::getline(*track_strm, line);	
	int end, rows, cols, margin = bbox_size/4;
	std::vector<CvPoint2D64f> fp_centroids, fn_centroids;	
			
	rows = marked_frame->rows;
	cols = marked_frame->cols;
	for (cvb::CvTracks::const_iterator it=tracks.begin(); it!=tracks.end(); ++it) {
		cvb::CvTrack *track = (*it).second;	
		if ((track->lifetime > active_time_th)) {
			fp_centroids.push_back(track->centroid);
		}
	}

	end = line.find("\t");
	line = line.substr(end+1, line.length()-1);
	while ((end = line.find("\t")) != std::string::npos) {		
		int bb[4], num_end=0;
		std::string num_str;
		long double best_score = 0.0;
		int best_match_idx = 0, idx = 0;
		float gt_bb_area;

		token = line.substr(0, end);
		for (int i=0; i<4; i++) {
			num_end = token.find(",");
			num_str = token.substr(0, num_end);
			bb[i] = atoi(num_str.c_str());
			token = token.substr(num_end+1, token.length()-1);
		}
		cv::Point gt_p1(bb[0], bb[1]); 
		cv::Point gt_p2(bb[2], bb[3]);
		CvPoint2D64f gt_centroid;

		if ((gt_p1.x > 0) && (gt_p1.y >0)) {
			gt_centroid.x = std::min<int>(cols, gt_p1.x + bbox_size/2);
			gt_centroid.y = std::min<int>(rows, gt_p1.y + bbox_size/2);
		}
		else {
			gt_centroid.x = std::max<int>(0, gt_p2.x - bbox_size/2);
			gt_centroid.y = std::max<int>(0, gt_p2.y - bbox_size/2);
		}			
		
		line = line.substr(end+1, line.length()-1);

		gt_bb_area = (float)((bb[2] - bb[0]) * (bb[3] - bb[1]));
		cv::rectangle(*marked_frame, gt_p1, gt_p2, bbox_color, 1);				

		for (std::vector<CvPoint2D64f>::iterator it=fp_centroids.begin(); it!=fp_centroids.end(); ++it) {
			int x = (int)it->x, y = (int)it->y;
			cv::Point p1(std::max<int>(0, x - bbox_size/2), std::max<int>(0, y - bbox_size/2)); 
			cv::Point p2(std::min<int>(cols, x + bbox_size/2), std::min<int>(rows, y + bbox_size/2));
                        
			float bb_area = (float)((p2.x - p1.x) * (p2.y - p1.y));
			cv::Point p1_intersection(std::max<int>(gt_p1.x, p1.x), std::max<int>(gt_p1.y, p1.y));
			cv::Point p2_intersection(std::min<int>(gt_p2.x, p2.x), std::min<int>(gt_p2.y, p2.y));

			float i_area = ((p2_intersection.x > p1_intersection.x) && (p2_intersection.y > p1_intersection.y))? 
				(float)((p2_intersection.x - p1_intersection.x) * (p2_intersection.y - p1_intersection.y)):0;
			float u_area = gt_bb_area + bb_area - i_area;
			float IoU = i_area/u_area;
			if (best_score < IoU) {
				best_score = IoU;
				best_match_idx = idx;
			}
			idx ++;
		}

		if ((best_score < DET_TH) || (fp_centroids.size() == 0)) {
                    int x = (int)gt_centroid.x, y = (int)gt_centroid.y;
                    if((x > margin) && (x < marked_frame->cols - margin) && (y > margin) && (y < marked_frame->rows - margin)) {
                        (*fn)++;
                        fn_centroids.push_back(gt_centroid);
                    }
		}
		else {			
                    (*tp)++;
                    fp_centroids.erase(fp_centroids.begin() + best_match_idx);			
		}
	}

	
	for (std::vector<CvPoint2D64f>::iterator it=fp_centroids.begin(); it!=fp_centroids.end(); ++it) {
		int x = (int)it->x, y = (int)it->y;
		if ((x > margin) && (x < marked_frame->cols - margin) && (y > margin) && (y < marked_frame->rows - margin)) {		
                    (*fp) ++;
		}	
	}
}
