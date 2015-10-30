#include "frame_stitcher.h"


void decomposeAffine(const cv::Mat &H, double *x, double *y, double *theta, double *sigma_x, double *sigma_y) {
  cv::Mat R(2,2,CV_64FC1);
  // Translation
  *x = H.at<double>(0, 2);
  *y = H.at<double>(1, 2);  
  
  for (int i = 0; i < 2; i++)
	  for (int j = 0; j < 2; j++)
		R.at<double>(i, j) = H.at<double>(i, j);

  // SVD
  cv::Mat W, U, Vt;
  cv::SVD::compute(R, W, U, Vt);
  R = U * Vt;
 
  *theta = atan(R.at<double>(1, 0)/R.at<double>(0, 0));
  *sigma_x = U.at<double>(0, 0);
  *sigma_y = U.at<double>(1, 1);
}


FrameStitcher::FrameStitcher() {
	srand((int)time(NULL));		
	initialized_ = false;	
	acc_H_ = cv::Mat::eye(3,3,CV_64FC1);		
	acc_center_H_ = cv::Mat::eye(3,3,CV_64FC1);	
}

void FrameStitcher::WarpPoints(const cv::Mat& H, 
	const std::vector<cv::Point2f> &pts_in, 
	std::vector<cv::Point2f> *pts_out) {

	cv::Mat pts_in_mat(pts_in), pts_out_mat;

	transform(pts_in_mat, pts_out_mat, H);    
	*pts_out = pts_out_mat;
}

void FrameStitcher::WarpKPoints(const cv::Mat& H, 
		const std::vector<cv::KeyPoint> &kpts_in, 
        std::vector<cv::KeyPoint> *kpts_out) {

	std::vector<cv::Point2f> pts_in, pts_out;
	cv::KeyPoint::convert(kpts_in, pts_in);
	cv::KeyPoint::convert(*kpts_out,pts_out);

	WarpPoints(H, pts_in, &pts_out);

	cv::KeyPoint::convert(pts_out, *kpts_out);
}


int FrameStitcher::FindTransformRANSAC(const std::vector<cv::Point2f> &src_pts, 
		const std::vector<cv::Point2f> &dst_pts,
		int iteration_num,
        double ransac_reproj_th,		
		cv::Mat *M) {	

	std::vector<int> best_subset_size;
	cv::Mat tmp_M;		
	int match_candidates, max_support = 0;
	double min = 10000000.0;

	if (iteration_num == 0)
		iteration_num = 52;//(int)ceil(log(1-0.999)/log(1-pow(0.5, 3)));	
		
	if (src_pts.size() < 60) {		
        //LOGW() << "Less than 60 points!";		
		return 0;
	}
	
	match_candidates = src_pts.size();		
	for (int i=0; i < iteration_num; i++) {
		//generate random combination		
		int support = 0, cur_pt = 0, min_pts = 3;
		double err = 0.0;
		std::vector<int> idx(min_pts);
		std::vector<cv::Point2f> pts_in, pts_out, trans_pts;

		while (cur_pt < min_pts) {
			int k;
			idx[cur_pt] = rand() % match_candidates;	
			for (k = 0; k < cur_pt; k++)
				if (idx[cur_pt] == idx[k])
					break;

			if (k < cur_pt)
				continue;

			pts_in.push_back(src_pts[idx[cur_pt]]);
			pts_out.push_back(dst_pts[idx[cur_pt]]);
			cur_pt++;
		}		

		//build model
		//tmp_M = getAffineTransform(pts_in, pts_out);
		tmp_M = estimateRigidTransform(pts_in, pts_out, false);
		if (tmp_M.empty())
			continue;
		
		//validate model
		WarpPoints(tmp_M, src_pts, &trans_pts);		 
		for (int j = 0; j < match_candidates; j++) {
			float w = weights_.at<float>((int)dst_pts[j].y, (int)dst_pts[j].x);
			double res = w * norm(trans_pts[j] - dst_pts[j]);
			err += res;
			if (res < ransac_reproj_th) {				
				support++;							
			}
		}
			
		if ((err < min) && (support > 10)) {			
			tmp_M.copyTo(*M);
			max_support = support;						
			min = err;	
		}		

		if ((i == (iteration_num - 1)) && ((max_support < 50) || (max_support < 0.4*match_candidates)))
			iteration_num = std::min<int>((int)(iteration_num * 1.2), MAX_RANSAC_ITER);		
	}		
	

	cv::Mat r = cv::Mat::zeros(1,3,CV_64FC1);
	r.at<double>(0,2) = 1.0;
	(*M).push_back(r);	

	return max_support;
}

void FrameStitcher::MatchFrames(const cv::Mat &gray, 
        std::vector<cv::Point2f> *prv_matching_pts, 
        std::vector<cv::Point2f> *cur_matching_pts) {

	std::vector<cv::Point2f> pred_cur_pts(prv_pts_.size());
	cv::Mat matching_mask;
	cv::BFMatcher desc_matcher(cv::NORM_L2);
	std::vector<cv::DMatch> matches;
	std::vector<uchar> status;
	std::vector<float> err;

	// find position of feature in new image
	calcOpticalFlowPyrLK(
		prv_gray_, gray, // 2 consecutive images
		prv_pts_, // input point positions in first im
		pred_cur_pts, // output point positions in the 2nd
		status,    // tracking success
		err      // tracking error
		);

	for( int i = 0; i < (int)pred_cur_pts.size(); i++ ) {	
		if (status[i]) {
			(*prv_matching_pts).push_back(prv_pts_[i]);
			(*cur_matching_pts).push_back(pred_cur_pts[i]);									
		}
	}
}


void FrameStitcher::FindVisualMapDim(const cv::Mat &H, 
	const cv::Size &frame_size, 
	const cv::Size &panorama_size, 
	double *min_cols, 
	double *max_cols, 
	double *min_rows, 
	double *max_rows) {
	// Warp curent frame corners
	cv::Mat corners = cv::Mat::zeros(cv::Size(4,3), CV_64FC1), corners_trans;
	cv::Point min_loc; 
	cv::Point max_loc;

	corners.at<double>(1,1) = frame_size.height - 1;
	corners.at<double>(0,2) = frame_size.width - 1;
	corners.at<double>(0,3) = frame_size.width - 1;
	corners.at<double>(1,3) = frame_size.height - 1;
	corners.row(2) = cv::Mat::ones(cv::Size(4,1), CV_64FC1);
	corners_trans = H*corners;
	corners_trans.row(0) = corners_trans.row(0)/corners_trans.row(2);
	corners_trans.row(1) = corners_trans.row(1)/corners_trans.row(2);


	corners.at<double>(1,1) = panorama_size.height-1;
	corners.at<double>(0,2) = panorama_size.width-1;
	corners.at<double>(0,3) = panorama_size.width-1;
	corners.at<double>(1,3) = panorama_size.height-1;
	hconcat(corners_trans, corners, corners_trans);	

	minMaxLoc(corners_trans.row(1), min_rows, max_rows, &min_loc, &max_loc);
	minMaxLoc(corners_trans.row(0), min_cols, max_cols, &min_loc, &max_loc);

	*min_cols = floor(*min_cols);
	*min_rows = floor(*min_rows);
	*max_cols = ceil(*max_cols);
	*max_rows = ceil(*max_rows);
}

void FrameStitcher::Stitch(const cv::Mat &org_frame, 
	const cv::Mat &org_feature_mask, 
	cv::Mat *visual_map, 
	cv::Mat *framed_visual_map) {	

	std::vector<cv::Point2f> cur_pts, prv_matching_pts, cur_matching_pts;
	cv::Mat gray, frame, feature_mask, H, center_H, total_perspective_H, total_affine_H;
	cv::Mat new_pano, framed_warp, stitching_mask;
	cv::Size new_pano_size, frame_size(org_frame.cols/2, org_frame.rows/2);
	cv::Rect rect;
	int inliers_num, margin = 3;
	double min_cols, max_cols, min_rows, max_rows;

	// Resize and convert to grayscale
	resize(org_feature_mask, feature_mask, frame_size);
	resize(org_frame, frame, frame_size);	
	cvtColor( frame, gray, CV_RGB2GRAY );

	// Extract good tracking features for OF
	goodFeaturesToTrack(gray, cur_pts, GOOD_KP_NUM, 0.001, 10, feature_mask, 4, false, 0.04);

	if (!initialized_) {
		frame.copyTo((*visual_map));	
		frame.copyTo((*framed_visual_map));		
		initialized_ = true;
		// Store state
		gray.copyTo(prv_gray_);
		prv_pts_ = cur_pts;
		weights_ = cv::Mat(frame.rows, frame.cols, CV_32FC1);
		for (int r = 0; r < frame.rows; r++)
			for (int c = 0; c < frame.cols; c++)
				weights_.at<float>(r, c) = - 1 * ((float)abs((r - frame.rows/2) * (c - frame.cols/2))) / (frame.rows * frame.cols / 4);
		cv::exp(weights_, weights_);

		return;
	}
	
	// Match key points
	MatchFrames(gray,
		&prv_matching_pts,
		&cur_matching_pts);
			
	// Store state
	gray.copyTo(prv_gray_);
	prv_pts_ = cur_pts;
	
	// Find frame transformation
	inliers_num = FindTransformRANSAC(cur_matching_pts, prv_matching_pts, 0, 1, &H);	
	// Test for outliers	
	if (inliers_num < MIN_INLIERS) 	{			
		//H = cv::Mat::eye(3,3,CV_64FC1);
		printf("dropping frame\n");
		return;
	}
	
	// Update transformation matrix
	acc_H_ = acc_H_*H;

	double x, y, theta, sigma_x, sigma_y;
	decomposeAffine(acc_H_, &x, &y, &theta, &sigma_x, &sigma_y);
	//printf("x=%04.4f, y=%04.4f, theta=%02.4f, sigma_x=%2.4f, sigma_y=%2.4f\n",x,y,theta * (180/3.14), sigma_x, sigma_y);
	acc_H_.at<double>(0, 0) = cos(theta);
	acc_H_.at<double>(0, 1) = -sin(theta);
	acc_H_.at<double>(1, 0) = sin(theta);
	acc_H_.at<double>(1, 1) = cos(theta);


	// Find new visual map dimensions	
	FindVisualMapDim(acc_center_H_*acc_H_, 
		gray.size(), 
		visual_map->size(), 
		&min_cols, &max_cols, &min_rows, &max_rows);

	// Shift origin to fit the whole curent image in the visual map
	center_H = cv::Mat::eye(3,3,CV_64FC1);
	center_H.at<double>(0,2) = MAX(0,-min_cols);
	center_H.at<double>(1,2) = MAX(0,-min_rows);			
	acc_center_H_ = center_H * acc_center_H_;
	total_perspective_H = acc_center_H_*acc_H_;	

	total_affine_H.push_back(total_perspective_H.row(0));
	total_affine_H.push_back(total_perspective_H.row(1));

	new_pano_size = cv::Size((int)(ceil(max_cols) - floor(min_cols)) + 1, 
		(int)(ceil(max_rows) - floor(min_rows)) + 1);		
	framed_warp = 255 * cv::Mat::ones(frame.size(), CV_8UC3);
	rect = cv::Rect(margin, margin, framed_warp.cols - 2 * margin, framed_warp.rows - 2 * margin);
	frame(rect).copyTo(framed_warp(rect));		
	stitching_mask = 255 * cv::Mat::ones(gray.size(), CV_8UC1);	

	// Warp current frame
	warpAffine(frame, new_pano, total_affine_H, new_pano_size);
	// Warped framed image
	warpAffine(framed_warp, (*framed_visual_map), total_affine_H, new_pano_size);
	// Calculate stitching mask
	warpAffine(stitching_mask, stitching_mask, total_affine_H, new_pano_size);

	stitching_mask = 255 - stitching_mask;
	// Copy previous visual map to new warped frame
	rect = cv::Rect((int)std::max<double>(0, -min_cols), (int)std::max<double>(0, -min_rows), visual_map->cols, visual_map->rows);	
	
	visual_map->copyTo(new_pano(rect), stitching_mask(rect));
	visual_map->copyTo((*framed_visual_map)(rect), stitching_mask(rect));

	*visual_map = new_pano;
}
