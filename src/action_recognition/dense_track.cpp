#include "dense_track.h"
int DenseTrack::unique_id_ = 0;

DenseTrack::DenseTrack(){
	id_ = DenseTrack::unique_id_++;
	frames_cnt_ = 0;
	InitTrackInfo(&track_info_, g_track_length, g_init_gap);
	InitDescInfo(&hog_info_, 8, false, g_patch_size, g_nxy_cell, g_nt_cell);
	InitDescInfo(&hof_info_, 9, true, g_patch_size, g_nxy_cell, g_nt_cell);
	InitDescInfo(&mbh_info_, 8, false, g_patch_size, g_nxy_cell, g_nt_cell);

	fscales_ = std::vector<float>(0);
	sizes_ = std::vector<cv::Size>(0);

	prev_grey_pyr_ = std::vector<cv::Mat>(0);
	grey_pyr_ = std::vector<cv::Mat>(0);
	flow_pyr_ = std::vector<cv::Mat>(0);
	flow_warp_pyr_ = std::vector<cv::Mat>(0);

	prev_poly_pyr_ = std::vector<cv::Mat>(0);
	poly_pyr_ = std::vector<cv::Mat>(0);
	poly_warp_pyr_ = std::vector<cv::Mat>(0);

	initialized_ = false;
}

DenseTrack::~DenseTrack() {
    if (initialized_) {
        for(int iScale = 0; iScale < g_scale_num; iScale++) {
            ReleDescMat(&hog_mat_[iScale]);
            ReleDescMat(&hof_mat_[iScale]);
            ReleDescMat(&mbh_mat_x_[iScale]);
            ReleDescMat(&mbh_mat_y_[iScale]);
        }
    }
    	
}

void DenseTrack::Init(const cv::Mat &frame) {
	grey_.create(frame.size(), CV_8UC1);
	prev_grey_.create(frame.size(), CV_8UC1);

	InitPry(frame, fscales_, sizes_);

	BuildPry(sizes_, CV_8UC1, prev_grey_pyr_);
	BuildPry(sizes_, CV_8UC1, grey_pyr_);
	BuildPry(sizes_, CV_32FC2, flow_pyr_);
	BuildPry(sizes_, CV_32FC2, flow_warp_pyr_);
	BuildPry(sizes_, CV_32FC(5), prev_poly_pyr_);
	BuildPry(sizes_, CV_32FC(5), poly_pyr_);
	BuildPry(sizes_, CV_32FC(5), poly_warp_pyr_);

	xy_scale_tracks_.resize(g_scale_num);	
	cvtColor(frame, prev_grey_, CV_BGR2GRAY);

        hog_mat_ = std::vector<DescMat>(g_scale_num);
        hof_mat_ = std::vector<DescMat>(g_scale_num);
        mbh_mat_x_ = std::vector<DescMat>(g_scale_num);
        mbh_mat_y_ = std::vector<DescMat>(g_scale_num);
	for(int iScale = 0; iScale < g_scale_num; iScale++) {
            int width, height;
            
		if(iScale == 0)
			prev_grey_.copyTo(prev_grey_pyr_[0]);
		else
			resize(prev_grey_pyr_[iScale-1], prev_grey_pyr_[iScale], prev_grey_pyr_[iScale].size(), 0, 0, cv::INTER_LINEAR);

		// dense sampling descriptor points
		std::vector<cv::Point2f> points(0);
		DenseSample(prev_grey_pyr_[iScale], points, g_quality, g_min_distance);

		// save the descriptor points
		std::list<Track>& tracks = xy_scale_tracks_[iScale];
		for(int i = 0; i < (int)points.size(); i++)
			tracks.push_back(Track(points[i], track_info_, hog_info_, hof_info_, mbh_info_));
                
                width = grey_pyr_[iScale].cols;
                height = grey_pyr_[iScale].rows;

                // initialize the integral histograms
                InitDescMat(height+1, width+1, hog_info_.nBins, &hog_mat_[iScale]);
                InitDescMat(height+1, width+1, hof_info_.nBins, &hof_mat_[iScale]);
                InitDescMat(height+1, width+1, mbh_info_.nBins, &mbh_mat_x_[iScale]);
                InitDescMat(height+1, width+1, mbh_info_.nBins, &mbh_mat_y_[iScale]);
	}

	// compute polynomial expansion
	my::FarnebackPolyExpPyr(prev_grey_, prev_poly_pyr_, fscales_, 7, 1.5);

	initialized_ = true;
}



int DenseTrack::Extract(const cv::Mat &frame, cv::Mat descriptor_set[], cv::Mat *tracks_image) {
	cv::Mat H, H_inv, grey_warp;
	int desc_cnt = 0;
	bool show_track = tracks_image != NULL;

	if (initialized_ == false) {
		Init(frame);
		return 0;
	}

	cvtColor(frame, grey_, CV_BGR2GRAY);
	// prepare tracks image
	if (show_track)
		frame.copyTo((*tracks_image));
        
        for(int iScale = 0; iScale < g_scale_num; iScale++) {

            if(iScale == 0) {
                grey_.copyTo(grey_pyr_[0]);
            }
            else {
                resize(grey_pyr_[iScale-1], grey_pyr_[iScale], grey_pyr_[iScale].size(), 0, 0, cv::INTER_LINEAR);                    
            }
        }
	// compute optical flow for all scales once
#ifndef CUDA       
	my::FarnebackPolyExpPyr(grey_, poly_pyr_, fscales_, 7, 1.5);
	my::calcOpticalFlowFarneback(prev_poly_pyr_, poly_pyr_, flow_pyr_, 10, 2);       
#else   
        for(int iScale = 0; iScale < g_scale_num; iScale++) {          
            cv::gpu::FarnebackOpticalFlow farn;
            cv::gpu::GpuMat d_frame0(prev_grey_pyr_[iScale]);
            cv::gpu::GpuMat d_frame1(grey_pyr_[iScale]);
            cv::gpu::GpuMat d_flowx(grey_pyr_[iScale].size(), CV_32FC1);
            cv::gpu::GpuMat d_flowy(grey_pyr_[iScale].size(), CV_32FC1);
            std::vector<cv::Mat> flow;

            farn(d_frame0, d_frame1, d_flowx, d_flowy);
            flow.push_back(cv::Mat(d_flowx));
            flow.push_back(cv::Mat(d_flowy));
            cv::merge(flow, flow_pyr_[iScale]);      
            }
#endif
	MatchFromFlow(prev_grey_, flow_pyr_[0], prev_pts_flow_, pts_flow_, cv::Mat::ones(frame.size(), CV_8UC1));	
	MergeMatch(prev_pts_flow_, pts_flow_, prev_pts_all_, pts_all_);

	//H = cv::Mat::eye(3, 3, CV_64FC1);
    grey_.copyTo(grey_warp);
	if(pts_all_.size() > 50) {
		std::vector<unsigned char> match_mask;
		cv::Mat temp = findHomography(prev_pts_all_, pts_all_, cv::RANSAC, 1, match_mask);
		if(countNonZero(cv::Mat(match_mask)) > 25) {
			H = temp;
                        H_inv = H.inv();
                        grey_warp = cv::Mat::zeros(grey_.size(), CV_8UC1);
                        MyWarpPerspective(prev_grey_, grey_, grey_warp, H_inv); // warp the second frame
                }
	}

	

#ifndef CUDA 
	// compute optical flow for all scales once
	my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr_, fscales_, 7, 1.5);
	my::calcOpticalFlowFarneback(prev_poly_pyr_, poly_warp_pyr_, flow_warp_pyr_, 10, 2);
#endif
	for(int iScale = 0; iScale < g_scale_num; iScale++) {
            int width, height;
            std::list<Track>& tracks = xy_scale_tracks_[iScale];                
               
            if(iScale != 0)
                cv::resize(grey_warp, grey_warp, grey_pyr_[iScale].size(), 0, 0, cv::INTER_LINEAR);
                
#ifdef CUDA
            cv::gpu::FarnebackOpticalFlow farn;
            cv::gpu::GpuMat d_frame0(prev_grey_pyr_[iScale]);
            cv::gpu::GpuMat d_frame1(grey_warp);
            cv::gpu::GpuMat d_flowx(grey_warp.size(), CV_32FC1);
            cv::gpu::GpuMat d_flowy(grey_warp.size(), CV_32FC1);
            std::vector<cv::Mat> flow;

            farn(d_frame0, d_frame1, d_flowx, d_flowy);
            flow.push_back(cv::Mat(d_flowx));
            flow.push_back(cv::Mat(d_flowy));
            cv::merge(flow, flow_warp_pyr_[iScale]); 
#endif  
                
            width = grey_pyr_[iScale].cols;
            height = grey_pyr_[iScale].rows;

        memset(hog_mat_[iScale].desc, 0, hog_mat_[iScale].size*sizeof(float));                
        memset(hof_mat_[iScale].desc, 0, hof_mat_[iScale].size*sizeof(float));
        memset(mbh_mat_x_[iScale].desc, 0, mbh_mat_x_[iScale].size*sizeof(float));
        memset(mbh_mat_y_[iScale].desc, 0, mbh_mat_y_[iScale].size*sizeof(float));

        HogComp(prev_grey_pyr_[iScale], hog_mat_[iScale].desc, hog_info_);
        HofComp(flow_warp_pyr_[iScale], hof_mat_[iScale].desc, hof_info_);
        MbhComp(flow_warp_pyr_[iScale], mbh_mat_x_[iScale].desc, mbh_mat_y_[iScale].desc, mbh_info_);
		// track descriptor points in each scale separately
                
		for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
			int index = iTrack->index;
			cv::Point2f prev_point = iTrack->point[index];
			int x = cv::min(cv::max(cvRound(prev_point.x), 0), width-1);
			int y = cv::min(cv::max(cvRound(prev_point.y), 0), height-1);
			cv::Point2f point;
			RectInfo rect;

			point.x = prev_point.x + flow_pyr_[iScale].ptr<float>(y)[2*x];
			point.y = prev_point.y + flow_pyr_[iScale].ptr<float>(y)[2*x+1];

			if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
				iTrack = tracks.erase(iTrack);
				continue;
			}

			iTrack->disp[index].x = flow_warp_pyr_[iScale].ptr<float>(y)[2*x];
			iTrack->disp[index].y = flow_warp_pyr_[iScale].ptr<float>(y)[2*x+1];

			// get the descriptors for the descriptor point			
			GetRect(prev_point, rect, width, height, hog_info_);
                     
			GetDesc(&hog_mat_[iScale], rect, hog_info_, iTrack->hog, index);
			GetDesc(&hof_mat_[iScale], rect, hof_info_, iTrack->hof, index);
			GetDesc(&mbh_mat_x_[iScale], rect, mbh_info_, iTrack->mbhX, index);
			GetDesc(&mbh_mat_y_[iScale], rect, mbh_info_, iTrack->mbhY, index);
			iTrack->addPoint(point);				

			// if the trajectory achieves the maximal length
			if(iTrack->index >= track_info_.length) {
				std::vector<cv::Point2f> trajectory(track_info_.length+1);
				std::vector<cv::Point2f> displacement(track_info_.length);
				float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);

				for(int i = 0; i <= track_info_.length; ++i)
					trajectory[i] = iTrack->point[i]*fscales_[iScale];
				
				for (int i = 0; i < track_info_.length; ++i)
					displacement[i] = iTrack->disp[i]*fscales_[iScale];

				if (IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {
					std::vector<float> trj, hog, hof, mbh;

					// draw the valid trajectories at the first scale				
					if(show_track && iScale == 0 && tracks_image)
						DrawTrack(iTrack->point, iTrack->index, fscales_[iScale], (*tracks_image));

					for (int i = 0; i < track_info_.length; ++i)
					{
						trj.push_back(displacement[i].x);
						trj.push_back(displacement[i].y);
					}
					CollapsetDesc(iTrack->hog, hog, hog_info_, track_info_);
					CollapsetDesc(iTrack->hof, hof, hof_info_, track_info_);
					CollapsetDesc(iTrack->mbhX, mbh, mbh_info_, track_info_);
					CollapsetDesc(iTrack->mbhY, mbh, mbh_info_, track_info_);

					descriptor_set[TRJ].push_back(cv::Mat(trj).reshape(1,1));
					descriptor_set[HOG].push_back(cv::Mat(hog).reshape(1,1));
					descriptor_set[HOF].push_back(cv::Mat(hof).reshape(1,1));
					descriptor_set[MBH].push_back(cv::Mat(mbh).reshape(1,1));
					desc_cnt++;
				}

				iTrack = tracks.erase(iTrack);
				continue;
			}
			++iTrack;
		}

		// detect new descriptor points every gap frames
		std::vector<cv::Point2f> points(0);
		for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
			points.push_back(iTrack->point[iTrack->index]);

		DenseSample(grey_pyr_[iScale], points, g_quality, g_min_distance);
		// save the new descriptor points
		for (int i = 0; i < (int)points.size(); i++)
			tracks.push_back(Track(points[i], track_info_, hog_info_, hof_info_, mbh_info_));
	}

	grey_.copyTo(prev_grey_);
	for (int i = 0; i < g_scale_num; i++) {
		grey_pyr_[i].copyTo(prev_grey_pyr_[i]);             
		poly_pyr_[i].copyTo(prev_poly_pyr_[i]);
	}

	frames_cnt_ ++;

	return desc_cnt;
}



