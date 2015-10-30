#ifndef DESCRIPTORS_H_
#define DESCRIPTORS_H_

#include "DenseTrackStab.h"
#include "opencv2/gpu/gpu.hpp"

// get the rectangle for computing the descriptor
void GetRect(const cv::Point2f& point, RectInfo& rect, const int width, const int height, const DescInfo& descInfo)
{
	int x_min = descInfo.width/2;
	int y_min = descInfo.height/2;
	int x_max = width - descInfo.width;
	int y_max = height - descInfo.height;

	rect.x = std::min<int>(std::max<int>(cvRound(point.x) - x_min, 0), x_max);
	rect.y = std::min<int>(std::max<int>(cvRound(point.y) - y_min, 0), y_max);
	rect.width = descInfo.width;
	rect.height = descInfo.height;
}


void BuildDescMat(const cv::Mat& xComp, const cv::Mat& yComp, float* desc, const DescInfo& descInfo)
{
	float maxAngle = 360.f;
	int nDims = descInfo.nBins;
	// one more bin for hof
	int nBins = descInfo.isHof ? descInfo.nBins-1 : descInfo.nBins;
	const float angleBase = float(nBins)/maxAngle;

	int step = (xComp.cols+1)*nDims;
	int index = step + nDims;
	for(int i = 0; i < xComp.rows; i++, index += nDims) {
		const float* xc = xComp.ptr<float>(i);
		const float* yc = yComp.ptr<float>(i);

		// summarization of the current line
		std::vector<float> sum(nDims);
		for(int j = 0; j < xComp.cols; j++) {
			float x = xc[j];
			float y = yc[j];
			float mag0 = sqrt(x*x + y*y);
			float mag1;
			int bin0, bin1;

			// for the zero bin of hof
			if(descInfo.isHof && mag0 <= g_min_flow) {
				bin0 = nBins; // the zero bin is the last one
				mag0 = 1.0;
				bin1 = 0;
				mag1 = 0;
			}
			else {
				float angle = cv::fastAtan2(y, x);
				if(angle >= maxAngle) angle -= maxAngle;

				// split the mag to two adjacent bins
				float fbin = angle * angleBase;
				bin0 = cvFloor(fbin);
				bin1 = (bin0+1)%nBins;

				mag1 = (fbin - bin0)*mag0;
				mag0 -= mag1;
			}

			sum[bin0] += mag0;
			sum[bin1] += mag1;

			for(int m = 0; m < nDims; m++, index++)
				desc[index] = desc[index-step] + sum[m];
		}
	}
}


void ExtractMagAndPhaseGPU(const cv::gpu::GpuMat &x_comp, const cv::gpu::GpuMat &y_comp, cv::Mat *magnitude, cv::Mat *phase) {
    cv::gpu::GpuMat d_xComp(x_comp);
        cv::gpu::GpuMat d_yComp(y_comp);
        cv::gpu::GpuMat d_magnitude, d_phase;
        
        cv::gpu::magnitude(d_xComp, d_yComp, d_magnitude);
        cv::gpu::phase(d_xComp, d_yComp, d_phase, true);
        
        *magnitude = cv::Mat (d_magnitude); 
        *phase = cv::Mat(d_phase);
}



void BuildDescMat2(const cv::Mat& magnitude, const cv::Mat& phase, float* desc, const DescInfo& descInfo)
{
	float maxAngle = 360.f;
	int nDims = descInfo.nBins;
	// one more bin for hof
	int nBins = descInfo.isHof ? descInfo.nBins-1 : descInfo.nBins;
	const float angleBase = float(nBins)/maxAngle;

	int step = (magnitude.cols+1)*nDims;
	int index = step + nDims;        
        
        
        //#pragma omp parallel for
	for(int i = 0; i < magnitude.rows; i++) {		
                const float* mag0_ptr = magnitude.ptr<float>(i);
		const float* angle_ptr = phase.ptr<float>(i);                

		// summarization of the current line
		std::vector<float> sum(nDims);
		for(int j = 0; j < magnitude.cols; j++) {			
			float mag0 = mag0_ptr[j];                        
			float mag1;
			int bin0, bin1;

			// for the zero bin of hof
			if(descInfo.isHof && mag0 <= g_min_flow) {
				bin0 = nBins; // the zero bin is the last one
				mag0 = 1.0;
				bin1 = 0;
				mag1 = 0;
			}
			else {
				float angle = angle_ptr[j];                                
                                if(angle >= maxAngle) angle -= maxAngle;

				// split the mag to two adjacent bins
				float fbin = angle * angleBase;
				bin0 = cvFloor(fbin);
				bin1 = (bin0+1)%nBins;

				mag1 = (fbin - bin0)*mag0;
				mag0 -= mag1;
			}

			sum[bin0] += mag0;
			sum[bin1] += mag1;

			for(int m = 0; m < nDims; m++, index++)
				desc[index] = desc[index-step] + sum[m];
		}
                index += nDims;
	}
}

// get a descriptor from the integral histogram
void GetDesc(const DescMat* descMat, RectInfo& rect, DescInfo descInfo, std::vector<float>& desc, const int index)
{
	int dim = descInfo.dim;
	int nBins = descInfo.nBins;
	int height = descMat->height;
	int width = descMat->width;

	int xStride = rect.width/descInfo.nxCells;
	int yStride = rect.height/descInfo.nyCells;
	int xStep = xStride*nBins;
	int yStep = yStride*width*nBins;

	// iterate over different cells
	int iDesc = 0;
	std::vector<float> vec(dim);
	for(int xPos = rect.x, x = 0; x < descInfo.nxCells; xPos += xStride, x++)
		for(int yPos = rect.y, y = 0; y < descInfo.nyCells; yPos += yStride, y++) {
			// get the positions in the integral histogram
			const float* top_left = descMat->desc + (yPos*width + xPos)*nBins;
			const float* top_right = top_left + xStep;
			const float* bottom_left = top_left + yStep;
			const float* bottom_right = bottom_left + xStep;

			for(int i = 0; i < nBins; i++) {
				float sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
				vec[iDesc++] = std::max<float>(sum, 0) + g_epsilon;
			}
		}

		float norm = 0;
		for(int i = 0; i < dim; i++)
			norm += vec[i];
		if(norm > 0) norm = 1.0f/norm;

		int pos = index*dim;
		for(int i = 0; i < dim; i++)
			desc[pos++] = sqrt(vec[i]*norm);
}

// for HOG descriptor
void HogComp(const cv::Mat& img, float* desc, DescInfo& descInfo)
{
	cv::Mat imgX, imgY;
	Sobel(img, imgX, CV_32F, 1, 0, 1);
	Sobel(img, imgY, CV_32F, 0, 1, 1);
	BuildDescMat(imgX, imgY, desc, descInfo);
}

// for HOF descriptor
void HofComp(const cv::Mat& flow, float* desc, DescInfo& descInfo)
{
	cv::Mat flows[2];
	split(flow, flows);
	BuildDescMat(flows[0], flows[1], desc, descInfo);
}

// for MBH descriptor
void MbhComp(const cv::Mat& flow, float* descX, float* descY, DescInfo& descInfo)
{
	cv::Mat flows[2];
	split(flow, flows);

	cv::Mat flowXdX, flowXdY, flowYdX, flowYdY;
	Sobel(flows[0], flowXdX, CV_32F, 1, 0, 1);
	Sobel(flows[0], flowXdY, CV_32F, 0, 1, 1);
	Sobel(flows[1], flowYdX, CV_32F, 1, 0, 1);
	Sobel(flows[1], flowYdY, CV_32F, 0, 1, 1);

	BuildDescMat(flowXdX, flowXdY, descX, descInfo);
	BuildDescMat(flowYdX, flowYdY, descY, descInfo);
}

// check whether a trajectory is valid or not
bool IsValid(std::vector<cv::Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length)
{
	int size = track.size();
	float norm = 1.0f/size;
	for(int i = 0; i < size; i++) {
		mean_x += track[i].x;
		mean_y += track[i].y;
	}
	mean_x *= norm;
	mean_y *= norm;

	for(int i = 0; i < size; i++) {
		float temp_x = track[i].x - mean_x;
		float temp_y = track[i].y - mean_y;
		var_x += temp_x*temp_x;
		var_y += temp_y*temp_y;
	}
	var_x *= norm;
	var_y *= norm;
	var_x = sqrt(var_x);
	var_y = sqrt(var_y);

	// remove static trajectory
	if(var_x < g_min_var && var_y < g_min_var)
		return false;
	// remove random trajectory
	if( var_x > g_max_var || var_y > g_max_var )
		return false;

	float cur_max = 0;
	for(int i = 0; i < size-1; i++) {
		track[i] = track[i+1] - track[i];
		float temp = sqrt(track[i].x*track[i].x + track[i].y*track[i].y);

		length += temp;
		if(temp > cur_max)
			cur_max = temp;
	}

	if(cur_max > g_max_dis && cur_max > length*0.7)
		return false;

	track.pop_back();
	norm = 1.0f/length;
	// normalize the trajectory
	for(int i = 0; i < size-1; i++)
		track[i] *= norm;

	return true;
}

bool IsCameraMotion(std::vector<cv::Point2f>& disp)
{
	float disp_max = 0;
	float disp_sum = 0;
	for(int i = 0; i < (int)disp.size(); ++i) {
		float x = disp[i].x;
		float y = disp[i].y;
		float temp = sqrt(x*x + y*y);

		disp_sum += temp;
		if(disp_max < temp)
			disp_max = temp;
	}

	if(disp_max <= 1)
		return false;

	float disp_norm = 1.0f/disp_sum;
	for (int i = 0; i < (int)disp.size(); ++i)
		disp[i] *= disp_norm;

	return true;
}

// detect new feature points in an image without overlapping to previous points
void DenseSample(const cv::Mat& grey, std::vector<cv::Point2f>& points, const double g_quality, const int g_min_distance)
{
	int width = grey.cols/g_min_distance;
	int height = grey.rows/g_min_distance;

	cv::Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);

	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*g_quality;

	std::vector<int> counters(width*height);
	int x_max = g_min_distance*width;
	int y_max = g_min_distance*height;

	for(int i = 0; i < (int)points.size(); i++) {
		cv::Point2f point = points[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
                
                //nan test
                if ((point.x != point.x) || (point.y != point.y))
                    continue;

		if(x >= x_max || y >= y_max)
                    continue;
		x /= g_min_distance;
		y /= g_min_distance;
		counters[y*width+x]++;
	}

	points.clear();
	int index = 0;
	int offset = g_min_distance/2;
	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++, index++) {
			if(counters[index] > 0)
				continue;

			int x = j*g_min_distance+offset;
			int y = i*g_min_distance+offset;

			if(eig.at<float>(y, x) > threshold)
				points.push_back(cv::Point2f(float(x), float(y)));
		}
}

void InitPry(const cv::Mat& frame, std::vector<float>& scales, std::vector<cv::Size>& sizes)
{
	int rows = frame.rows, cols = frame.cols;
	float min_size = (float)std::min<int>(rows, cols);

	int nlayers = 0;
	while(min_size >= g_patch_size) {
		min_size /= g_scale_stride;
		nlayers++;
	}

	if(nlayers == 0) nlayers = 1; // at least 1 scale 

	g_scale_num = std::min<int>(g_scale_num, nlayers);

	scales.resize(g_scale_num);
	sizes.resize(g_scale_num);

	scales[0] = 1.;
	sizes[0] = cv::Size(cols, rows);

	for(int i = 1; i < g_scale_num; i++) {
		scales[i] = scales[i-1] * g_scale_stride;
		sizes[i] = cv::Size(cvRound(cols/scales[i]), cvRound(rows/scales[i]));
	}
}

void BuildPry(const std::vector<cv::Size>& sizes, const int type, std::vector<cv::Mat>& grey_pyr)
{
	int nlayers = sizes.size();
	grey_pyr.resize(nlayers);

	for(int i = 0; i < nlayers; i++)
		grey_pyr[i].create(sizes[i], type);
}

void DrawTrack(const std::vector<cv::Point2f>& point, const int index, const float scale, cv::Mat& image)
{
	cv::Point2f point0 = point[0];
	point0 *= scale;

	for (int j = 1; j <= index; j++) {
		cv::Point2f point1 = point[j];
		point1 *= scale;

		line(image, point0, point1, cv::Scalar(0,cvFloor(255.0*(j+1.0)/float(index+1.0)),0), 2, 8, 0);
		point0 = point1;
	}
	circle(image, point0, 2, cv::Scalar(0,0,255), -1, 8, 0);
}

void PrintDesc(std::vector<float>& desc, DescInfo& descInfo, TrackInfo& trackInfo)
{
	int tStride = cvFloor(trackInfo.length/descInfo.ntCells);
	float norm = 1.0f/float(tStride);
	int dim = descInfo.dim;
	int pos = 0;
	for(int i = 0; i < descInfo.ntCells; i++) {
		std::vector<float> vec(dim);
		for(int t = 0; t < tStride; t++)
			for(int j = 0; j < dim; j++)
				vec[j] += desc[pos++];
		for(int j = 0; j < dim; j++)
			printf("%.7f\t", vec[j]*norm);
	}
}

void CollapsetDesc(std::vector<float>& inDesc, std::vector<float>& outDesc, DescInfo& descInfo, TrackInfo& trackInfo)
{
	int tStride = cvFloor(trackInfo.length/descInfo.ntCells);
	float norm = 1.0f/float(tStride);
	int dim = descInfo.dim;
	int pos = 0;
	for(int i = 0; i < descInfo.ntCells; i++) {
		std::vector<float> vec(dim);
		for(int t = 0; t < tStride; t++)
			for(int j = 0; j < dim; j++)
				vec[j] += inDesc[pos++];		
		for(int j = 0; j < dim; j++)
			outDesc.push_back(vec[j]*norm);
	}
}

void MyWarpPerspective(cv::Mat& prev_src, cv::Mat& src, cv::Mat& dst, cv::Mat& M0, int flags,
	int borderType, const cv::Scalar& borderValue)
{
	int width = src.cols;
	int height = src.rows;
	dst.create( height, width, CV_8UC1 );

	cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
	const int margin = 5;

	const int BLOCK_SZ = 32;
	short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];

	int interpolation = flags & cv::INTER_MAX;
	if( interpolation == cv::INTER_AREA )
		interpolation = cv::INTER_LINEAR;

	double M[9];
	cv::Mat matM(3, 3, CV_64F, M);
	M0.convertTo(matM, matM.type());
	if( !(flags & cv::WARP_INVERSE_MAP) )
		invert(matM, matM);

	int x, y, x1, y1;

	int bh0 = cv::min(BLOCK_SZ/2, height);
	int bw0 = cv::min(BLOCK_SZ*BLOCK_SZ/bh0, width);
	bh0 = cv::min(BLOCK_SZ*BLOCK_SZ/bw0, height);

	for( y = 0; y < height; y += bh0 ) {
		for( x = 0; x < width; x += bw0 ) {
			int bw = cv::min( bw0, width - x);
			int bh = cv::min( bh0, height - y);

			cv::Mat _XY(bh, bw, CV_16SC2, XY);
			cv::Mat matA;
			cv::Mat dpart(dst, cv::Rect(x, y, bw, bh));

			for( y1 = 0; y1 < bh; y1++ ) {

				short* xy = XY + y1*bw*2;
				double X0 = M[0]*x + M[1]*(y + y1) + M[2];
				double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
				double W0 = M[6]*x + M[7]*(y + y1) + M[8];
				short* alpha = A + y1*bw;

				for( x1 = 0; x1 < bw; x1++ ) {

					double W = W0 + M[6]*x1;
					W = W ? cv::INTER_TAB_SIZE/W : 0;
					double fX = cv::max((double)INT_MIN, cv::min((double)INT_MAX, (X0 + M[0]*x1)*W));
					double fY = cv::max((double)INT_MIN, cv::min((double)INT_MAX, (Y0 + M[3]*x1)*W));

					double _X = fX/double(cv::INTER_TAB_SIZE);
					double _Y = fY/double(cv::INTER_TAB_SIZE);

					if( _X > margin && _X < width-1-margin && _Y > margin && _Y < height-1-margin )
						mask.at<uchar>(y+y1, x+x1) = 1;

					int X = cv::saturate_cast<int>(fX);
					int Y = cv::saturate_cast<int>(fY);

					xy[x1*2] = cv::saturate_cast<short>(X >> cv::INTER_BITS);
					xy[x1*2+1] = cv::saturate_cast<short>(Y >> cv::INTER_BITS);
					alpha[x1] = (short)((Y & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (X & (cv::INTER_TAB_SIZE-1)));
				}
			}

			cv::Mat _matA(bh, bw, CV_16U, A);
			remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
		}
	}

	for( y = 0; y < height; y++ ) {
		const uchar* m = mask.ptr<uchar>(y);
		const uchar* s = prev_src.ptr<uchar>(y);
		uchar* d = dst.ptr<uchar>(y);
		for( x = 0; x < width; x++ ) {
			if(m[x] == 0)
				d[x] = s[x];
		}
	}
}

void ComputeMatch(const std::vector<cv::KeyPoint>& prev_kpts, const std::vector<cv::KeyPoint>& kpts,
	const cv::Mat& prev_desc, const cv::Mat& desc, std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& pts)
{
	prev_pts.clear();
	pts.clear();

	if(prev_kpts.size() == 0 || kpts.size() == 0)
		return;

	cv::Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);

	cv::BFMatcher desc_matcher(cv::NORM_L2);
	std::vector<cv::DMatch> matches;

	desc_matcher.match(desc, prev_desc, matches, mask);

	prev_pts.reserve(matches.size());
	pts.reserve(matches.size());

	for(size_t i = 0; i < matches.size(); i++) {
		const cv::DMatch& dmatch = matches[i];
		// get the point pairs that are successfully matched
		prev_pts.push_back(prev_kpts[dmatch.trainIdx].pt);
		pts.push_back(kpts[dmatch.queryIdx].pt);
	}

	return;
}

void MergeMatch(const std::vector<cv::Point2f>& prev_pts1, const std::vector<cv::Point2f>& pts1,
	std::vector<cv::Point2f>& prev_pts_all, std::vector<cv::Point2f>& pts_all)
{
	prev_pts_all.clear();
	prev_pts_all.reserve(prev_pts1.size());

	pts_all.clear();
	pts_all.reserve(pts1.size());

	for(size_t i = 0; i < prev_pts1.size(); i++) {
		prev_pts_all.push_back(prev_pts1[i]);
		pts_all.push_back(pts1[i]);
	}

	return;
}

void MatchFromFlow(const cv::Mat& prev_grey, const cv::Mat& flow, std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& pts, const cv::Mat& mask)
{
	int width = prev_grey.cols;
	int height = prev_grey.rows;
	prev_pts.clear();
	pts.clear();

	const int MAX_COUNT = 1000;
	goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);

	if(prev_pts.size() == 0)
		return;

	for(int i = 0; i < (int)prev_pts.size(); i++) {
		int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
		int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

		const float* f = flow.ptr<float>(y);
		pts.push_back(cv::Point2f(x+f[2*x], y+f[2*x+1]));
	}
}

#endif /*DESCRIPTORS_H_*/
