#include "hog.h"



void BuildHogMat(const cv::Mat& x_comp, const cv::Mat& y_comp, int nbins, float* desc) {
	float max_angle = 360.f;		
	const float angle_base = float(nbins)/max_angle;

	for(int i = 0; i < x_comp.rows; i++) {
		const float* xc = x_comp.ptr<float>(i);
		const float* yc = y_comp.ptr<float>(i);

		// summarization of the current line
		for(int j = 0; j < x_comp.cols; j++) {
			float x = xc[j];
			float y = yc[j];
			float mag0 = sqrt(x*x + y*y);
			float mag1;
			int bin0, bin1;			
			float angle = cv::fastAtan2(y, x);

			if(angle >= max_angle) 
				angle -= max_angle;

			// split the mag to two adjacent bins
			float fbin = angle * angle_base;
			bin0 = cvFloor(fbin);
			bin1 = (bin0+1)%nbins;

			mag1 = (fbin - bin0)*mag0;
			mag0 -= mag1;

			desc[bin0] += mag0;
			desc[bin1] += mag1;			
		}
	}
}

void NormDesc(float *desc, int nbins) {
	float norm = 0.0f, g_epsilon = 0.000001f;

	for(int i = 0; i < nbins; i++)
		norm += desc[i]*desc[i];

	norm = 1.0f/(sqrt(norm)+g_epsilon);

	for(int i = 0; i < nbins; i++)
		desc[i] = desc[i]*norm;
}

void ExtractHog(const cv::Mat &org_img, int nx_blocks, int ny_blocks, int nbins, cv::Mat *frame_desc) {
	cv::Mat img, imgX, imgY;
	int ndims = nbins * nx_blocks * ny_blocks;
	std::vector<float> desc(ndims, 0);

	if (org_img.channels() > 1)
		cvtColor(org_img, img, CV_BGR2GRAY);
	else
		img = org_img;

	if ((img.rows != OP_RECT_SIZE) || (img.cols != OP_RECT_SIZE))
		cv::resize(img, img, cv::Size(OP_RECT_SIZE, OP_RECT_SIZE));

	Sobel(img, imgX, CV_32F, 1, 0, 1);
	Sobel(img, imgY, CV_32F, 0, 1, 1);

	for (int nx = 0; nx < nx_blocks; nx++)	{
		for (int ny=0; ny < ny_blocks; ny++) {
			cv::Rect r(ny*img.rows/ny_blocks, nx*img.cols/nx_blocks, img.rows/ny_blocks, img.cols/nx_blocks);
			BuildHogMat(imgX(r), imgY(r), nbins, &desc[nbins * (nx_blocks * nx + ny)]);				
		}		
	}	
	NormDesc(&desc[0], ndims);
	frame_desc->push_back(cv::Mat(desc).reshape(1, 1));
}