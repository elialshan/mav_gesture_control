#include "utils.h"

namespace boost {
	namespace serialization {
		template<class Archive>
		void serialize(Archive &ar, cv::Mat& mat, const unsigned int)
		{
			int cols, rows, type;
			bool continuous;

			if (Archive::is_saving::value) {
				cols = mat.cols; rows = mat.rows; type = mat.type();
				continuous = mat.isContinuous();
			}

			ar & cols & rows & type & continuous;

			if (Archive::is_loading::value)
				mat.create(rows, cols, type);

			if (continuous) {
				const unsigned int data_size = rows * cols * mat.elemSize();
				ar & boost::serialization::make_array(mat.ptr(), data_size);
			} else {
				const unsigned int row_size = cols*mat.elemSize();
				for (int i = 0; i < rows; i++) {
					ar & boost::serialization::make_array(mat.ptr(i), row_size);
				}
			}

		}
	}
}

MatReader::MatReader() {
    in_ = NULL;
	ia_ = NULL;
	intialized_ = false;
}

MatReader::~MatReader() {
	Deinit();
}

int MatReader::Init(std::string filename) {	
	try {
            in_ = new std::ifstream(filename.c_str(), std::ios::in | std::ios::binary);
            ia_ = new boost::archive::binary_iarchive(*in_);
	} catch (const boost::archive::archive_exception &e) {
            LOGE() << "Unable to load " <<  filename << ", code: " << e.code << std::endl;
            return -1;
	}
	intialized_ = true;
	return 0;
}

void MatReader::Deinit() {
        if (ia_) {
            delete ia_;
            ia_ = NULL;
        }
        if (in_) {
            in_->close();
            delete in_;	
            in_ = NULL;
        }            
	intialized_ = false;
}

int MatReader::Read(cv::Mat *m) {
	if (!intialized_)
		return -1;

	try {
		(*ia_) >> (*m);
	} catch (const boost::archive::archive_exception &e) {
		if (e.code != boost::archive::archive_exception::input_stream_error)
			LOGE() << "Unable to read mat, code: " << e.code << std::endl;
		return -1;
	}
	if ((m->rows == 0) || (m->cols == 0)) {
		m->release();
	}


	return 0;
}

MatWriter::MatWriter() {
        out_ = NULL;
	oa_ = NULL;
	intialized_ = false;
}

MatWriter::~MatWriter() {
	Deinit();
}

int MatWriter::Init(std::string filename) {	
	out_ = new std::ofstream(filename.c_str(), std::ios::out | std::ios::binary);
	oa_ = new boost::archive::binary_oarchive(*out_);
	intialized_ = true;
	return 0;
}

void MatWriter::Deinit() {
        if (oa_) {
		delete oa_;    
                oa_ = NULL;
        }        
        if (out_) {
            out_->close();
            delete out_;
            out_ = NULL;
        }
	intialized_ = false;
}

int MatWriter::Write(cv::Mat &m) {
	if (!intialized_)
		return -1;

	try {
		(*oa_) << m;
	} catch (const boost::archive::archive_exception &e) {
		LOGE() << "Unable to write mat, code: " << e.code << std::endl;
		return -1;
	}
	return 0;
}

void RotateFrame(const cv::Mat &src, float angle, int rect_width, cv::Mat *dst) {
	cv::Mat _dst(src.rows, src.cols, src.type());
	if (angle)	{
		cv::Mat bordered_source;
		int top,bottom,left,right;				

		top=bottom=left=right=20;
		copyMakeBorder( src, bordered_source, top, bottom, left, right, cv::BORDER_CONSTANT,cv::Scalar() );
		cv::Point2f src_center(bordered_source.cols/2.0F, bordered_source.rows/2.0F);
		cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
		warpAffine(bordered_source, _dst, rot_mat, bordered_source.size());   				
		_dst.copyTo((*dst));
	}
	else
		src.copyTo((*dst));

	if (rect_width) {
		cv::Rect r(((*dst).rows-rect_width)/2, ((*dst).cols-rect_width)/2, rect_width,rect_width);
		(*dst) = (*dst)(r);
	}	
}


float AngleDif(float angle1, float angle2) {
	if (angle1 < angle2) {
		float tmp = angle1;
		angle1 = angle2;
		angle2 = tmp;
	}
	return cv::min(angle1 - angle2, 360 - angle1 + angle2);
}

VideoHandle::~VideoHandle() {
	if (capture_.isOpened())
		capture_.release();
}

int VideoHandle::Init(const char* filename, const float angle, const int rect_size, const float scale) {
	capture_ = cv::VideoCapture(filename);
	if (!capture_.isOpened()) {
		LOGE() << "Could not open <<" << filename << std::endl;
		return -1;
	}
	angle_ = angle;
	rect_size_ = rect_size;
	scale_ = scale;

	return 0;
}



int VideoHandle::GetFrame(cv::Mat *frame){		
	capture_ >> (*frame);
	if (frame->empty())
		return -1;	

	if (scale_ != 1.0f)
		resize((*frame), (*frame), cv::Size((int)(frame->cols*scale_), (int)(frame->rows*scale_)));	
	RotateFrame((*frame), angle_, rect_size_, frame);
	return 0;
}

RandGen::RandGen() {
	srand ((int)time(NULL));
}

float	RandGen::RandFloat(float rng) {
	return ((float)(rand() % (int)(RAND_MAX*rng)))/RAND_MAX;
}

int		RandGen::RandInt(int rng) {
	return rand() % rng;
}
