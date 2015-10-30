#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/archive/basic_binary_iprimitive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/thread/thread.hpp>


#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv/ml.h"
#include "opencv/highgui.h"

namespace fs = boost::filesystem;

/********************************* LOG *********************************/
#ifndef __func__
#define __func__ __FUNCTION__
#endif

#define LOGT() std::cout
#define LOGD() std::cout
#define LOGI() std::cout
#define LOGW() std::cout << __FILE__ << "(" << __LINE__ << "): Warning: " 
#define LOGE() std::cout << __FILE__ << "(" << __LINE__ << "): Error: " 
#define LOGF() std::cout << __FILE__ << "(" << __LINE__ << "): Fatal: " 


#define CHECK(cond) if (!(cond)) {LOGF() << "CHECK failed!" << std::endl; exit(1); }
/**************************************************************************************/

/********************************* VIDEO MANIPULATION *********************************/
void	RotateFrame(const cv::Mat &src, float angle, int rect_width, cv::Mat *dst);
float	AngleDif(float angle1, float angle2);

/**
* @brief Video capture wrapper
*/
class VideoHandle {
private:	
	float				angle_;
	int				rect_size_;
	float				scale_;
	cv::VideoCapture                capture_;

public:
	~VideoHandle();

	/**
   * @brief Set parameters
   *
   * @param filename video input path
   * @param angle video rotation angle
   * @param rect_size rotated video crop size
   * @param scale scale factor
   */
	int Init(const char* filename, const float angle = 0.0f, const int rect_size = 100, const float scale = 1.0f);

	/**
   * @brief Fetch new frame
   *
   * @param frame output
   */
	int GetFrame(cv::Mat *frame);

	/**
   * @brief Get video FPS
   */
	int get_fps() { return (int)capture_.get(CV_CAP_PROP_FPS);}

	/**
   * @brief Get video frame width
   */
	int get_frame_widht() { return (int)capture_.get(CV_CAP_PROP_FRAME_WIDTH);}

	/**
   * @brief Get video frame height
   */
	int get_frame_height() { return (int)capture_.get(CV_CAP_PROP_FRAME_HEIGHT);}
};


/**
* @brief Random numbers generator
*/
class RandGen {
public:
	RandGen();
	/**
	 * @brief Generate random float
     *
     * @param rng maximum random number
	 */
	float	RandFloat(float rng);

	/** 
	 * @brief Generate random int
     *
     * @param rng maximum random number
	 */
	int		RandInt(int rng);
};

/**
* @brief OpenCV matrix de-serializer
*/
class MatReader {	
	std::ifstream	*in_;
	bool			intialized_;
	boost::archive::binary_iarchive *ia_;	
	
public:
	MatReader();
	~MatReader();

	/**
	 * @brief Set input path
     *
     * @param filename input path
	 */
	int Init(std::string filename);
	void Deinit();

	/**
	 * @brief Read matrix
     *
     * @param m output
	 */
	int Read(cv::Mat *m);
};

/**
* @brief OpenCV matrix serializer
*/
class MatWriter {
	std::ofstream	*out_;
	bool			intialized_;
	boost::archive::binary_oarchive *oa_;
	
public:
	MatWriter();
	~MatWriter();

	/**
	 * @brief Set output path
     *
     * @param filename input path
	 */
	int Init(std::string filename);
	void Deinit();

	/**
	 * @brief Read matrix
     *
     * @param m input
	 */
	int Write(cv::Mat &m);
};

/**
 * @brief General cyclic buffer
 */
template <class T>
class CyclicQueue { 
private: 
	std::vector<T>	elems_;
	int				max_size_;

public: 
	CyclicQueue() {
		max_size_ = 0;
	}

	CyclicQueue(int max_size):max_size_(max_size) {}

	~CyclicQueue() {
		reset();
	}

	/**
   * @brief insert new element
   *
   * @param elem new element   
   */
	int enqueue(T const& elem) { 
		if (full()) {
			return -1;
		}

		elems_.push_back(elem);    
		return 0;
	} 

	/**
   * @brief get next element
   *
   * @param elem destanation pointer for the element
   */
	int dequeue(T *elem) 
	{ 
		if (empty()) { 
			return -1;
		}

		*elem = T(elems_.front());
		// remove last element 
		elems_.erase(elems_.begin());
		return 0;
	} 

	/**
   * @brief check if queue is empty   
   */
	bool empty() {
		return elems_.size() == 0;
	}

	/**
   * @brief check if queue is full
   */
	bool full() {
		return elems_.size() == max_size_;
	}

	/**
   * @brief delete queue elements
   */
	void reset() {
		elems_.clear();
	}
}; 
/**************************************************************************************/

#endif //UTILS_H