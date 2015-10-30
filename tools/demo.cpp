#include <fstream>
#include <iostream>
#include <string> 
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdio.h>
#include <map>
#include <queue>
#include <boost/asio.hpp>
#include <boost/thread.hpp>

#include "foreground_detector.h"
#include "tracker.h"
#include "frame_stitcher.h"
#include "operators.h"


class ThreadPool
{
private:
  boost::asio::io_service io_service_;
  boost::asio::io_service::work work_;
  boost::thread_group threads_;
  std::size_t available_;
  std::size_t used_;
  boost::mutex mutex_;
  boost::mutex lock_mutex_;
  boost::mutex::scoped_lock lock_;
  boost::condition_variable condition_;
public:

  /// @brief Constructor.
  ThreadPool(std::size_t pool_size): work_(io_service_), available_( pool_size )  {
	used_ = 0;
	lock_ = boost::mutex::scoped_lock(lock_mutex_);
    for ( std::size_t i = 0; i < pool_size; ++i) {
      threads_.create_thread(boost::bind( &boost::asio::io_service::run, &io_service_));
    }
  }

  /// @brief Destructor.
  ~ThreadPool()  {
    // Force all threads to return from io_service::run().
    io_service_.stop();

    // Suppress all exceptions.
    try {
      threads_.join_all();
    }
    catch ( ... ) {}
  }

  /// @brief Adds a task to the thread pool if a thread is currently available.
  template <typename Task>
  void run_task(Task task) {
    boost::unique_lock<boost::mutex> lock(mutex_);

    // If no threads are available, then return.
    if ( 0 == available_ ) return;

    // Decrement count, indicating thread is no longer available.
    --available_;
    ++used_;

    // Post a wrapped task into the queue.
    io_service_.post(boost::bind( &ThreadPool::wrap_task, this, boost::function<void()>(task)));
  }

  void wait_for_all() {
	  while (used_)
		  condition_.wait(lock_);
	}  

private:
  /// @brief Wrap a task so that the available count can be increased once
  ///        the user provided task has completed.
  void wrap_task( boost::function< void() > task )  {
    // Run the user supplied task.
    try {
      task();
    }
    // Suppress all exceptions.
    catch ( ... ) {}

    // Task has finished, so increment count of available threads.
    boost::unique_lock<boost::mutex> lock(mutex_);
    ++available_;
    --used_;    
    condition_.notify_one();
  }
};


int main(int argc, char *argv[]) {    
    cv::Mat         cur_frame, visual_map;

    //Visual mapping
    FrameStitcher   stitcher;        

    //Detection
    cvb::CvTracks   target_tracks;		
    CvSVM           operator_classifier;	
    Operators       operators;

    char            action_indicator[ACTIONS_NUM][3] = {"D","LR","RR","B"};	
    cv::Scalar      color[ACTIONS_NUM] = {CV_RGB(255.,0,0),CV_RGB(255.,0,0),CV_RGB(255.,0,0),CV_RGB(0,255.,0)};			
    float           radius = 3.0f;
    fs::path        base_path = fs::system_complete( fs::path( argv[0] ) ).parent_path();
    fs::path        op_class_path((base_path/fs::path("models")/fs::path("operator_detector")/fs::path("operator_classifier.model")).string().c_str());
    fs::path        video_path;
    std::string     video_name;
    VideoHandle     video_handle;
    int             option, frame_num = 0, max_operators = 1, op_active_th = 5, tar_active_th = 50;    
    
    
    while ((option = getopt(argc, argv,"i:m:o")) != -1) {
        switch (option) {
             case 'm' : max_operators = atoi(optarg); 
                 break;
            case 'i' : video_path = fs::path(optarg);
                break;
            default: LOGD() << "Usage: demo.bin -i <video_path> -m <maximum tracked operators>" << std::endl;
                 exit(EXIT_FAILURE);
        }
    }    
  
    video_name = fs::basename(video_path);
    //Set action names (must be the same names used during training)
    ActionClassifier::actions_names_[ACTION0] = "direction";
    ActionClassifier::actions_names_[ACTION1] = "lrotation";
    ActionClassifier::actions_names_[ACTION2] = "rrotation";        

    //Load classifiers
    CHECK(fs::exists(op_class_path));
    CHECK(fs::exists(video_path));
    CHECK(video_handle.Init(video_path.string().c_str(), 0.0f, 0) == 0);
    
    operator_classifier.load(op_class_path.string().c_str());	
    ActionClassifier::LoadStaticMembers(
            (base_path/fs::path("models")/fs::path("codebooks")).string().c_str(),
            (base_path/fs::path("models")/fs::path("action_classifiers")).string().c_str());
    OrientationEstimator::LoadStaticMembers(
            (base_path/fs::path("models")/fs::path("orientation_estimator")/fs::path("double_direction")).string().c_str(),
            (base_path/fs::path("models")/fs::path("orientation_estimator")/fs::path("single_direction")).string().c_str());
    
    ForegroundDetector	fd(video_handle.get_frame_height(), video_handle.get_frame_widht(), 32, 32, 0.5f);    

    ThreadPool pool(max_operators + 2);	
    
    LOGI() << "Processing " << video_name;    
    cv::namedWindow("Gesture Recognition");
    cv::moveWindow("Gesture Recognition", 0, 0);
    cv::namedWindow("Visual Map");
    cv::moveWindow("Visual Map", video_handle.get_frame_widht(), 0);
    
    while (video_handle.GetFrame(&cur_frame) == 0) {		
        cv::Mat                     stitcher_frame, stitcher_mask, framed_visual_map;
        cv::Mat                     gray, marked_frame, operator_mask, target_mask, object_mask;
        std::vector<OperatorROI>    op_roi;		
        cvb::CvBlobs                blobs;
        cur_frame.copyTo(marked_frame);
        pool.run_task(boost::bind(MarkTarget, boost::ref(fd), boost::ref(cur_frame), &target_tracks));

        GetObjectsMask(cur_frame.size(), *operators.get_tracks(), &object_mask);
        cv::cvtColor(cur_frame, gray, CV_RGB2GRAY);
        //calculate operator mask
        fd.GetOperatorMask(cur_frame, object_mask, &operator_mask);	

        cur_frame.copyTo(stitcher_frame);

        // Remove frames edges
        cv::Rect r(10, 10, stitcher_frame.cols - 20, stitcher_frame.rows - 20);
        stitcher_frame = stitcher_frame(r);
	object_mask.copyTo(stitcher_mask);
        stitcher_mask = stitcher_mask(r);

        stitcher_mask = 255 - stitcher_mask;                
        // Stitch frames to panorama	
        pool.run_task(boost::bind(&FrameStitcher::Stitch, &stitcher, stitcher_frame, stitcher_mask, &visual_map, &framed_visual_map));
		
        //detect and track operators
        TrackOperators(operator_mask, cur_frame, operator_classifier, operators.get_tracks());
        operators.UpdateOperators();			
        //draw detected operators
        RenderTracks(*operators.get_tracks(), op_active_th, CV_RGB(0,255.,0), OP_RECT_SIZE, &marked_frame);

	operators.ExtractROI(cur_frame, &op_roi, max_operators);       

        //classify each roi in a separate thread
        for (std::vector<OperatorROI>::iterator it=op_roi.begin(); it!=op_roi.end(); ++it){
            cvb::CvTrack *track = it->first;	
            cv::Mat op = it->second;
                
            pool.run_task(boost::bind(&Operator::Process, operators.get_operator(track->id), op));
        }

		pool.wait_for_all();
	
	//wait for orientation estimation to end and summarize the results
	for (std::vector<OperatorROI>::iterator it=op_roi.begin(); it!=op_roi.end(); ++it){
            cvb::CvTrack *track = it->first;	
            cv::Mat op = it->second;
            int cur_label;
            float action_angle;
			
            cur_label = operators.get_label(track->id);
            action_angle = (float)(M_PI/180) * operators.get_angle(track->id);			
            if (action_angle >= 0.0f) {                
                cv::Point pt1((int)track->centroid.x, (int)track->centroid.y);
                cv::Point pt2((int)(track->centroid.x + radius*cos(action_angle)), (int)(track->centroid.y - radius*sin(action_angle)));
                cv::line(marked_frame, pt1, pt2, CV_RGB(0, 255, 0), 2);
                cv::circle(marked_frame, pt2, 2, CV_RGB(0, 255, 0), 2);
            }
            else 
                cv::circle(marked_frame, cv::Point((int)track->centroid.x, (int)track->centroid.y), 2, CV_RGB(0, 255, 0), 2);
            
            cv::putText(marked_frame, std::string(action_indicator[(int)cur_label]), 
            cv::Point((int)track->centroid.x, (int)track->centroid.y), 0, 0.6, 
            color[(int)cur_label] , 2, 1);			
        }

        
	RenderTracks(target_tracks, tar_active_th, CV_RGB(255.,0,0), TAR_RECT_SIZE, &marked_frame);

        if (!framed_visual_map.empty())
            cv::imshow( "Visual Map", framed_visual_map );
        if (!marked_frame.empty()) {
            cv::imshow("Gesture Recognition",marked_frame);
            if (cv::waitKey(3) >= 0) break;
        }

        frame_num++;

    }
    printf("\n");

    return 0;
}