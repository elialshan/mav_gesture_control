#include "operators.h"

#define ROI_SIZE	(100)


Operator::Operator(cvb::CvID id):id_(id) {
	prev_label_ = cur_label_ = BACKGROUND;
    action_angle_ = -1.0;
}        
      
void Operator::Process(const cv::Mat &frame) {                        
    ac_.ClassifyFrame(frame, &cur_label_);
    if ((cur_label_ == prev_label_))
        consistency_cnt_ ++;					
    else
        consistency_cnt_ = 0;
    prev_label_ = cur_label_;

    if (consistency_cnt_ < WINDOW_LENGTH)
		cur_label_ = BACKGROUND;				                        
            
    oe_.Evaluate(frame, (int)cur_label_, &action_angle_);            
}


void Operators::UpdateOperators()	{		
	//remove deleted tracks
	for (OperatorsMap::const_iterator it = ops_.begin(); it!=ops_.end(); ++it) {
		cvb::CvID op_id = it->second->get_id();
		if (tracks_.count(op_id) == 0) {
                    OperatorsMap::const_iterator next_op = it;
                    int track_id = it->first;				

                    for (std::vector<int>::iterator track_it = track_id_queue_.begin(); track_it != track_id_queue_.end(); ++track_it)
                        if (*track_it == track_id) {						
                                track_id_queue_.erase(track_it);
                                break;
                        }

                        delete it->second;                
                        std::advance(next_op, 1);
                        if (next_op == ops_.end()) {
                                ops_.erase(it->first);
                                break;
                        }
                        ops_.erase(it->first);
                        it = next_op;
		}
	}
	//add new tracks
	for (cvb::CvTracks::const_iterator it = tracks_.begin(); it!=tracks_.end(); ++it) {
		cvb::CvID track_id = it->second->id;
		if (ops_.count(track_id) == 0) {
			Operator *op = new Operator(track_id);
			ops_.insert(CvIDOperator(track_id, op));
			track_id_queue_.push_back(track_id);
		}			
	}
}

void Operators::StoreROI(const cv::Mat &frame, const char* output_path_char) {	
	static int cnt = 0;
	fs::path output_path;
	int dim = ROI_SIZE;

	output_path = fs::path(output_path_char);
	//CHECK(fs::exists(output_path));

	for (cvb::CvTracks::const_iterator it=tracks_.begin(); it!=tracks_.end(); ++it) {
		// Store 25 percent of potential operators
		if ((rand() % 100) <= 25) {
			cv::Mat op = cv::Mat::zeros(dim, dim, frame.type());
			cvb::CvTrack *track = (*it).second;			
			cv::Rect r1,r2;
			cv::Mat tmp;			
			char filename[50];

			sprintf(filename, "_object_%03d.jpg", cnt++);				
			r1 = cv::Rect(cv::Point(std::max<int>(0, (int)track->centroid.x - dim/2), std::max<int>(0, (int)track->centroid.y - dim/2)), 
				cv::Point(std::min<int>(frame.cols, (int)track->centroid.x + dim/2), std::min<int>(frame.rows, (int)track->centroid.y + dim/2)));
			tmp = frame(r1);
			r2 = cv::Rect((dim - tmp.cols)/2, (dim - tmp.rows)/2, tmp.cols, tmp.rows);
			tmp.copyTo(op(r2));		
			cv::imwrite(output_path.string() + std::string(filename), op);
		}			
	}//for (cvb::CvTracks::const_iterator it=opTracks.begin(); it!=opTracks.end(); ++it)	
}

void Operators::ExtractROI(const cv::Mat &frame, std::vector<OperatorROI> *roi, int max_operators=1) {
	for (cvb::CvTracks::const_iterator it=tracks_.begin(); it!=tracks_.end(); ++it) {			
		cv::Mat op = cv::Mat::zeros(ROI_SIZE, ROI_SIZE, frame.type());
		cvb::CvTrack *track = (*it).second;
		bool	processed_track = false;

		for (int i = 0; i < max_operators; i++)
			if (track->id == track_id_queue_[i]) {
				processed_track = true;
				break;
			}

		if (processed_track && (track->lifetime > 5)) {	
			cv::Rect r1,r2;
			cv::Mat tmp;				

			r1 = cv::Rect(cv::Point(std::max<int>(0, (int)track->centroid.x - ROI_SIZE/2), std::max<int>(0, (int)track->centroid.y - ROI_SIZE/2)), 
				cv::Point(std::min<int>(frame.cols, (int)track->centroid.x + ROI_SIZE/2), std::min<int>(frame.rows, (int)track->centroid.y + ROI_SIZE/2)));                               
			tmp = frame(r1);
			r2 = cv::Rect((ROI_SIZE - tmp.cols)/2, (ROI_SIZE - tmp.rows)/2, tmp.cols, tmp.rows);
			tmp.copyTo(op(r2));		
			roi->push_back(OperatorROI(track, op));
		}//if (track->lifetime > 5)
	}//for (cvb::CvTracks::const_iterator it=opTracks.begin(); it!=opTracks.end(); ++it)	
}
        
 