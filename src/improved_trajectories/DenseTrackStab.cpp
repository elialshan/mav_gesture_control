#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

int g_start_frame = 0;
int g_end_frame = INT_MAX;
int g_scale_num = 2;
const float g_scale_stride = sqrt(2.0f);

// parameters for descriptors
int g_patch_size = 16;//32;
int g_nhog_bins = 8;
int g_nxy_cell = 2;
int g_nt_cell = 3;
float g_epsilon = 0.05f;
const float g_min_flow = 0.4f;

// parameters for tracking
double g_quality = 0.2;//ea 0.001;
int g_min_distance = 5;
int g_init_gap = 1;
int g_track_length = 15;

// parameters for rejecting trajectory
const float g_min_var = sqrt(3.0f);
const float g_max_var = 50;
const float g_max_dis = 20;

