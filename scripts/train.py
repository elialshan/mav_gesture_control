__author__ = 'Eli'

import os
import shutil
from multiprocessing import Pool,cpu_count
import subprocess
import sys
import glob
import argparse

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_NUM = max(1, cpu_count() - 2)

def clean_dir(folder):
    print 'Cleaning %s' % folder	
    cmd = 'rm $(find  %s -type f -empty)' % folder    
    print cmd    
    os.system(cmd)
    print 'Done'

def submit_task(cmd_str):
    args = cmd_str.split('!')
    log_path = args[-2]
    task_name = args[-1]
    args = args[:-2]
    f_out = open(log_path + '.out', 'w')
    f_err = open(log_path + '.err', 'w')
    cmd = ' '.join(args)
    f_out.write('%s\n' % cmd)
    print cmd
    p = subprocess.Popen(args, stdout=f_out, stderr=f_err, shell=False)
    print task_name
    p.communicate()
    f_out.close()
    f_err.close()
    print '%s complete' % task_name


def train_operator_classifier(bin, data_dir, output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args = [bin, 'train_op_class', data_dir, output_path]
    task_name = 'operator classifier training'
    log_path = os.path.join(output_dir, 'operator_classifier.log')
    f_out = open(log_path + '.out', 'w')
    f_err = open(log_path + '.err', 'w')
    cmd = ' '.join(args)
    print cmd
    f_out.write('%s\n' % cmd)
    p = subprocess.Popen(args, stdout=f_out, stderr=f_err, shell=False)
    print task_name
    p.communicate()
    f_out.close()
    f_err.close()
    p = subprocess.Popen(args)
    p.communicate()
    print '%s complete' % task_name

def evaluate_detection(bin, mav_videos_dir, models_dir, gt_path, output_dir):
    mav_videos = glob.glob(os.path.join(mav_videos_dir, '*.mp4'))
    for video_path in mav_videos:
        fname = os.path.basename(video_path)
        op_gt = os.path.join(gt_path, fname.replace('.mp4', '.op.gt.csv'))
        tar_gt = os.path.join(gt_path, fname.replace('.mp4', '.tar.gt.csv'))
        if os.path.exists(op_gt) and os.path.exists(tar_gt):
            args = [bin, 'evaluate_detection', video_path, models_dir, gt_path, output_dir]
            log_path = os.path.join(output_dir, fname + '_detection_test.log')
            res_path = os.path.join(output_dir, fname.replace('.mp4', '_track_res.txt'))
            if os.path.exists(res_path):				
				print '%s detection evaluation results found (%s). Skipping.' %(fname, res_path)
				continue
		
            task_name = fname + ' detection evaluation'
            f_out = open(log_path + '.out', 'w')
            f_err = open(log_path + '.err', 'w')
            cmd = ' '.join(args)
            print cmd
            f_out.write('%s\n' % cmd)
            p = subprocess.Popen(args, stdout=f_out, stderr=f_err, shell=False)
            print task_name
            p.communicate()
            f_out.close()
            f_err.close()
            print '%s complete' % task_name

def extract_trj_desc(bin, videos_dir, output_dir):
    video_paths = os.listdir(videos_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args = ['%s!extract_trj!%s!%s!%s!%s' %
            (bin, os.path.join(videos_dir, video_path), output_dir,
             os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + '.log'),
            '%s trajectories extraction' % os.path.basename(video_path)) for video_path in video_paths]
    pool = Pool(processes=PROC_NUM)
    pool.map(submit_task, args)

def calc_codebooks(bin, trajectories_dir_path, cb_dir):
    desc_types = ['TRJ', 'HOG', 'HOF', 'MBH']
    if not os.path.exists(cb_dir):
        os.makedirs(cb_dir)

    args = ['%s!calc_cb!%s!%s!%s!%s!%s' %
            (bin, trajectories_dir_path, desc_type, cb_dir,
             os.path.join(cb_dir, desc_type + '.log'),
            '%s codebook calculation' % desc_type) for desc_type in desc_types]
    pool = Pool(processes=PROC_NUM)
    pool.map(submit_task, args)

def extract_action_desc(bin, trj_desc_dir, cb_dir, output_dir):
    trj_desc_paths = os.listdir(os.path.join(trj_desc_dir, 'TRJ'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args = ['%s!extract_action_desc!%s!%s!%s!%s!%s!%s' %
            (bin, trj_desc_dir, cb_dir, trj_desc_path, output_dir,
             os.path.join(output_dir, os.path.basename(trj_desc_path).split('.')[0] + '.log'),
            '%s action descriptors extraction' % os.path.basename(trj_desc_path)) for trj_desc_path in trj_desc_paths]
    pool = Pool(processes=PROC_NUM)
    pool.map(submit_task, args)

def split_data_set(bin, base_path, split_path):
    if not os.path.exists(split_path):
        os.makedirs(split_path)

    args = [bin, 'split_data_set', base_path, split_path]
    task_name = 'split train data set'
    p = subprocess.Popen(args)
    print task_name
    p.communicate()
    print '%s complete' % task_name

def calc_base_action_classifiers(bin, train_data, test_data, output_dir):
    desc_types = ['TRJ', 'HOG', 'HOF', 'MBH']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args = ['%s!train_base_action!%s!%s!%s!%s!%s!%s' %
            (bin, train_data, test_data, desc_type, output_dir,
            os.path.join(output_dir, desc_type + '.log'),
            '%s classifiers training' % desc_type) for desc_type in desc_types]
    pool = Pool(processes=PROC_NUM)
    pool.map(submit_task, args)


def calc_combined_action_classifiers(bin, train_data, test_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args = [bin, 'train_combined_action', train_data, test_data, output_dir]
    log_path = os.path.join(output_dir, 'combined.log')
    task_name = 'combined classifier training'

    f_out = open(log_path + '.out', 'w')
    f_err = open(log_path + '.err', 'w')
    cmd = ' '.join(args)
    print cmd
    f_out.write('%s\n' % cmd)
    p = subprocess.Popen(args, stdout=f_out, stderr=f_err, shell=False)
    print task_name
    p.communicate()
    f_out.close()
    f_err.close()
    print '%s complete' % task_name


def extract_orientation_desc(bin, videos_dir, output_dir):
    video_paths = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args = ['%s!extract_orientation!%s!%s!%s!%s' %
            (bin, os.path.join(videos_dir, video_path), output_dir,
             os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + '.log'),
            '%s orientation descriptors extraction' % os.path.basename(video_path)) for video_path in video_paths]
    pool = Pool(processes=PROC_NUM)
    pool.map(submit_task, args)

def calc_orientation_estimator(bin, train_data, test_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args = [bin, 'train_orientation_estimator', train_data, test_data, output_dir]
    log_path = os.path.join(output_dir, 'orientation_estimator.log')
    task_name = 'orientation estimation training'
    with open(log_path, 'w') as f:
        p = subprocess.Popen(args, stdout=f, shell=False)
        print task_name
        p.communicate()
        print '%s complete' % task_name



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='MAV Gesture Cotrol demo')    
    parser.add_argument('--train_data_dir', dest='train_data_dir',
                        help='Training data folder', type=str)
    parser.add_argument('--train_output_dir', dest='train_output_dir', 
						help='temporary training folder', type=str)
    parser.add_argument('--models_dir', dest='models_dir', 
						help='new models output folder', type=str)
    parser.add_argument('--mav_videos_dir', dest='mav_videos_dir', 
						help='MAV video folder', type=str)


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    
    train_data_dir = args.train_data_dir
    train_output_dir = args.train_output_dir
    models_dir = args.models_dir
    mav_videos_dir = args.mav_videos_dir
    if os.path.exists(models_dir):
		print '%s already exists. Please set different output folder' % models_dir
		sys.exit()

    src_bin_path = os.path.join(SELF_DIR, '..', '.build_release', 'tools', 'train.bin')
    if not os.path.exists(os.path.join(SELF_DIR, 'bin')):
		os.makedirs(os.path.join(SELF_DIR, 'bin'))
    dst_bin_path = os.path.join(SELF_DIR, 'bin', 'train.bin')
    dst_bin_path = os.path.normpath(dst_bin_path)

    if os.path.exists(dst_bin_path):
        os.remove(dst_bin_path)
    shutil.copy(src_bin_path, dst_bin_path)
    
    # TRAIN OPERATOR CLASSIFIER
    data_path = os.path.join(train_data_dir, os.path.join('detection', 'operator_classifer_data'))
    data_path = os.path.normpath(data_path)
    output_dir = os.path.join(train_output_dir, 'operator_detector')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'operator_classifier.model')
    output_path = os.path.normpath(output_path)
    train_operator_classifier(dst_bin_path, data_path, output_path)
    clean_dir(output_dir)
    dst_dir = os.path.join(models_dir, os.path.basename(output_dir))
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(output_dir, dst_dir)

    # EVALUATE DETECTION
    output_dir = os.path.join(train_output_dir, 'operator_detector')
    gt_path = os.path.join(train_data_dir, os.path.join('detection', 'evaluation_ground_truth'))
    gt_path = os.path.normpath(gt_path)
    evaluate_detection(dst_bin_path, mav_videos_dir, models_dir, gt_path, output_dir)  
    clean_dir(output_dir)

    # EXTRACT TRAIN SET DENSE TRAJECTORIES
    train_videos_dir = os.path.join(train_data_dir, 'gestures', 'gesture_recognition', 'train')
    train_videos_dir = os.path.normpath(train_videos_dir)
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'dense_trajectories', 'train')
    output_dir = os.path.normpath(output_dir)
    extract_trj_desc(dst_bin_path, train_videos_dir, output_dir)
    clean_dir(output_dir)

    # EXTRACT TEST SET DENSE TRAJECTORIES
    test_videos_dir = os.path.join(train_data_dir, 'gestures', 'gesture_recognition', 'test')
    test_videos_dir = os.path.normpath(test_videos_dir)
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'dense_trajectories', 'test')
    output_dir = os.path.normpath(output_dir)
    extract_trj_desc(dst_bin_path, test_videos_dir, output_dir)
    clean_dir(output_dir)

    # COMPUTE CODEBOOKS
    trajectories_dir_path = os.path.join(train_output_dir, 'action_recognition_data', 'dense_trajectories', 'train')
    trajectories_dir_path = os.path.normpath(trajectories_dir_path)
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'codebooks')
    output_dir = os.path.normpath(output_dir)
    calc_codebooks(dst_bin_path, trajectories_dir_path, output_dir)
    clean_dir(output_dir)
    dst_dir = os.path.join(models_dir, os.path.basename(output_dir))
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(output_dir, dst_dir)

    # EXTRACT AND SPLIT TRAIN SET ACTION DESCRIPTORS
    train_trj_desc_dir = os.path.join(train_output_dir, 'action_recognition_data', 'dense_trajectories', 'train')
    cb_path = os.path.join(train_output_dir, 'action_recognition_data', 'codebooks')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'action_descriptors', 'train')
    split_path = os.path.join(train_output_dir, 'action_recognition_data', 'action_descriptors', 'train_combined')
    # merge trainig set to avoid redundant extraction
    if os.path.exists(split_path):
        descs = ['TRJ', 'HOF', 'HOG', 'MBH']
        for desc in descs:
            split_desc_dir_name = os.path.join(split_path, desc)
            base_desc_dir_name = os.path.join(output_dir, desc)
            if os.path.exists(split_desc_dir_name):
                if os.path.exists(output_dir):
                    fnames = os.listdir(split_desc_dir_name)
                    for fname in fnames:
                        shutil.move(os.path.join(split_desc_dir_name, fname), os.path.join(output_dir, fname))
                shutil.rmtree(split_desc_dir_name)
        shutil.rmtree(split_path)

    extract_action_desc(dst_bin_path, train_trj_desc_dir, cb_path, output_dir)
    clean_dir(output_dir)
    split_data_set(dst_bin_path, output_dir, split_path)


    # EXTRACT TEST SET ACTION DESCRIPTORS
    test_trj_desc_dir = os.path.join(train_output_dir, 'action_recognition_data', 'dense_trajectories', 'test')
    cb_path = os.path.join(train_output_dir, 'action_recognition_data', 'codebooks')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'action_descriptors', 'test')
    extract_action_desc(dst_bin_path, test_trj_desc_dir, cb_path, output_dir)
    clean_dir(output_dir)

    # TRAIN BASE CLASSIFIERS
    train_data = os.path.join(train_output_dir, 'action_recognition_data', 'action_descriptors', 'train')
    test_data = os.path.join(train_output_dir, 'action_recognition_data', 'action_descriptors', 'test')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'action_classifiers')
    calc_base_action_classifiers(dst_bin_path, train_data, test_data, output_dir)
    clean_dir(output_dir)

    # TRAIN COMBINED CLASSIFIER
    train_data = os.path.join(train_output_dir, 'action_recognition_data', 'action_descriptors', 'train_combined')
    test_data = os.path.join(train_output_dir, 'action_recognition_data', 'action_descriptors', 'test')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'action_classifiers')
    calc_combined_action_classifiers(dst_bin_path, train_data, test_data, output_dir)
    clean_dir(output_dir)
    dst_dir = os.path.join(models_dir, os.path.basename(output_dir))
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(output_dir, dst_dir)

    # EXTRACT SINGLE HAND TRAIN ORIENTATION DESCRIPTORS
    train_videos_dir = os.path.join(train_data_dir, 'gestures', 'gesture_orientation', 'single_direction', 'train')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_descriptors', 'single_direction', 'train')
    extract_orientation_desc(dst_bin_path, train_videos_dir, output_dir)

    # EXTRACT SINGLE HAND TEST ORIENTATION DESCRIPTORS
    train_videos_dir = os.path.join(train_data_dir, 'gestures', 'gesture_orientation', 'single_direction', 'test')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_descriptors', 'single_direction', 'test')
    extract_orientation_desc(dst_bin_path, train_videos_dir, output_dir)
    clean_dir(output_dir)

    # TRAIN SINGLE HAND ORIENTATION ESTIMATOR
    train_data = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_descriptors', 'single_direction', 'train')
    test_data = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_descriptors', 'single_direction', 'test')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_estimator', 'single_direction')
    calc_orientation_estimator(dst_bin_path, train_data, test_data, output_dir)
    clean_dir(output_dir)
    
    # EXTRACT DOUBLE HAND TRAIN ORIENTATION DESCRIPTORS
    train_videos_dir = os.path.join(train_data_dir, 'gestures', 'gesture_orientation', 'double_direction', 'train')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_descriptors', 'double_direction', 'train')
    extract_orientation_desc(dst_bin_path, train_videos_dir, output_dir)

    # EXTRACT DOUBLE HAND TEST ORIENTATION DESCRIPTORS
    train_videos_dir = os.path.join(train_data_dir, 'gestures', 'gesture_orientation', 'double_direction', 'test')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_descriptors', 'double_direction', 'test')
    extract_orientation_desc(dst_bin_path, train_videos_dir, output_dir)
    clean_dir(output_dir)

    # TRAIN DOUBLE HAND ORIENTATION ESTIMATOR
    train_data = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_descriptors', 'double_direction', 'train')
    test_data = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_descriptors', 'double_direction', 'test')
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_estimator', 'double_direction')
    calc_orientation_estimator(dst_bin_path, train_data, test_data, output_dir)
    clean_dir(output_dir)
    
    output_dir = os.path.join(train_output_dir, 'action_recognition_data', 'orientation_estimator')
    dst_dir = os.path.join(models_dir, os.path.basename(output_dir))
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)    
    shutil.copytree(output_dir, dst_dir)

