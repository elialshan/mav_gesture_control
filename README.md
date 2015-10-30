# mav_gesture_control
### Installation (sufficient for the demo)

1. Clone the MAV gesture control repository
  ```Shell
  git clone https://github.com/elialshan/mav_gesture_control.git
  ```
 We'll call the directory that you cloned the repository into `MGC_ROOT`


2. Compiling the code
	In order to compile the mav gesture control code, you need to have the following libraries installed in your system:
	* ffmpeg library (tested with ffmpeg 0.6.5)	
	* OpenCV library (tested with OpenCV-2.4.9) with ffmpeg, tbb, OpenMP and CUDA support.
	* boost library (tested with boost 1.55.0)
	
	Edit Makefile.config to set the path to your libraries folder.
		
    ```Shell
    cd $MGC_ROOT
    make
    ```


4. (Optional) Generating doxygen documentation
    ```Shell
    cd $MGC_ROOT
    make docs
    ```

5. Download pre-computed models
    ```Shell
    cd $MGC_ROOT
    ./scripts/fetch_demo_models.sh
    ```

    This will populate the `$MCG_ROOT` with `models`.    

6. Download demo videos
    ```Shell
    cd $MGC_ROOT
    ./scripts/fetch_demo_data.sh
    ```

    This will populate the `$MGC_ROOT/data` folder with `mav_videos`.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

```Shell
cd $MGC_ROOT
./scripts/run_demo.sh
```

The demo performs operator and target detection, gesture recognition, gesture orientation estimation and visual map generation on one of the videos supplied with this package.
The demo presents two windows:

1. Gesture recognition window:

    * The operators are marked by green bounding boxes and the detected targets are marked by red bounding boxes.
    * By default one operator is analyzed for gesture recognition (this option can be modified in run_demo.sh script),
      the processed operator will have a letter in the center of his bounding box. 
      The letter indicates the type of the detected action:
        - B - background
        - D - direction
        - LR - left hand rotation
        - RR - right hand rotation
    * After a gesture is detected the algorithm will estimate the gesture orientation.
      The detected orientation will be marked by a green arrow.
2. Visual map window:

    The window presents real time construction of the visual map.

To try a different input video modify .scripts/run_demo.sh

### Beyond the demo: training and testing models
1. Download the training data.

	```Shell
	./scripts/fetch_train_data.sh
	```
	
	This will populate the `$MGC_ROOT/data` folder with `detection` and `gestures`.    

2. Train new models.

	```Shell
	./scripts/train_models.sh
	```
	
    The script will:
	
        a. Train an operator classifier used by the detector. And evaluate the detection performance.
        b. Extract and store dense trajectories from all available gesture examples.
        c. Compute BoF codebooks.
        d. Extract gesture descriptors.
        e. Train and test base and combined gesture classifiers.
        f. Extract gesture orientation descriptors.
        g. Train orientation estimator and evaluate its performance.

	The new models will be copied to `$MGC_ROOT/new_models`. In order to use those models replace the existing models folder with the newly generated one.
	


### LICENSE CONDITIONS ###

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.


