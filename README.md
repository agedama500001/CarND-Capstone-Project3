# Project: Programming a Real Self-Driving Car
This project was created by an individual.

## Project overview
---
### Project ROS architecture
* The ROS architecture used in the project is shown below.   
![ROS architecture](./imgs/final-project-ros-graph-v2.png "ROS architecture")

### Modifications to the base project repository
* I modified the following nodes according to the walkthrough.  
(1) Waypoint Updater Node  
　I made a modification to publish the waypoint that the loaded map dataset vehicle should run as / finalwaypoint.  
(2) DBW None  
　Modified twist_controller.py, yaw_controller.py, and dbw_node.py to control the vehicle according to the / twist command.  
(3) Traffic Light Detection Node  
　Modified to generate /traffic_waypoints that the vehicle should run according to the signal status.(tl_detector.py)　We implemented an algorithm that detects the signal from the front image and identifies the color state of the signal.(tl_classfier.py:See next paragraph)

### Traffic Light detection and classfication
* Traffic Light detection  
 I used [keras-yolo3](https://github.com/qqwweee/keras-yolo3) provided by qqwweee for signal detection. An example of the detection result is shown below.  
 ![Traffic light detection image (On simulator)](./imgs/Det0039.bmp "Traffic light detection image (On simulator)")  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fig.1  Traffic light detection image (On simulator)
![Traffic light detection image (On realworld)](./imgs/DetDebug0024.png "Traffic light detection image (On realworld)")  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fig.2  Traffic light detection image (On realworld)
  
* Traffic Light classfication  
　The detected signal area was cropped and converted to the HSV image space to determine which color of the signal was issued. As shown in the figure, when the signal is red, the H value is lower than that of the yellow or green region. 
Otherwise, it was judged green.  
  ![Traffic light classification result image](./imgs/writeup.png "Traffic light classification result image")   
  &nbsp;&nbsp;&nbsp;&nbsp;fig.3 raffic light classification result image


---
### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
