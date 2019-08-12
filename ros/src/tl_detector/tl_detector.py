#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')


        sub7 = rospy.Subscriber('/image_raw', Image, self.image2_cb)

        self.bridge = CvBridge()
        self.frameCounter = 0

        rospy.spin()



    def image2_cb(self, img):

        cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        if(cv2.imwrite('/home/student/Desktop/workspace/test{:0>8}.png'.format(self.frameCounter),cv_image )):
        #if(cv2.imwrite('/home/student/Desktop/workspace/test{}.png'.format(self.frameCounter),cv_image )):
        #if(cv2.imwrite('/home/student/Desktop/workspace/test.png',cv_image )):
            rospy.logwarn("image OK")
            self.frameCounter += 1
        else:
            rospy.logerr("image NG")



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
