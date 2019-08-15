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
from PIL import Image as PILImage

# for YOLO
import os
import sys
sys.path.append("../../../keras-yolo3")
from yolo import YOLO

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.bridge = CvBridge()
        self.frameCounter = 0
        self.frontImage = None

        rospy.loginfo("Creating YOLO object.")
        self.yolo = YOLO()
        rospy.loginfo("Create success YOLO object")

        sub7 = rospy.Subscriber('/image_raw', Image, self.image2_cb)

        self.loop()
        #rospy.spin()

    def loop(self):
        rospy.loginfo("Proc loop")
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if(self.frontImage):
                det_image, results = self.yolo.detect_image(self.frontImage)
                rospy.loginfo(det_image)

                for ret in results:
                    print("Result:"+ret[0])
                det_image.save('/home/student/Desktop/workspace/out{:0>8}.png'.format(self.frameCounter))
                #cv2.imwrite('/home/student/Desktop/workspace/out{:0>8}.png'.format(self.frameCounter),det_image )
                self.frameCounter += 1
                
            rate.sleep()


    def image2_cb(self, img):

        cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")

        #rospy.loginfo("Image data recieved")
        # rospy.loginfo("Original:{}".format(cv_image.shape))
        cv_image = cv2.resize(cv_image, (416,416))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(cv_image)
        pil_image = pil_image.convert('RGB')

        self.frontImage = pil_image

        # rospy.loginfo("Resized:{}
        # ".format(cv_image.shape))
        # pil_image.save("PIL.bmp")

        # src = PILImage.open("PIL.bmp")

        # print(str(self.frameCounter))
        # image = PILImage.open("/home/student/Desktop/bag_file/keras-yolo3/test.bmp")
        # det_image = self.yolo.detect_image(image)

        # if(cv2.imwrite('/home/student/Desktop/workspace/test{:0>8}.png'.format(self.frameCounter),det_image )):
        #     rospy.logwarn("image OK")
        #     self.frameCounter += 1
        # else:
        #     rospy.logerr("image NG")



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
