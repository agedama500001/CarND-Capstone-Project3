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
import numpy as np

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
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if(self.frontImage):
                # Save original image in the buffer because it will be updated.
                origin_image = self.frontImage.copy()
                target_image = origin_image.copy()

                det_image, results = self.yolo.detect_image(target_image)
                det_image.save('/home/student/Desktop/workspace/DetDebug{:0>4}.png'.format(self.frameCounter))
                #rospy.loginfo(det_image)

                for ret in results:
                    predicted_class = ret[0]
                    score = ret[1]
                    left = ret[2]
                    top = ret[3]
                    right = ret[4]
                    bottom = ret[5]
                    print("Result:"+ret[0])

                    # Crop traffic light area
                    tl_image = None
                    if(predicted_class == 'traffic light'):
                        print('Traffic light:found')
                        tl_image = origin_image.crop((left, top, right, bottom))
                        #tl_image.save('/home/student/Desktop/workspace/crop{:0>8}.png'.format(self.frameCounter))

                    # Judge traffic light color
                    isRedSign = False
                    if(tl_image):

                        # Convert color space
                        tl_cv_image = np.asarray(tl_image.copy().convert("HSV"))
                        h_image = tl_cv_image[:, :, 0]
                        s_image = tl_cv_image[:, : ,1]
                        v_image = tl_cv_image[:, :, 2]

                        cv2.imwrite('/home/student/Desktop/workspace/crop_h_Debug_{:0>4}.bmp'.format(self.frameCounter),h_image)
                        #cv2.imwrite('/home/student/Desktop/workspace/crop_s_{:0>8}.bmp'.format(self.frameCounter),s_image)
                        #cv2.imwrite('/home/student/Desktop/workspace/crop_v_{:0>8}.bmp'.format(self.frameCounter),v_image)

                        tl_top = 0
                        tl_bottom = tl_height = h_image.shape[0]

                        # crop red area
                        area_height = tl_height // 3
                        red_top = tl_top
                        red_bottom = area_height
                        red_area_h_image = h_image[red_top:red_bottom, :]
                        #print("h_image:{}".format(h_image.shape))
                        #print("red_area_h_image:{}".format(red_area_h_image.shape))
                        #print("red area top:{},bottom:{}  area height:{}".format(red_top,red_bottom,area_height))

                        # crop yellow area
                        yellow_top = red_bottom + 1
                        yellow_bottom = yellow_top + area_height
                        yellow_area_h_image = h_image[yellow_top:yellow_bottom, :]

                        # crop green area
                        green_top = yellow_bottom + 1
                        green_bottom = green_top + area_height
                        green_area_h_image = h_image[green_top:green_bottom, :]

                        red_area_h_mean = red_area_h_image.mean()
                        yellow_area_h_mean = yellow_area_h_image.mean()
                        green_area_h_mean = green_area_h_image.mean()

                        tl_state = "GREEN"
                        if ((red_area_h_mean < yellow_area_h_mean)&((red_area_h_mean < yellow_area_h_mean))):
                            tl_state = "RED"
                        print("State:{}  {}".format(tl_state,red_area_h_mean))

                        #cv2.imwrite('/home/student/Desktop/workspace/redArea{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,red_area_h_mean),red_area_h_image)
                        #cv2.imwrite('/home/student/Desktop/workspace/yellowArea{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,yellow_area_h_mean),yellow_area_h_image)
                        #cv2.imwrite('/home/student/Desktop/workspace/greenArea{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,green_area_h_mean),green_area_h_image)
                        #tl_image.save('/home/student/Desktop/workspace/crop{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,red_area_h_mean))


                        #area_red = tl_image.crop()


                #target_image.save('/home/student/Desktop/workspace/out{:0>8}.png'.format(self.frameCounter))
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
