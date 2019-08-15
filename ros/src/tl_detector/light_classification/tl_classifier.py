from styx_msgs.msg import TrafficLight
import rospy
import sys
import os
sys.path.append("../../../keras-yolo3")
from yolo import YOLO
import cv2
from PIL import Image as PILImage
import numpy as np


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.frameCounter = 0

        rospy.loginfo("Creating YOLO object.")
        self.yolo = YOLO()
        rospy.loginfo("Create success YOLO object")

    def get_classification(self, cv_image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        tl_state = TrafficLight.GREEN

        cv_image = cv2.resize(cv_image, (416,416))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        origin_image = PILImage.fromarray(cv_image)
        origin_image = origin_image.convert('RGB')

        target_image = origin_image.copy()

        det_image, results = self.yolo.detect_image(target_image)
        det_image.save('/home/student/Desktop/workspace/Det{:0>4}.bmp'.format(
            self.frameCounter))
        #rospy.loginfo(det_image)

        for ret in results:
            predicted_class = ret[0]
            score = ret[1]
            left = ret[2]
            top = ret[3]
            right = ret[4]
            bottom = ret[5]
            #print("Result:"+ret[0])

            # Crop traffic light area
            tl_image = None
            if(predicted_class == 'traffic light'):
                #print('Traffic light:found')
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

                cv2.imwrite('/home/student/Desktop/workspace/crop_h_{:0>8}.bmp'.format(self.frameCounter),h_image)
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

                tl_state = TrafficLight.GREEN
                print("State:GREEN({})  ({:.0f},{:.0f},{:.0f})".format(tl_state,green_area_h_mean,yellow_area_h_mean,red_area_h_mean))
                if ((red_area_h_mean > yellow_area_h_mean)&(red_area_h_mean > green_area_h_mean)):
                    tl_state = TrafficLight.RED
                    print("State:RED({})  ({:.0f},{:.0f},{:.0f})".format(tl_state,green_area_h_mean,yellow_area_h_mean,red_area_h_mean))

                #cv2.imwrite('/home/student/Desktop/workspace/redArea{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,red_area_h_mean),red_area_h_image)
                #cv2.imwrite('/home/student/Desktop/workspace/yellowArea{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,yellow_area_h_mean),yellow_area_h_image)
                #cv2.imwrite('/home/student/Desktop/workspace/greenArea{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,green_area_h_mean),green_area_h_image)
                tl_image.save('/home/student/Desktop/workspace/crop{:}_{}_{:.0f}.bmp'.format(self.frameCounter, tl_state,red_area_h_mean))

                self.frameCounter += 1

        #TODO implement light color prediction
        return tl_state
