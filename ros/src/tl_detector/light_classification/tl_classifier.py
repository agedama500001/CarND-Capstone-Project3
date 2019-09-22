from styx_msgs.msg import TrafficLight
import rospy
from sensor_msgs.msg import Image
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

        with open("/home/student/Desktop/workspace/debug.csv", "w") as f:
            f.writelines("frame,subframe,result,green_area_h_mean,yellow_area_h_mean,red_area_h_mean,green_area_r_mean,yellow_area_r_mean,red_area_r_mean,result2\n")



    def get_classification(self, cv_image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        tl_state = TrafficLight.GREEN
        tl_state2 = TrafficLight.GREEN

        cv_image = cv2.resize(cv_image, (416,416))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        origin_image = PILImage.fromarray(cv_image)
        origin_image = origin_image.convert('RGB')

        target_image = origin_image.copy()

        det_image, results = self.yolo.detect_image(target_image)
        #det_image.save('/home/student/Desktop/workspace/Det{:0>4}.bmp'.format(self.frameCounter))
        #rospy.loginfo(det_image)


        subFrCounter = 0
        clf_results = None
        finaljudge = []
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
            curr_judge = None
            if(tl_image):

                # Convert color space
                tl_cv_image = np.asarray(tl_image.copy().convert("HSV"))
                h_image = tl_cv_image[:, :, 0]
                s_image = tl_cv_image[:, : ,1]
                v_image = tl_cv_image[:, :, 2]

                tl_cv_imageRGB = np.asarray(tl_image.copy().convert("RGB"))
                r_image = tl_cv_imageRGB[:, :, 0]
                g_image = tl_cv_imageRGB[:, :, 1]
                b_image = tl_cv_imageRGB[:, :, 2]

                #cv2.imwrite('/home/student/Desktop/workspace/crop_h_{:0>8}.bmp'.format(self.frameCounter),h_image)
                #cv2.imwrite('/home/student/Desktop/workspace/crop_s_{:0>8}.bmp'.format(self.frameCounter),s_image)
                #cv2.imwrite('/home/student/Desktop/workspace/crop_v_{:0>8}.bmp'.format(self.frameCounter),v_image)

                ############ method 1 : mean ###############
                tl_top = 0
                tl_bottom = tl_height = h_image.shape[0]

                # crop red area
                area_height = tl_height // 3
                red_top = tl_top
                red_bottom = area_height
                red_area_h_image = h_image[red_top:red_bottom, :]
                red_area_r_image = r_image[red_top:red_bottom, :]
                #print("h_image:{}".format(h_image.shape))
                #print("red_area_h_image:{}".format(red_area_h_image.shape))
                #print("red area top:{},bottom:{}  area height:{}".format(red_top,red_bottom,area_height))

                # crop yellow area
                yellow_top = red_bottom + 1
                yellow_bottom = yellow_top + area_height
                yellow_area_h_image = h_image[yellow_top:yellow_bottom, :]
                yellow_area_r_image = r_image[yellow_top:yellow_bottom, :]

                # crop green area
                green_top = yellow_bottom + 1
                green_bottom = green_top + area_height
                green_area_h_image = h_image[green_top:green_bottom, :]
                green_area_r_image = h_image[green_top:green_bottom, :]

                red_area_h_mean = red_area_h_image.mean()
                yellow_area_h_mean = yellow_area_h_image.mean()
                green_area_h_mean = green_area_h_image.mean()

                red_area_r_mean = red_area_r_image.mean()
                yellow_area_r_mean = yellow_area_r_image.mean()
                green_area_r_mean = green_area_r_image.mean()

                tl_state = TrafficLight.GREEN
                tl_state_str = "GREEN"
                if ((red_area_h_mean > yellow_area_h_mean)&(red_area_h_mean > green_area_h_mean)):
                    tl_state = TrafficLight.RED
                    #print("State:RED({})  ({:.0f},{:.0f},{:.0f})".format(tl_state,green_area_h_mean,yellow_area_h_mean,red_area_h_mean))
                    tl_state_str = "RED"
                #else:
                    #print("State:GREEN({})  ({:.0f},{:.0f},{:.0f})".format(tl_state,green_area_h_mean,yellow_area_h_mean,red_area_h_mean))

                ############ method 2 : histogram ###############
                tl_cv_imageRGB32 = cv2.resize(tl_cv_imageRGB,(32,32))

                r_hist = np.histogram(tl_cv_imageRGB32[5:10,14:19,0], bins=32, range=(0, 255))
                g_hist = np.histogram(tl_cv_imageRGB32[15:18,12:17,1], bins=32, range=(0, 255))
                b_hist = np.histogram(tl_cv_imageRGB32[25:28,12:17,2], bins=32, range=(0, 255))

                bin_edges = r_hist[1]
                bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

                fullest_r = np.argmax(r_hist[0])
                fullest_g = np.argmax(g_hist[0])
                fullest_b = np.argmax(b_hist[0])
                feature = [fullest_r, fullest_g ,fullest_b]

                predColor = np.argmax(feature)

                tl_state_str2 = "GREEN"
                tl_state2 = TrafficLight.GREEN
                if(predColor==0 or predColor==1):
                    tl_state_str2 = "RED"
                    tl_state2 = TrafficLight.RED

                ############ for multiple tl detection ###############
                curr_judge = tl_state2
                if(subFrCounter==0):
                    clf_results = [curr_judge]
                else:
                    clf_results.append(curr_judge)

                det_image.save('/home/student/Desktop/workspace/Det{:0>4}_{}.bmp'.format(self.frameCounter,tl_state_str2))

                with open("/home/student/Desktop/workspace/debug.csv", "a") as f:
                    f.writelines("{},{},{},{},{},{},{},{},{},{}\n".format(self.frameCounter,
                                                                    subFrCounter,
                                                                    tl_state_str,
                                                                    green_area_h_mean,
                                                                    yellow_area_h_mean,
                                                                    red_area_h_mean,
                                                                    green_area_r_mean,
                                                                    yellow_area_r_mean,
                                                                    red_area_r_mean,
                                                                    tl_state_str2))

                #cv2.imwrite('/home/student/Desktop/workspace/redArea{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,red_area_h_mean),red_area_h_image)
                #cv2.imwrite('/home/student/Desktop/workspace/yellowArea{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,yellow_area_h_mean),yellow_area_h_image)
                #cv2.imwrite('/home/student/Desktop/workspace/greenArea{:0>8}_{}_{:0>3}.bmp'.format(self.frameCounter, tl_state,green_area_h_mean),green_area_h_image)
                #tl_image.save('/home/student/Desktop/workspace/crop{:}_Obj{}_{}.bmp'.format(self.frameCounter, subFrCounter,tl_state_str))

                subFrCounter += 1
        ######### Majority vote ###########
        if (subFrCounter == 1):
            finaljudge = clf_results
            print("final:RED")
        elif (subFrCounter > 1):
            red_count = clf_results.count(TrafficLight.RED)
            green_count = clf_results.count(TrafficLight.GREEN)
            if(red_count >= green_count):
                finaljudge = TrafficLight.RED
                print("Majority vote! red:{},green:{},final:RED".format(red_count,green_count))
            else:
                finaljudge = TrafficLight.GREEN
                print("Majority vote! red:{},green:{},final:GREEN".format(red_count,green_count))
        self.frameCounter += 1

        #TODO implement light color prediction
        return finaljudge

    def pil2cv(self, image):

        new_image = np.array(image, dtype=np.uint8)
        if new_image.ndim == 2:
            pass
        elif new_image.shape[2] == 3:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        elif new_image.shape[2] == 4:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        return new_image
