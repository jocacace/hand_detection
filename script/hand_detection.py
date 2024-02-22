#!/usr/bin/env python3

# Mediapipe imports

import cv2
import mediapipe as mp

# Image processing
from cv_bridge import CvBridge

# ROS
import rospy
## Message definitions
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from human_detection_msgs.msg import Skeleton2d


from std_msgs.msg import Bool
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

class HandDetection():
    def __init__(self ):
       rospy.init_node('HandDetection_node', anonymous=True)
       topic_rgbImg = "/xtion/rgb/image_raw"
       self.pub_targetStatus = rospy.Publisher("/hand/detected", Bool, queue_size=10)  
       self.msg_rgbImg  = None 
       self.newRgbImg = False   
       self.sub_rgbImg = rospy.Subscriber(topic_rgbImg, Image, self.callback_rgbImg)
       self.cvBridge = CvBridge()
       self.ProcessImg()
       
    
    def callback_rgbImg(self, msg):
       self.msg_rgbImg = msg
       self.newRgbImg = True


    def ProcessImg(self):
        msg_targetStatus  = Bool()    


        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1,  min_tracking_confidence=0.8)
        detector = vision.HandLandmarker.create_from_options(options)
        loopRate = rospy.Rate(10)

        while rospy.is_shutdown() == False:
            if self.newRgbImg == True: 
                cvImg = self.cvBridge.imgmsg_to_cv2(self.msg_rgbImg, "bgr8")
                #cv2.imshow("window_name", cvImg) 
                #cv2.waitKey(0) 
                cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cvImg)

                detection_result = detector.detect(mp_image)
                
                hand_landmarks_list = detection_result.hand_landmarks
                handedness_list = detection_result.handedness
                if( len( handedness_list ) > 0 ):
                    handedness = handedness_list[0]
                    if( handedness[0].score > 0.7 ):
                        
                        msg_targetStatus = True
                    else:
                        msg_targetStatus = False
                else:
                   msg_targetStatus = False
            else:
                msg_targetStatus = False
                   
            self.pub_targetStatus.publish(msg_targetStatus)
                 
         
            loopRate.sleep()
                
if __name__ == "__main__":
    HandDetection()
    
    
