#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('marker_detection')
import sys
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
"""
Creates a video from the bebop/image_raw topic at 30 fps and with dimensions 856 by 480 pixel.
"""


video_name = "exam_01.avi"

class Record_video:

  def __init__(self):

    #cycle rate of the node (Hz)
    self.loop_rate = rospy.Rate(30)
    #camera frame
    self.frame = None
    self.bridge = CvBridge()

    self.circle_center = (None, None)
    self.distance = 0
    self.draw_target = False

    self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    self.out = cv2.VideoWriter(video_name,self.fourcc, 30.0, (856,480))

    #subscriber
    self.image_sub = rospy.Subscriber("bebop/image_raw",Image,self.callback)
    self.target_sub = rospy.Subscriber("target", Pose, self.update_target )
  def update_target(self, data):
    self.circle_center = (data.position.x, data.position.y)
    self.distance = data.position.z
    if self.draw_target == False:
      self.draw_target = True

  def callback(self,data):
    try:
      self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
      text = "distance: {}".format(self.distance)
      #cv2.putText(self.frame, text, (50,50), 0, 1, (255, 0, 0), 2, cv2.LINE_4)
      if self.draw_target:
        cv2.circle(self.frame, (int(self.circle_center[0]), int(self.circle_center[1])), 3, (0, 255, 0), 2 )
    except CvBridgeError as e:
      print(e)
    self.out.write(self.frame)
    cv2.imshow("Image window",self.frame)
    cv2.waitKey(3)
 
  def start(self):
    while not rospy.is_shutdown():
      self.loop_rate.sleep()
    self.out.release()
  
if __name__ == '__main__':
    rospy.init_node("record_video", anonymous=True)
    recorder = Record_video()
    recorder.start()