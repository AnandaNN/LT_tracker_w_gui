#!/usr/bin/env python2
import cv2
from sensor_msgs.msg import Image as SensorImage
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError
import rospy


"""
Program for subscribing to the image from the drone, and adding the target point to the image.
The image is then locally saved for the gui to use

The program is subscribing to the drone-image with a frequency of 10 Hz, and re-uploads the image everytime an image from the drone is received 
"""

class DroneCameraSub:
    def __init__(self):
        self.loop_rate = rospy.Rate(10)

        self.img = None
        self.circle_center = [None, None]
        
        rospy.Subscriber("bebop/image_raw", SensorImage, self.callback)
        rospy.Subscriber("target", Pose, self.draw_target)

    def draw_target(self,data):
        self.circle_center = [data.position.x, data.position.y]
        
    def callback(self, data):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(data, "rgb8")
            self.img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            if self.circle_center[0] != None:
                cv2.circle(self.img, (int(self.circle_center[0]), int(self.circle_center[1])), 3, (0, 255, 0), 10)
            cv2.imwrite('src/marker_detection/camera_image.jpeg', self.img)
        except CvBridgeError as e:
            print(e)  

    def run(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()


if __name__ == '__main__':
    rospy.init_node("drone_cam_node", anonymous=True)
    drone = DroneCameraSub()
    drone.run()
    
