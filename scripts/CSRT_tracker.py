#! /usr/bin/env python2

# Imports
import numpy as np
import cv2 as cv2

# ROS related imports
import rospy
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# Define middle and focals
C_MID = (640, 360)
FOCAL_LENGTH = [1068.0, 1072.0]


# Define the target tracker class
class Target_tracker():
    def __init__(self):
        
        cam_info = rospy.wait_for_message('/usb_cam/camera_info', CameraInfo)
        FOCAL_LENGTH[0] = cam_info.K[0]
        FOCAL_LENGTH[1] = cam_info.K[4]
        print(FOCAL_LENGTH)
        self.hoz_fov = np.arctan(C_MID[0] / FOCAL_LENGTH[0])

        # For converting to openCV
        self.bridge = CvBridge()

        # Publisher
        self.target_pub = rospy.Publisher('/target', Twist, queue_size=1)
        self.distance_error_pub = rospy.Publisher('/distance_error', Point, queue_size=1)

        # Subscriber
        self.image_sub = rospy.Subscriber("/camera/image_decompressed",Image,self.newFrameCallback)
        #self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.read_frame)
        self.gui_target_sub = rospy.Subscriber("/gui_target", Point, self.guiTragetCallback)
        self.distance_sub = rospy.Subscriber("/dtu_controller/current_frame_pose", Twist, self.positionCallback)

        # Class variables
        self.new_target = (None, None)
        self.box_size = (None, None)
        self.frame = None
        self.no_change = 0
        self.distance_to_wall = None
        self.wall_angle = None
        self.distance_error = Point()
        self.tracker = None
        self.initBB = None
        self.bbSize = (30,30)

        print('Target tracking initialised')

    # Callback for getting the image frame
    def newFrameCallback(self, data):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    # Callback for reading the target and initilize the CSRT Tracker and a bounding box around the target
    def guiTragetCallback(self, data):
        print("Target from GUI read")
        
        self.new_target = (data.x, data.y)
        
        self.initBB = (int(data.x-self.bbSize[0]), int(data.y-self.bbSize[1]), self.bbSize[0]*2, self.bbSize[1]*2)
        
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(self.frame, self.initBB)

    # Update the CSRT tracker
    def updateTracker(self): 
        if self.initBB != None and self.tracker != None:
            (success, box) = self.tracker.update(self.frame)

            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                self.new_target = (x+w/2, y+h/2)
                self.box_size = (w, h)
                self.no_change = 1
            else:
                self.no_change = 0

    # Convert target point to position errors
    def calculateError(self):
        if self.distance_to_wall != None and self.new_target[0] != None:
            self.distance_error.x = self.distance_to_wall
            # y_offset = (C_MID[0]) * np.sin(self.wall_angle)
            # self.distance_error.y = -((self.new_target[0]-y_offset) - C_MID[0])/FOCAL_LENGTH[0] * self.distance_to_wall
            # self.distance_error.z = -(self.new_target[1] - C_MID[1])/FOCAL_LENGTH[1] * self.distance_to_wall
            # print("{} {}".format(self.new_target[0], self.new_target[1]))
            # print("{} {}".format(self.distance_error.y, self.distance_error.z))
            theta = self.wall_angle - (self.hoz_fov) * float(self.new_target[0] - C_MID[0])/float(C_MID[0])

            self.distance_error.y = self.distance_to_wall * np.sin(theta) # - self.distance_to_wall * np.sin(self.wall_angle)
            self.distance_error.z = -(self.new_target[1] - C_MID[1])/FOCAL_LENGTH[1] * self.distance_to_wall

            #print(theta * 180.0/np.pi)

    # Publish the target for the GUI to read for visualization
    def publishTarget(self):
        if self.new_target[0] != None and self.box_size[0] != None:
            p = Twist()
            p.linear.x = float(self.new_target[0]) # Current target position in image
            p.linear.y = float(self.new_target[1]) # Current target position in image
            p.linear.z = self.no_change            # Tell if target was succesfully tracked
            
            p.angular.x = float(self.box_size[0]) # Send the current size of the bounding box
            p.angular.y = float(self.box_size[1]) # Send the current size of the bounding box
            p.angular.z = 1 # Tell if bb is used

            self.target_pub.publish(p)
        
            # If distance errors are calculateable and tracking is succesfull publish
            if self.distance_to_wall != None and self.no_change:
                self.distance_error_pub.publish(self.distance_error)

    # Callback for reading the current distance to wall and yaw
    def positionCallback(self, data):
        self.distance_to_wall = data.linear.x
        self.wall_angle = data.angular.z

    def run(self):
        while not rospy.is_shutdown():

            if self.new_target[0] != None:
                self.updateTracker()
                self.calculateError()
                self.publishTarget()
            else:
                rospy.Rate(30).sleep()  

if __name__ == '__main__':
    rospy.init_node("target_node", anonymous=True)
    my_tracker = Target_tracker()
    my_tracker.run()




