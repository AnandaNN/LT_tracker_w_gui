#! /usr/bin/env python2
import numpy as np
import cv2 as cv2
import math
import roslib
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg  import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

"""
From the target point given by the user interface, this program computes the next target point

The constant C_MID denotes the center of the image at 1/2 width and 1/4 height
The file_name is the output of the .csv file containing the total number of mathces avaiable on the frame, the number of used matches for the frame, 
as well as the estimated distance to the wall 
"""

# C_MID = (428, 120)
C_MID = (320, 240)

file_name = 'test.csv'

class Target_tracker():
    def __init__(self):
        self.loop_rate = rospy.Rate(10)

        #for converting to openCV
        self.bridge = CvBridge()

        #for keeping track of the target
        self.previous_target = (None, None)
        self.new_target = (None, None)

        #the different frames
        self.frame = None
        self.previous_frame = None
        self.undistorted_frame = None

        self.no_previous_frame = True

        #for the feature detector
        self.keypoints_previous_frame = None
        self.descriptors_previous_frame = None
        self.laser_pos = None
        self.laser_found = False

        self.distance_to_wall = None
        self.pix_distance = 0.0
        self.pix_distance_prev = None

        self.first_flag = True

        # for csv
        self.matches_array = []
        self.matches_used_array = [] 
        self.distance_array = []
        
        self.pitch = 0

        #camera parameters
        self.camera_distortion = np.array([-0.000316, 0.016703, -0.007112, 0.000509, 0.000000])
        self.camera_matrix = np.array([[520.235734, 0.000000, 421.473550],
                         [0.000000, 512.906134, 218.998044],
                         [0.000000, 0.000000, 1.000000]])

        # publisher
        self.target_pub = rospy.Publisher('target', Pose, queue_size=1)
        self.distance_pub = rospy.Publisher('distance', Pose, queue_size=1)
        self.target_tracking_enable = rospy.Publisher('target_tracking_enable', Bool, queue_size= 1)

        # subscriber
        self.image_sub = rospy.Subscriber("webcam/image_raw",Image,self.read_frame)
        self.gui_target_sub = rospy.Subscriber("gui_target", Pose, self.read_gui_target)
        self.abort_sub = rospy.Subscriber("abort", Bool, self.abort)
        self.pitch_sub = rospy.Subscriber('/bebop/odom',Odometry,self.odometryCb)

        print('Target tracking initialised')

    def odometryCb(self, msg):
        ## getting the pitch of the drone to calculate the distance to the wall, even though the drone is tilted
        euler = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.pitch = euler[1]

    def estimate_distance(self):
        ## estimating distance based on the centers of the two laser dots

       center_of_laser = self.find_laser()
       if(len(center_of_laser) == 2):
            distance_x = abs(center_of_laser[1][0] - center_of_laser[0][0])
            distance_y = abs(center_of_laser[1][1] - center_of_laser[0][1])
            self.pix_distance = math.sqrt((distance_x ** 2) + (distance_y ** 2))
            self.laser_found =True
            self.pix_distance = self.pix_distance * np.cos(self.pitch)

            # converting it to meters from pixels, The equation is found from experiment
            self.distance_to_wall = 20.381 * (self.pix_distance ** -0.876)
            
            p = Pose()
            p.position.x = self.distance_to_wall
            self.distance_pub.publish(p)
       else:
            self.laser_found = False 
       self.laser_pos = center_of_laser

    def read_frame(self, data):
        ## converting image from drone to opencv format
        try:
            self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.first_flag = False
            # for when it runs for the first time
            if (self.no_previous_frame):
                self.previous_frame = self.frame[:]
                self.no_previous_frame = False

        except CvBridgeError as e:
            print(e)

    def read_gui_target(self, data):
        print("Target from GUI read")
        if(self.new_target[0] == None):
            self.new_target = (data.position.x, data.position.y)
        if(self.previous_target[0] == None):
            self.previous_target = (data.position.x, data.position.y)

    def threshold_image(self):
        ## thresholding the image to separate the laser dots from the rest. Values of 'lower' and 'upper' should be modifed to fit the lighting conditions
        frame = self.frame.copy()

        lower = np.array([215, 200, 200])
        upper = np.array([255, 255, 255])

        mask = cv2.inRange(frame, lower, upper)
        frame_laser = cv2.bitwise_and(frame, frame, mask=mask)

        return mask, frame_laser

    def find_laser(self):
        #isolate the lasers in the image
        mask, frame_laser = self.threshold_image()
        
        center_of_laser = []
        
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=lambda x: cv2.boundingRect(x)[0])

        max_radius = 0
        for c in contours:
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))

            if radius >= 0.5 and radius <= 25:
                if radius > max_radius:
                    max_radius = radius
                cv2.circle(frame_laser, center, int(radius), (0, 255, 0), 2)
                center_of_laser.append(center)
        self.laser_radius = max_radius*3 + 5 

        return center_of_laser

    def display_frames(self):
        ## method not in use
        ## shows the previous and current frame side by side with the target visible

        cols1 = self.previous_frame.shape[1]

        combined_frames = np.hstack((self.previous_frame, self.frame))

        cv2.circle(combined_frames, (int(self.previous_target[0]), int(self.previous_target[1])), 3, (255, 0, 0), 10)
        cv2.circle(combined_frames, (cols1 + int(self.new_target[0]), int(self.new_target[1])), 3, (255, 0, 0), 10)

        cv2.line(combined_frames, (int(self.previous_target[0]), int(self.previous_target[1])), (cols1 + int(self.new_target[0]), int(self.new_target[1])), (255, 0, 0), 3)
        if(self.laser_found == True):
            cv2.putText(combined_frames, "distance to wall{:10.4f} m".format(self.distance_to_wall), (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,cv2.LINE_AA)

        cv2.namedWindow('concrete wall', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('concrete wall', 900, 500)
        cv2.imshow('concrete wall', combined_frames)

    def find_matches(self):
        ## finds usable matches from previous and current frame, as well as update the csv file
        ## usable matches: excluding those within a radius of the laser dots and excluding those far away from target point
        orb = cv2.ORB_create()
        keypoints_frame, descriptors_frame = orb.detectAndCompute(self.undistorted_frame, None)

        if( self.keypoints_previous_frame == None ):
            self.keypoints_previous_frame = keypoints_frame
            self.descriptors_previous_frame = descriptors_frame

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.descriptors_previous_frame, descriptors_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        coordinates_best_matches_previous_frame = []
        coordinates_best_matches_frame = []

        number_of_matches_to_use = int(len(matches))

        radi = math.sqrt((self.previous_target[0]-C_MID[0])**2 + (self.previous_target[1]-C_MID[1])**2)
        factor = 100
        if radi < factor:
            radi = factor
        elif radi > 476 - factor:
            radi = 476 - factor

        for mat in matches[:number_of_matches_to_use]:
            laser_flag = False
            previous_frame_idx = mat.queryIdx
            frame_idx = mat.trainIdx

            (x1, y1) = self.keypoints_previous_frame[previous_frame_idx].pt
            (x2, y2) = keypoints_frame[frame_idx].pt
            if(self.laser_found == True):
                for i in range (len(self.laser_pos)):
                    if (x1 <= self.laser_pos[i][0] + self.laser_radius and x1 >= self.laser_pos[i][0] -self.laser_radius and y1 <= self.laser_pos[i][1] +self.laser_radius and y1 >= self.laser_pos[i][1] -self.laser_radius) or (x2 <= self.laser_pos[i][0] +self.laser_radius and x2 >= self.laser_pos[i][0] -self.laser_radius and y2 <= self.laser_pos[i][1] +self.laser_radius and y2 >= self.laser_pos[i][1] -self.laser_radius):
                        laser_flag = True
                        break
            if laser_flag != True and math.sqrt((x1-C_MID[0])**2 + (y1-C_MID[1])**2) >= radi -factor and math.sqrt((x1-C_MID[0])**2 + (y1-C_MID[1])**2) <= radi + factor:
                coordinates_best_matches_previous_frame.append((x1, y1))
                coordinates_best_matches_frame.append((x2, y2))
        self.matches_array.append(len(matches))
        self.matches_used_array.append(len(coordinates_best_matches_frame)) 
        self.distance_array.append(self.distance_to_wall)

        np.savetxt(file_name, np.transpose([np.array(self.matches_array), np.array(self.matches_used_array), np.array(self.distance_array)]), delimiter=',',fmt="%s") 
        self.keypoints_previous_frame = keypoints_frame
        self.descriptors_previous_frame = descriptors_frame

        return (coordinates_best_matches_previous_frame, coordinates_best_matches_frame)

    def find_new_target(self):
        ## computes current target point based of coordinates of matches 

        (coordinates_best_matches_previous_frame, coordinates_best_matches_frame) = self.find_matches()

        prev_dist_to_target = []
        for i in range(0,len(coordinates_best_matches_frame)):
            prev_dist_to_target.append( (coordinates_best_matches_previous_frame[i][0]-self.previous_target[0], coordinates_best_matches_previous_frame[i][1]-self.previous_target[1]) )

        new_target_x = []
        new_target_y = []
        error_margin = 20
        for i in range(0, len(prev_dist_to_target)-1):
            if prev_dist_to_target[i][0] != prev_dist_to_target[i+1][0] and prev_dist_to_target[i][1] != prev_dist_to_target[i+1][1]:
               
                a_x = (coordinates_best_matches_frame[i+1][0]-coordinates_best_matches_frame[i][0]) / (prev_dist_to_target[i][0]- prev_dist_to_target[i+1][0])
                a_y = (coordinates_best_matches_frame[i+1][1]-coordinates_best_matches_frame[i][1]) / (prev_dist_to_target[i][1]- prev_dist_to_target[i+1][1])

                potentiel_new_x = prev_dist_to_target[i][0]*a_x + coordinates_best_matches_frame[i][0]
                potentiel_new_y = prev_dist_to_target[i][1]*a_y + coordinates_best_matches_frame[i][1]
                
                if potentiel_new_x-self.previous_target[0] < error_margin and potentiel_new_x-self.previous_target[0] > -error_margin and potentiel_new_y-self.previous_target[1] < error_margin and potentiel_new_y-self.previous_target[1] > -error_margin:
                    new_target_x.append( potentiel_new_x )
                    new_target_y.append( potentiel_new_y )
                    
        if len(new_target_y) != 0 and len(new_target_x) != 0 :
            self.new_target = (sum(new_target_x)/len(new_target_x), sum(new_target_y)/len(new_target_y))
        else:
            self.new_target = self.previous_target
            print("old value used")
        
        self.previous_target = self.new_target

    def publish_new_target(self):
        if self.new_target[0] != None:
            print("sending target")
            p = Pose()
            p.position.x = float(self.new_target[0])
            p.position.y = float(self.new_target[1])
            if self.laser_found == True:
                p.position.z = self.distance_to_wall
            else:
                p.position.z = 0
            p.orientation.x = 0.0
            p.orientation.x = 0.0
            p.orientation.x = 0.0
            p.orientation.w = 1.0
            self.target_pub.publish(p)


    def undistort_frame(self):
        height, width = self.frame.shape[:2]

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.camera_distortion, (width, height), 1, (width, height))
        dst = cv2.undistort(self.frame, self.camera_matrix, self.camera_distortion, None, new_camera_matrix)

        x, y, w, h = roi
        self.undistorted_frame = dst[y:y + h, x:x + w]

    def abort(self, data):
        ## if user chooses to abort calculations will be stopped 
        data = str(data)
        if data == 'data: True':
            self.new_target = (None, None)
            self.previous_target = (None, None)
            self.first_flag = True
            self.no_previous_frame = True

    def run(self):
        while not rospy.is_shutdown():

            if self.new_target[0] != None:
                self.target_tracking_enable.publish(True)
                if(not self.first_flag):
                    self.undistort_frame()
                    self.estimate_distance()
                    self.find_new_target()
                    self.publish_new_target()

                    self.previous_frame = self.frame[:]
                    #print(self.frame.shape)

                self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("target_node", anonymous=True)
    my_tracker = Target_tracker()
    my_tracker.run()




