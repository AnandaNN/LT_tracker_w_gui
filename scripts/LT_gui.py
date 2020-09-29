#!/usr/bin/env python2
from Tkinter import Tk, Label, Button, Frame
import tkMessageBox
import rospy
from PIL import Image
from PIL import ImageTk
import cv2
from geometry_msgs.msg import Pose, Twist, Point
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Empty, UInt8
import base64
from cv_bridge import CvBridge, CvBridgeError
import numpy
from PIL import ImageFile
import os
import numpy as np

"""
Code for the gui, which enables the user to view the image from the drone, control the drone, 
and allows the user to select a target point on the live image. 
"""

pos_x = -1
pos_y = -1

def save_pos(event): 
    ## updating the position of the target point from position of mouse click on image 
    global pos_x
    global pos_y
    pos_x = event.x
    pos_y = event.y


def display_message_box(message):
    return tkMessageBox.askyesno("Information", message)
    

class DroneGUI:
    def __init__(self, master):
        self.master = master
        master.title("Drone GUI")
        
        ## Initialising framework of GUI
        frame1 = Frame(master, height = 480, width = 150, bd = 2, relief = "sunken")
        frame1.grid(row = 3, column = 3, rowspan = 15)
        frame1.grid_propagate(False)
        
        self.select_target_button = Button(master, text="Select target", command=self.select_target)
        self.select_target_button.grid(row = 4, column = 3)

        self.battery_label = Label(master, text = "Battery level: NA")
        self.battery_label.grid(row = 6, column = 3)

        self.x_distance_label = Label(master, text = "X Distance: NA")
        self.x_distance_label.grid(row = 8, column = 3)

        self.y_distance_label = Label(master, text = "Y Distance: NA")
        self.y_distance_label.grid(row = 10, column = 3)

        self.z_distance_label = Label(master, text = "Z Distance: NA")
        self.z_distance_label.grid(row = 12, column = 3)

        

        header_label = Label(master, text="Choosing target for drone")
        header_label.grid(row = 1, column = 8)
       
        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.grid(row = 1, column = 20)

        # self.image_label = Label(text = "", height = 480, width = 640)
        # self.image_label = Label(text = "", height = 720, width = 1280)
        self.image_label = Label(text = "", height = 448, width = 800)
        self.image_label.grid(row = 3, column = 6,  columnspan = 15, rowspan = 15)
   
        self.frame_num = 0

        ## Initialising variables for selecting target
        self.imgClick = False
        self.bridge = CvBridge()
        self.enable_video_stream = None
        self.prev_img = None
        self.select_target_bool = False
        
        self.circle_center = [None, None]

        ## Initialising variables for autonoumous flight
        self.flying = False
        self.auto_flying = False
        self.abort_bool = False

        ## Initialising of publishers and subscribers        
        # self.distance_sub = rospy.Subscriber('/dtu_controller/current_frame_pose', Twist, self.update_distance_label)
  
        self.battery_sub = rospy.Subscriber('/dji_sdk/battery_state', BatteryState, self.update_battery_label)

        self.target_sub = rospy.Subscriber("/target", Point, self.draw_target)

        self.distance_error_sub = rospy.Subscriber("/distance_error", Point, self.update_distance_error)

        self.image_sub = rospy.Subscriber('/webcam/image_raw', SensorImage, self.image_subscriber_callback)
        self.image_sub = rospy.Subscriber('/camera/image_raw', SensorImage, self.image_subscriber_callback)
        #self.image_sub = rospy.Subscriber('/camera/image_decompressed', SensorImage, self.image_subscriber_callback)

        self.gui_target_pub = rospy.Publisher('/gui_target', Point , queue_size=1)
        
        self.frame_pub = rospy.Publisher('/frame_num', UInt8 , queue_size=1)

        rospy.init_node('gui', anonymous=False)

        self.rate = rospy.Rate(10)
        rospy.loginfo("GUI initialised")


    def image_subscriber_callback(self, image):
        cv_image = CvBridge().imgmsg_to_cv2(image, "rgb8")
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        if self.circle_center[0] != None:
            cv2.circle(cv_image, (int(self.circle_center[0]), int(self.circle_center[1])), 3, (0, 255, 0), 10)
        self.img = Image.fromarray(cv_image)
        # print("got image")

    def draw_target(self,data):
        self.circle_center = [data.x, data.y]

    def update_image(self):
        ## Updating the image from the 'drone_cam_sub.py', if it's new. The update is automatic with a frequency 20 Hz (50 ms)
        frequency = 20
        try:
            if self.img != self.prev_img: 
                self.imgtk = ImageTk.PhotoImage(self.img)
                self.image_label.pic = self.imgtk
                self.image_label.configure(image=self.imgtk)
                self.prev_img = self.img
                #nn = UInt8()
                #nn = self.frame_num
                #while self.frame_pub.get_num_connections() < 1:
                #    rospy.loginfo("%d",self.frame_pub.get_num_connections())
                #    rospy.Rate(10).sleep()
                #rospy.loginfo("Done: %d",self.frame_pub.get_num_connections())
                #self.frame_pub.publish(nn)
                #self.frame_num += 1
                # rospy.loginfo("%d", nn)
        except:
            print("Image not updated")
        self.enable_video_stream = self.image_label.after(int(1000/frequency), self.update_image)

    def select_target(self):
        ## Allows the user to select target, and interrupt selection if wanted
        if not self.select_target_bool:
            rospy.loginfo( "User is selecting target")
            self.select_target_bool = True
            self.imgClick = True
            self.select_target_button.configure(text = "Cancel")
            self.image_label.bind("<Button-1>", self.target_selected)
            self.image_label.configure(cursor="dotbox")
        else:
            rospy.loginfo("User cancelled selection")
            self.select_target_bool = False
            self.imgClick = False
            self.select_target_button.configure(text="Select target")
            self.image_label.unbind("<Button-1>")
            self.image_label.configure(cursor="")  

    def target_selected(self, event):
        ## Once target has been selected, variables and functions need to be reset. 
        ## By un-commenting line 158 control will be disabled, once autonomous flight is enabled 
        ## (For now it is possible to interfere with the drone by using the keyboard)
        self.select_target_bool = False
       
        rospy.loginfo("User selected target")
        self.imgClick = False
        save_pos(event)
        self.publish_pos()
        self.update_image()
        self.select_target_button.configure(text="Select target")
        self.image_label.unbind("<Button-1>")
        self.image_label.configure(cursor="") 
        #self.auto_flying = True
        
    def update_distance_label(self, data):
        self.x_distance_label.configure( text = 'Distance:\n{:02.2f} m'.format(data.linear.x) )

    def update_distance_error(self, data):
        self.x_distance_label.configure( text = 'X Distance:\n{:02.2f} m'.format(data.x))
        self.y_distance_label.configure( text = 'Y Distance:\n{:02.2f} m'.format(data.y))
        self.z_distance_label.configure( text = 'Z Distance:\n{:02.2f} m'.format(data.z))

    def update_battery_label(self, data):
        self.battery_label.configure( text = 'Battery level:\n{} %'.format(data.percentage))

    def publish_pos(self):
        #publishing the position of the target position in pixels
        if not rospy.is_shutdown():
            p = Point()
            p.x = pos_x
            p.y = pos_y
            p.z = 0
            self.gui_target_pub.publish(p)
            self.rate.sleep()
            rospy.loginfo("New Gui target published (%d, %d)", pos_x, pos_y)



## sizing the gui window and initialising
ImageFile.LOAD_TRUNCATED_IMAGES = True
root = Tk()
root.geometry('1600x850')

gui = DroneGUI(root)
gui.update_image()

col_count, row_count = root.grid_size()
for col in xrange(col_count):
    root.grid_columnconfigure(col, minsize=40)

for row in xrange(row_count):
    root.grid_rowconfigure(row, minsize=20)

root.mainloop()

