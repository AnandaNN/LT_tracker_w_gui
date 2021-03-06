#!/usr/bin/env python2
from Tkinter import Tk, Label, Button, Frame
import tkMessageBox
import rospy
from PIL import Image
from PIL import ImageTk
import cv2
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Image as SensorImage
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
        self.frame1 = Frame(master, height = 480, width = 200, bd = 2, relief = "sunken")
        self.frame1.grid(row = 3, column = 1, rowspan = 15)
        self.frame1.grid_propagate(False)
        
        explanation_label = Label(master, justify = 'left',  text = "How you control the drone \nW - move forwards \nS - move down \nA - move let \nD - move right \nI - move up \nK - move down \nJ - rotate counterclockwise \nL - rotate clockwise \nENTER - Takeoff/Land")
        explanation_label.grid(row = 4, column = 1, rowspan = 15)
        
        self.abort_button = Button(master, text = "ABORT", command = self.abort_auto_flight, bg = "grey", fg = "lightgrey", state = "disabled")
        self.abort_button.grid(row = 17, column = 1)

        frame2 = Frame(master, height = 480, width = 200, bd = 2, relief = "sunken")
        frame2.grid(row = 3, column = 3, rowspan = 15)
        frame2.grid_propagate(False)
        
        self.select_target_button = Button(master, text="Select target", command=self.select_target)
        self.select_target_button.grid(row = 4, column = 3)

        self.distance_label = Label(master, text = "Distance: NA")
        self.distance_label.grid(row = 6, column = 3)

        self.battery_label = Label(master, text = "Battery level: NA")
        self.battery_label.grid(row = 8, column = 3)

        header_label = Label(master, text="Choosing target for drone")
        header_label.grid(row = 1, column = 8)
       
        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.grid(row = 1, column = 20)

        self.image_label = Label(text = "", height = 480, width = 640)
        self.image_label.grid(row = 3, column = 6,  columnspan = 15, rowspan = 15)
   

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

        ## Enabling keyboard control of drone
        self.master.bind("<Key>", self.move_drone)
        self.master.bind("<KeyRelease>", self.key_release)
        self.master.bind("<Return>", self.return_key_pressed)

        ## Initialising of publishers and subscribers        
        self.distance_sub = rospy.Subscriber('distance', Pose, self.update_distance_label)
        self.pid_enable_sub = rospy.Subscriber('pid_enable', Bool, self.pid_enabled)
        self.battery_sub = rospy.Subscriber('bebop/CommonStates/BatteryLevelChanged', UInt8, self.update_battery_label)

        self.target_sub = rospy.Subscriber("target", Pose, self.draw_target)

        self.image_sub = rospy.Subscriber('/webcam/image_raw', SensorImage, self.image_subscriber_callback)

        self.gui_target_pub = rospy.Publisher('gui_target', Pose , queue_size=10)
        self.abort_pub = rospy.Publisher('abort', Bool, queue_size=10)
        self.drone_vel_pub = rospy.Publisher('bebop/cmd_vel', Twist,queue_size=10)
        self.takeoff_pub = rospy.Publisher('bebop/takeoff', Empty,queue_size=10)
        self.land_pub = rospy.Publisher('bebop/land', Empty,queue_size=10)

        rospy.init_node('gui', anonymous=True)
        self.rate = rospy.Rate(10)
        rospy.loginfo("GUI initialised")

        self.abort_pub.publish(self.abort_bool)

    def image_subscriber_callback(self, image):
        cv_image = CvBridge().imgmsg_to_cv2(image, "rgb8")
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        if self.circle_center[0] != None:
            cv2.circle(cv_image, (int(self.circle_center[0]), int(self.circle_center[1])), 3, (0, 255, 0), 10)
        self.img = Image.fromarray(cv_image)
        # print("got image")

    def draw_target(self,data):
        self.circle_center = [data.position.x, data.position.y]

    def update_image(self):
        ## Updating the image from the 'drone_cam_sub.py', if it's new. The update is automatic with a frequency 20 Hz (50 ms)
        frequency = 20
        try:
            if self.img != self.prev_img: 
                self.imgtk = ImageTk.PhotoImage(self.img)
                self.image_label.pic = self.imgtk
                self.image_label.configure(image=self.imgtk)
                self.prev_img = self.img
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
        
            
    def move_drone(self, event):
        ## if auto_flying = True no other option than pressing 'g' is possible 
        if self.flying and not self.auto_flying: 
            cmd = Twist()
            factor = 0.5
            rospy.loginfo( "User pressed " + repr(event.char))
            if event.char == 'a':
                cmd.linear.y = factor
            elif event.char == 'd':
                cmd.linear.y = -factor
            elif event.char == 'w':
                cmd.linear.x = factor
            elif event.char == 's':
                cmd.linear.x = -factor
            elif event.char == 'j':
                cmd.angular.z = factor
            elif event.char == 'l':
                cmd.angular.z = -factor
            elif event.char == 'i':
                cmd.linear.z = factor
            elif event.char == 'k':
                cmd.linear.z = -factor
            elif event.char == 'g':
                if not self.abort_bool:
                    self.abort_auto_flight()
                cmd.linear.x= - factor
            self.drone_vel_pub.publish(cmd)
        elif self.flying:
            if event.char == 'g':
                if not self.abort_bool:
                    self.abort_auto_flight()
                cmd.linear.x= - factor
            self.drone_vel_pub.publish(cmd)
                
    def key_release(self,event):
        cmd = Twist()
        cmd.linear.x = 0
        cmd.linear.y = 0
        cmd.linear.z = 0
        cmd.angular.z = 0
        self.drone_vel_pub.publish(cmd) 
    
    def return_key_pressed(self,event):
        ## enabling takeoff and landing
        if not self.flying: 
            self.flying = True
            e = Empty()
            self.takeoff_pub.publish(e)
        else:
            self.abort_pub.publish(True)

            self.flying = False
            e = Empty()
            self.land_pub.publish(e)

    def abort_auto_flight(self):
        ## aborting autonousmous flight and allowing the user the have full control of the drone again
        rospy.loginfo("Aborting")

        self.abort_bool = True
        self.abort_pub.publish(self.abort_bool)

        cmd = Twist()
        cmd.linear.x = 0
        cmd.linear.y = 0
        cmd.linear.z = 0
        cmd.angular.z = 0
        self.drone_vel_pub.publish(cmd) 
        
    def pid_enabled(self, data):
        ## cheching whether the move_to_target program is running
        data = str(data)
        if data == 'data: True': 
            self.abort_button.configure(state = "active", bg = "grey", fg = "black")
        if data == False and self.abort_bool == True:
            self.abort_bool = False
            self.abort_pub.publish(self.abort_bool)

    def update_distance_label(self, data):
        self.distance_label.configure( text = 'Distance: {:02.2f} m'.format(data.position.x) )

    def update_battery_label(self, data):
        self.battery_label.configure( text = 'Battery level: {}'.format(data))

    def publish_pos(self):
        #publishing the position of the target position in pixels
        if not rospy.is_shutdown():
            p = Pose()
            p.position.x = pos_x
            p.position.y = pos_y   
            self.gui_target_pub.publish(p)
            self.rate.sleep()



## sizing the gui window and initialising
ImageFile.LOAD_TRUNCATED_IMAGES = True
root = Tk()
root.geometry('1800x600')

gui = DroneGUI(root)
gui.update_image()

col_count, row_count = root.grid_size()
for col in xrange(col_count):
    root.grid_columnconfigure(col, minsize=40)

for row in xrange(row_count):
    root.grid_rowconfigure(row, minsize=20)

root.mainloop()

