#!/usr/bin/env python2
import roslib
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import geometry_msgs.msg
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion,quaternion_from_matrix,compose_matrix, concatenate_matrices, translation_from_matrix
import tf
import tf_conversions
import tf2_ros
import tf

"""
move_to_target controls the drone according to the distance to the target point

The constant defines the center of the image at 1/2 width and 1/4 height
The file name is the name of the .csv file, which outputs the cmd_vel and dfference between target and drone 
"""


C_MID = (421, 120)
file_name = 'test.csv'


class Move_to_target:

  def __init__(self):
    #cycle rate of the node (Hz)
    self.loop_rate = rospy.Rate(30)

    self.listener = tf.TransformListener()

    self.time = 0
    self.abort_bool = False
    self.x_movement_enable = False

    #PID
    self.dt = []
    self.previous_time = None

    self.error_buffer_x = []
    self.error_buffer_y = []
    self.error_buffer_z = []

    self.distance_buffer = [0]

    #PID values for the x direction
    self.integrator_x = 0
    self.Kp_x = 0.101
    self.Ki_x = 0
    self.Kd_x = 0.26

    #PID values for the y direction
    self.integrator_y = 0
    self.Kp_y = 0.101
    self.Ki_y = 0.00045
    self.Kd_y = 0.26


    #PID values for the z direction
    self.integrator_z = 0
    self.Kp_z = 0.3
    self.Ki_z = 0.0008
    self.Kd_z = 0.6

    #the anti windup values for the integrator
    self.anti_windup = 20

    #publisher
    self.drone_vel = rospy.Publisher('bebop/cmd_vel',Twist,queue_size=1)
    self.error_pub = rospy.Publisher('error_to_target',Pose,queue_size=1)
    self.error_velocity_pub = rospy.Publisher('error_velocity_to_marker',Twist,queue_size=1)
    self.pid_enable_pub = rospy.Publisher('pid_enable', Bool, queue_size= 1)
    
    #subscriber
    self.target_sub = rospy.Subscriber('target', Pose, self.read_target)
    self.abort_sub = rospy.Subscriber('abort', Bool, self.abort)
    
    self.trans = [None,None,None]
    self.rot = [None,None,None,None]
    self.recieved_target = False
    self.pid_enable_pub.publish(self.recieved_target)
    
    #for the csv
    self.array_x = []
    self.array_y = []
    self.array_z = []
    self.time    = 0
    self.time_array = []
    self.cmd_velx = []
    self.cmd_vely = []
    self.cmd_velz = []
    
    self.img_x = []
    self.img_y = []
    print("Control initialised")

    
  def read_target(self, data): 
        ## Reads the target from the target_tracking program, and converts the coordinates accordingly:
        ## img -> drone coordinates: x -> y, y -> -z, z -> x
        ## The target is in the z-direction the approximated, if no value was read, and limited at 2 m
        print("read target")
        print(data.position.x)
        z = data.position.z/np.cos(np.radians(13))
        n = len(self.distance_buffer)
        if not self.x_movement_enable and z != 0:
            self.x_movement_enable = True
        elif z == 0:
            if sum(self.distance_buffer) == 0 or n == 0:
                self.x_movement_enable = False
            else:
                diff = []
                for i in range(n-1):
                    diff.append(self.distance_buffer[i+1] - self.distance_buffer[i])
                z = sum(diff)/len(diff)
        elif z > self.distance_buffer[n-1] + 0.5:
            z = self.distance_buffer[n-1] + 0.5
        elif z < self.distance_buffer[n-1] - 0.5:
            z = self.distance_buffer[n-1] - 0.5

        if z > 2.0:
            z = 2.0

        self.distance_buffer.append(z)
        self.distance_buffer.pop(0)

        self.trans = [data.position.x - C_MID[0], data.position.y - C_MID[1], z]
        self.rot = [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]
        self.recieved_target = True

        self.pid_enable_pub.publish(self.recieved_target)
       
  def publish_error(self): 
    #publish the position of the marker to enable view in rqt_plot
    n = len(self.error_buffer_x)   
    p = Pose()
    p.position.x = self.error_buffer_x[n-1]
    p.position.y = self.error_buffer_y[n-1]
    p.position.z = self.error_buffer_z[n-1]
    self.error_pub.publish(p)

  def filter_errors(self, trans): #to ensure that the derivative doesn't spike by using 10 last values
    error_x, error_y , error_z = trans

    self.error_buffer_x.append(error_x)
    self.error_buffer_y.append(error_y)
    self.error_buffer_z.append(error_z)

    if len(self.error_buffer_x)>10:
        self.error_buffer_x.pop(0)
    if len(self.error_buffer_y)>10:
        self.error_buffer_y.pop(0)
    if len(self.error_buffer_z)>10:
        self.error_buffer_z.pop(0)

    ## calculating the output for x, y and z direction using PIDs
  def PID_x(self):  
    n = len(self.error_buffer_x)
    self.integrator_x = self.integrator_x + self.error_buffer_x[n-1]* self.dt[n-1]

    if(self.integrator_x > self.anti_windup):
        self.integrator_x = self.anti_windup
    elif(self.integrator_x < -self.anti_windup):
        self.integrator_x = -self.anti_windup

    derivative_x = (self.error_buffer_x[n-1] - self.error_buffer_x[0])/ np.sum(self.dt)

    output_x = self.Kp_x*self.error_buffer_x[n-1] + self.Ki_x*self.integrator_x + self.Kd_x*derivative_x

    limit = 0.1
    if( output_x > limit):
        output_x = limit
    elif(output_x < -limit):
        output_x = -limit

    return output_x


  def PID_y(self):
    n = len(self.error_buffer_y)
    self.integrator_y = self.integrator_y + self.error_buffer_y[n-1]*self.dt[n-1]

    if(self.integrator_y > self.anti_windup):
        self.integrator_y = self.anti_windup
    elif(self.integrator_y < -self.anti_windup):
        self.integrator_y = -self.anti_windup

    if n > 1:
        derivative_y = (self.error_buffer_y[n-1] - self.error_buffer_y[0])/np.sum(self.dt)
    else:
        derivative_y = 0

    output_y = self.Kp_y*self.error_buffer_y[n-1] + self.Ki_y*self.integrator_y + self.Kd_y*derivative_y
  
    limit = 0.1
    if( output_y > limit):
        output_y = limit
    elif(output_y < -limit):
        output_y = -limit

    return output_y

  def PID_z(self):
    n = len(self.error_buffer_z)
    self.integrator_z = self.integrator_z + self.error_buffer_z[n-1]*self.dt[n-1]

    if(self.integrator_z > self.anti_windup):
        self.integrator_z = self.anti_windup
    elif(self.integrator_z < -self.anti_windup):
        self.integrator_z = -self.anti_windup

    if n > 1:
        derivative_z = (self.error_buffer_z[n-1] - self.error_buffer_z[0])/ np.sum(self.dt)
    else: 
        derivative_z = 0

    output_z = self.Kp_z*self.error_buffer_z[n-1] + self.Ki_z*self.integrator_z + self.Kd_z*derivative_z
    
    limit = 0.1
    if( output_z > limit):
        output_z = limit
    elif(output_z < -limit):
        output_z = -limit

    return  output_z

  def calculate_dt(self):
    ## calculating the time from last 10 values
    current_time = rospy.get_time()
    if self.previous_time == None:
        self.previous_time = current_time
    self.dt.append(current_time - self.previous_time)
    self.time += current_time - self.previous_time
    self.time_array.append(self.time)

    if len(self.dt)>10:
        self.dt.pop(0)
    self.previous_time = current_time

  def pixel_to_meter(self):
    ## Converting the pixel value from the target_tracking program to meters
    u = self.trans[0] 
    v = self.trans[1] 
    fx = 520.235734
    fy = 512.906134

    if self.trans[2] == 0:
        dist = 2.5
    else:
        dist = self.trans[2]
 
    d_c = dist / np.cos(np.deg2rad(30))
    y = np.cos(np.deg2rad(30)) * v * d_c / fy
    x = u * dist / fx
    return x , y
  
  def find_error_to_target(self):
    ## finding the difference in distance between target point and drone using transformations 
    try:  
        (trans_drone,rot_drone) = self.listener.lookupTransform('/end_effector','/camera_optical', rospy.Time(0))
        x, y = self.pixel_to_meter()
        print("x" , x)
        print("y" , y)
        trans_target = [x,y,self.trans[2]]
        rot_target = [0.11320321, 0, 0, 0.99357186]
        euler_target = euler_from_quaternion(rot_target)
        euler_drone = euler_from_quaternion(rot_drone)
        
        matrix_target = compose_matrix(None,None,euler_target,trans_target,None)
        matrix_drone = compose_matrix(None,None,euler_drone,trans_drone,None)

        matrix_result = concatenate_matrices(matrix_drone, matrix_target)

        trans_result = translation_from_matrix(matrix_result)
        rot_result = quaternion_from_matrix(matrix_result)

        roll_result,pitch_result,yaw_result = euler_from_quaternion(rot_result)

        #for the .csv file 
        self.array_x.append(trans_result[0])
        self.array_y.append(trans_result[1])
        self.array_z.append(trans_result[2])


        self.img_x.append(self.trans[0] + C_MID[0])
        self.img_y.append(self.trans[1] + C_MID[1])
        return True , trans_result

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        print(e)
        return False , None
    
  def publish_new_velocity(self):
    cmd = Twist()
    if self.x_movement_enable:
        x = self.PID_x()
    else: 
        x = 0
    cmd.linear.x = x
    cmd.linear.y = self.PID_y()
    cmd.linear.z = self.PID_z()
    self.cmd_velx.append(cmd.linear.x)
    self.cmd_vely.append(cmd.linear.y)
    self.cmd_velz.append(cmd.linear.z)
    print("published cmd_vel")
    self.drone_vel.publish(cmd)

  def abort(self, data):
      ## If the user chooses to abort the autonomous flight, then the program won't publish new values to 
      data = str(data)
      if data == 'data: True':
        self.pid_enable_pub.publish(False)
        self.abort_bool = True

  def update_PID(self):
    ## running relevant methods for publishing cmd_vel and updating .csv file
    if(self.recieved_target):
        self.pid_enable_pub.publish(True)
        succesful, trans = self.find_error_to_target()   
        if(succesful):
            self.calculate_dt()
            self.filter_errors(trans)
            self.publish_error() 
            
            self.publish_new_velocity()
            np.savetxt(file_name, np.transpose([np.array(self.time_array), np.array(self.array_x), np.array(self.array_y), np.array(self.array_z), np.array(self.cmd_velx), np.array(self.cmd_vely), np.array(self.cmd_velz)]), delimiter=',',fmt="%s") 
            
  def start(self):
    while not rospy.is_shutdown():
        if self.abort_bool == False:
            self.update_PID()
            self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("pid_node", anonymous=True)
    move = Move_to_target()
    move.start()
