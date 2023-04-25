import cv2 as cv
import mediapipe as mp
import numpy as np
import sys,json,time,math
from utils import DLT, get_projection_matrix
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [900, 1500]

client = mqtt.Client("3dHandTracking")    
client.connect('10.1.1.243',1883)

def fingerMqttSend(fingerName,frame_p3ds,fingerIndex,poseIndex,hand):
    points = np.asarray([frame_p3ds[0], frame_p3ds[fingerIndex], frame_p3ds[fingerIndex+1]])
    normal_vector = np.cross(points[2] - points[0], points[1] - points[2])
    if np.linalg.norm(normal_vector) != 0:
        normal_vector /= np.linalg.norm(normal_vector)
    if hand == 'l':
        normal_vector = 90*normal_vector+60
    else:
        normal_vector = 90*normal_vector-60

    client.publish(hand+fingerName+str(poseIndex),str(0)+','+str(0)+','+str(normal_vector[2]))

def handPositionMqttSend(frame_p2ds,hand):
    if frame_p2ds == []:
        return
    tempStr = ""
    if hand == 'r':
        for j,item in enumerate(frame_p2ds[0]):
            if j<1:
                tempStr = tempStr + str(0.001*(float(item))) + ','
            else:
                tempStr = tempStr + str(-0.001*(float(item)))
    elif hand == 'l':
        for j,item in enumerate(frame_p2ds[0]):
            if j<1:
                tempStr = tempStr + str(-0.003*(float(item))) + ','
            else:
                tempStr = tempStr + str(-0.003*(float(item)))

    client.publish(hand+"3dHandPosition",tempStr+',0.0')

def handOrientationMqttSend(frame_p3ds,hand):
    if len(frame_p3ds)==0:
        return
    #points = np.asarray([frame_p3ds[0], frame_p3ds[5], frame_p3ds[17]])
    x = 680*math.atan2(frame_p3ds[1][1]-frame_p3ds[1][0],frame_p3ds[0][1]-frame_p3ds[0][0])/3.14
    client.publish(hand+"3dHandOrientation",str(x)+','+str(0)+','+str(0))

def sendHandMqttData(points2D,hand):
    handPositionMqttSend(points2D,hand[0])
    handOrientationMqttSend(points2D,hand[0])

    # fingerMqttSend('thumb',points2D,1,1,hand[1])
    # fingerMqttSend('thumb',points2D,2,2,hand[1])
    # fingerMqttSend('thumb',points2D,3,3,hand[1])

    # fingerMqttSend('index',points2D,5,1,hand[1])
    # fingerMqttSend('index',points2D,6,2,hand[1])
    # fingerMqttSend('index',points2D,7,3,hand[1])

    # fingerMqttSend('middle',points2D,9,1,hand[1])
    # fingerMqttSend('middle',points2D,10,2,hand[1])
    # fingerMqttSend('middle',points2D,11,3,hand[1])

    # fingerMqttSend('ring',points2D,13,1,hand[1])
    # fingerMqttSend('ring',points2D,14,2,hand[1])
    # fingerMqttSend('ring',points2D,15,3,hand[1])

    # fingerMqttSend('pinkie',points2D,17,1,hand[1])
    # fingerMqttSend('pinkie',points2D,18,2,hand[1])
    # fingerMqttSend('pinkie',points2D,19,3,hand[1])

def run_mp(input_stream):
    #input video stream
    cap = cv.VideoCapture(input_stream)
    cap.set(3, frame_shape[1])
    cap.set(4, frame_shape[0])

    #create hand keypoints detector object.
    hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands =2, min_tracking_confidence=0.5)

    #containers for detected keypoints for each camera
    kpts_cam = []
    while True:

        #read frames from stream
        ret, frame = cap.read()
        if not ret: break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame.shape[1] != 720:
            frame = frame[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = hands.process(frame)

        #prepare list of hand keypoints of this frame
        #frame0 kpts
        hand = []
        if results.multi_handedness != None:
            if results.multi_handedness[0].classification[0].label[0] == 'L':
                hand.append('r')
                hand.append('l')
            elif results.multi_handedness[0].classification[0].label[0] == 'R':
                hand.append('l')
                hand.append('r')


        frame_keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for p in range(21):
                    #print(results0.multi_handedness[0])
                    #print(p, ':', hand_landmarks.landmark[p].x, hand_landmarks.landmark[p].y)
                    pxl_x = int(round(frame.shape[1]*hand_landmarks.landmark[p].x))
                    pxl_y = int(round(frame.shape[0]*hand_landmarks.landmark[p].y))
                    kpts = [pxl_x, pxl_y]
                    frame_keypoints.append(kpts)

        #no keypoints found in frame:
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame_keypoints = [[-1, -1]]*21

        points2D = frame_keypoints
        kpts_cam.append(frame_keypoints)
        
        if len(hand) == 2: 
              sendHandMqttData(points2D[0:21],hand)
              sendHandMqttData(points2D[21:],[hand[1],hand[0]])
        
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv.imshow('cam0', frame)

        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.


    cv.destroyAllWindows()
    cap.release()

    return np.array(kpts_cam)

if __name__ == '__main__':

    input_stream = '/dev/video0'#'media/cam0_test.mp4'
    #input_stream2 = '/dev/video2'#'media/cam1_test.mp4'

    if len(sys.argv) == 3:
        input_stream = int(sys.argv[1])
        #input_stream2 = int(sys.argv[2])


    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream)
