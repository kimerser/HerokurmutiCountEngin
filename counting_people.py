import cv2 as cv
from imutils.video import FPS
import argparse
import numpy as np
from numpy.core.numeric import False_
import database
from numpy.lib.function_base import average
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
import requests
import time
from datetime import datetime, time as datetime_time, timedelta



def time_diff(start, end):
    if isinstance(start, datetime_time):  # convert to datetime
        assert isinstance(end, datetime_time)
        start, end = [datetime.combine(datetime.min, t) for t in [start, end]]
    if start <= end:  # e.g., 10:33:26-11:15:49
        return end - start
    else:  # end < start e.g., 23:55:00-00:25:00
        end += timedelta(1)  # +day
        assert end > start
        return end - start

timedeley = [];
left = [];
right = [];
linetoken = [];
timeNotify = [];
for parameter in database.parameterResult:
    print(parameter)
    timedeley.append(parameter[0])
    left.append(parameter[1])
    right.append(parameter[2])
    linetoken.append(parameter[3])
    timeNotify.append(parameter[4])

url = 'https://notify-api.line.me/api/notify'
token = linetoken[0]
headers = {
    'content-type':
    'application/x-www-form-urlencoded',
    'Authorization': 'Bearer '+token
}

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

total = 0
checktotal = 0
checklen = 0
count = 0
pluslineWidth = right[0]
minus = left[0]
timedelay = timedeley[0]
t = datetime.now()
startTime = datetime.now()
average = []
timeminute = []
amount = []
amount.append(0)
txt = "เตรียมข้อมูลบัณฑิต "
sumnum = 0
num_of_graduates = []
fac_name = []
fac_id = []
rountTimedelay = []
count_id = []
for data in database.myresult:
    txtnum = data[0]
    sumnum += txtnum
    txtfacname = data[1]
    num_of_graduates.append(data[0])
    fac_name.append(data[1])
    fac_id.append(data[2])
    count_id.append(data[3])
    txt += "[ "+txtfacname+" จำนวน "+str(txtnum)+" คน]"

txt += " จำนวนนักศึกษาทั้งสิ้น "+str(sumnum) + " คน"
msg = txt
r = requests.post(url, headers=headers, data={'message': msg})
msg = "## เริ่มการนับ การปรับปรุงสถานะทุกๆ "+str(timeNotify[0]) + " วินาที"
r = requests.post(url, headers=headers, data={'message': msg})

today = t.strftime("%Y-%m-%d")
# insertcountProc = "INSERT INTO count_proc (fac_id, start_time, end_time, data_stamp, current_person, time_per_person) VALUES( %s, %s,null, %s, 0, 0)";
insertcountProc = "UPDATE count_proc SET fac_id=%s, start_time=%s,  date_stamp=%s WHERE count_id=%s";
val = [fac_id[0],startTime,today,count_id[0]]
database.mycursor.execute(insertcountProc,val)
database.mydb.commit()


maxresult = len(fac_name)

parser = argparse.ArgumentParser(
    description='Object Detection using YOLO in OPENCV')

parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


modelConfiguration = "C:/OPENCVP1/yolofile/custom-yolov4-detector_touch.cfg"
modelWeights = "C:/OPENCVP1/yolofile/custom-yolov4-detector_final_touch3.weights"
print("[INFO] loading model...")
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
# initialize the video writer
writer = None

W = None
H = None

ct = CentroidTracker(maxDisappeared=0, maxDistance=25)
trackers = []
trackableObjects = {}

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(frame, outs):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    rects = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.

    # สแกนผ่านกล่องขอบเขตทั้งหมดที่ส่งออกจากเครือข่ายและเก็บเฉพาะไฟล์
    # คนที่มีคะแนนความมั่นใจสูง กำหนดป้ายชื่อชั้นของกล่องเป็นชั้นเรียนที่มีคะแนนสูงสุด
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    # ดำเนินการปราบปรามแบบไม่สูงสุดเพื่อกำจัดกล่องที่ทับซ้อนกันด้วย
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class "person"
        if classIds[i] == 0:
            rects.append((left, top, left + width, top + height))
            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            # ใช้ตัวติดตามเซนทรอยด์เพื่อเชื่อมโยงวัตถุเก่า (1)
            # centroids กับ (2) วัตถุ centroids คำนวณใหม่
            objects = ct.update(rects)
            counting(objects)

            # drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

def counting(objects):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global total
    global count
    global checklen
    global today
    global checktotal
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            x = [c[0] for c in to.centroids]
            direction = centroid[0] - np.mean(x)
            to.centroids.append(centroid)
            now = datetime.now()
            if(total == 0):
                if centroid[0] in range(frameWidth//2-minus, frameWidth//2 + pluslineWidth):
                    total += 1
                    timeminute.append(now)
                    rountTimedelay.append(now)
                    current_time = now.strftime("%H:%M:%S")
                    average.append(current_time)
                    to.counted = True
            else:
                if centroid[0] in range(frameWidth//2-minus, frameWidth//2 + pluslineWidth):
                    delta = now-rountTimedelay[0]
                    # delta.seconds >= timedelay or
                    if  not to.counted:
                        if delta.seconds < 0.3:
                            to.counted = True
                            continue
                        total += 1
                        rountTimedelay[0] = now
                        to.counted = True
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")

                        now = datetime.now()
                        if (total == 1):
                            timeminute.append(now)
                        average.append(current_time)
                        updateDataBaseCompletely();
            # now = datetime.now()
            sendLineAndUpdateDatabase();
            
    
        trackableObjects[objectID] = to
        cv.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        
def updateDataBaseCompletely():
    global count
    global checktotal
    global checklen
    if(count < len(num_of_graduates)):
                            check_num_of_graduates = total
                            print(checktotal)
                            if(checktotal != 0) :
                                print(checktotal)
                                check_num_of_graduates  =  total - checktotal
                                if (num_of_graduates[count] == check_num_of_graduates):
                                    print("checklen : ", checklen)
                                    checklen = checklen+1
                                    print("นักศึกษาคณะ ",
                                        fac_name[count], " ครบแล้ว")
                                    insertcountProc = "UPDATE count_proc SET   current_person=%s  where count_id = %s";
                                    val = [check_num_of_graduates,count_id[count]]
                                    database.mycursor.execute(insertcountProc,val)
                                    database.mydb.commit()
                                    count += 1
                                    checktotal = total
                                    insertcountProc = "UPDATE count_proc SET fac_id=%s, start_time=%s,  date_stamp=%s WHERE count_id=%s";
                                    # val = [fac_id[count],startTime,today]
                                    # insertcountProc = "UPDATE count_proc SET   start_time=%s  where count_id = %s";
                                    val = [fac_id[count],startTime,today,count_id[count]]
                                    database.mycursor.execute(insertcountProc,val)
                                    database.mydb.commit()
                            else :
                                if (num_of_graduates[count] == total):
                                    print("checklen : ", checklen)
                                    print("total : ", total)
                                    checklen = checklen+1
                                    print("นักศึกษาคณะ ",
                                        fac_name[count], " ครบแล้ว")
                                    insertcountProc = "UPDATE count_proc SET   current_person=%s  where count_id = %s";
                                    val = [check_num_of_graduates,count_id[count]]
                                    database.mycursor.execute(insertcountProc,val)
                                    database.mydb.commit()
                                    count += 1
                                    checktotal = total
                                    print("นักศึกษาคณะ : " , checklen)
                                    # insertcountProc = "UPDATE count_proc SET   start_time=%s  where count_id = %s";
                                    insertcountProc = "UPDATE count_proc SET fac_id=%s, start_time=%s,  date_stamp=%s WHERE count_id=%s";
                                    val = [fac_id[count],startTime,today,count_id[count]]
                                    
                                    database.mycursor.execute(insertcountProc,val)
                                    database.mydb.commit()

def sendLineAndUpdateDatabase():
    now = datetime.now()
    if (total >= 1 and   checklen < len(fac_name) ):    
                delta = now-timeminute[0]
                if delta.seconds >= timeNotify[0]:
                    # print("30 sec")
                    current_time = now.strftime("%H:%M:%S")
                    s = datetime.strptime(
                        timeminute[0].strftime("%H:%M:%S"), '%H:%M:%S')
                    e = datetime.strptime(
                        now.strftime("%H:%M:%S"), '%H:%M:%S')
                    start = datetime.strptime(
                        startTime.strftime("%H:%M:%S"), '%H:%M:%S')
                    # print(time_diff(s, e))
                    t = time_diff(s, e).seconds
                    endt = time_diff(start, e).seconds
                    endt = endt/total
                    print("--------end--------")
                    print(endt)
                    avtime = t/(total - amount[0])
                    minutAvg = 60 / avtime 
                    minutAvg = f'{minutAvg:.2f}'
                    x = f'{avtime:.2f}'
                    balance = sumnum-total
                    finish = float(balance) * float(avtime)
                    finish = (datetime.now() + timedelta(seconds=finish)  )
                    endTime  = finish.strftime("%H:%M:%S")

                    amount[0] = (total - amount[0])
                    
                    txt = "รายงานนับจำนวนบัณฑิต " + \
                        str(sumnum) + " คน รับแล้ว "+str(total) + \
                        " คน เหลืออีก " + str(balance) + " คน"
                    txt += "เฉลี่ยคนละ : " + \
                        str(x) + " วินาทีหรือ  "+ str(minutAvg) +" คน/นาที เวลาปัจจุบัน [" + \
                        current_time+"] คาดการณ์เวลาสิ้นสุด ["+ endTime
                    txt += "] บัณฑิตที่เข้ารับปัจจุบัน ลำดับ " + \
                        str(total) + " : " + fac_name[count]
                    msg = txt
                    r = requests.post(url, headers=headers,
                                      data={'message': msg})
                    print("รายงานนับจำนวนบัณฑิต "+str(sumnum) + " คน รับแล้ว " +
                          str(total)+" คน เหลืออีก " + str(balance) + " คน")
                    print("เฉลี่ยคนละ : ", avtime,
                          " วินาทีหรือ ", minutAvg ," คน/นาที เวลาปัจจุบัน "+current_time+"คาดการณ์เวลาสิ้นสุด "+ endTime )
                    print("บัณฑิตที่เข้ารับปัจจุบัน ลำดับ " +
                          str(total) + " : " + fac_name[count])
                    amount[0] = total
                    print(" = ", amount[0])

                    timeminute[0] = now
                    print(timeminute[0])
                    check_num_of_graduates = total;
                    if(checklen != 0):
                        check_num_of_graduates  =  total - checktotal
                    insertcountProc = "UPDATE count_proc SET   end_time=%s, current_person=%s, time_per_person=%s where count_id = %s";
                    val = [finish,check_num_of_graduates,endt,count_id[count]]
                    database.mycursor.execute(insertcountProc,val)
                    database.mydb.commit()
# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
outputFile = "yolo_out_py2.avi"
cap = cv.VideoCapture(0)
cv.namedWindow('Control')
button = [20,60,50,250]
def process_click(event, x, y,flags, params):
    global total
    # check if the click is within the dimensions of the button
    if event == cv.EVENT_LBUTTONDOWN:
        if y > button[0] and y < button[1] and x > button[2] and x < button[3]:   
            total += 1
            updateDataBaseCompletely()
            sendLineAndUpdateDatabase()
cv.setMouseCallback('Control',process_click)  
control_image = np.zeros((80,300), np.uint8)
control_image[button[0]:button[1],button[2]:button[3]] = 250
cv.putText(control_image, 'Increase Count',(50,50), cv.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255), 2)

while cv.waitKey(1) < 0:
    # get frame from the video
    hasFrame, frame = cap.read()
    # frame = imutils.resize(frame, width=608, height=608)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    info = [
        ("Count", total),
    ]
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
    # cv.line(frame, (0, frameHeight // 2),
    #         (frameWidth, frameHeight // 2), (0, 255, 255), 2)
        cv.putText(frame, text,  (10, frameHeight - ((i * 20) + 20)),
                cv.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
    cv.line(frame, (frameWidth//2 - minus, 0),
            (frameWidth//2 - minus, frameHeight), (0, 255, 255), 2)
    cv.line(frame, (frameWidth//2 + pluslineWidth, 0),
            (frameWidth//2 + pluslineWidth, frameHeight), (0, 255, 255), 2)
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break 
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(
        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 4, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)
    cv.imshow(winName, frame)
    cv.imshow('Control', control_image)
