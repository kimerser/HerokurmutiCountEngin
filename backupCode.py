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

# Give the configuration and weight files for the model and load the network using them.4
# ให้ไฟล์การกำหนดค่าและน้ำหนักสำหรับรุ่นและโหลดเครือข่ายโดยใช้
modelConfiguration = "C:/OPENCVP1/yolofile/custom-yolov4-detector_touch.cfg"
modelWeights = "C:/OPENCVP1/yolofile/custom-yolov4-detector_final_touch3.weights"
# modelConfiguration2 = "MobileNetSSD_deploy.prototxt"
# modelWeights2 =  "MobileNetSSD_deploy.caffe model"
# load our serialized model from disk
# โหลดโมเดลซีเรียลของเราจากดิสก์
print("[INFO] loading model...")
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# net = cv.dnn.readNetFromCaffe(modelConfiguration2, modelWeights2)

# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
# initialize the video writer
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None


# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
# สร้างอินสแตนซ์ตัวติดตาม Centroid ของเราจากนั้นเริ่มต้นรายการเพื่อจัดเก็บ
# ตัวติดตามสหสัมพันธ์ dlib ของเราตามด้วยพจนานุกรมถึง
# แมป ID ออบเจ็กต์ที่ไม่ซ้ำกันแต่ละรายการกับ TrackableObject
ct = CentroidTracker(maxDisappeared=0, maxDistance=25)
trackers = []
trackableObjects = {}

# เริ่มต้นจำนวนเฟรมทั้งหมดที่ประมวลผลจนถึงตอนนี้
# กับจำนวนวัตถุทั้งหมดที่ย้ายขึ้นหรือลง
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
# Get the names of the output layers


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box


# def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    # cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    # Draw a center of a bounding box
    # frameHeight = frame.shape[0]
    # frameWidth = frame.shape[1]
    # cv.line(frame, (0, frameHeight//2 - 50),
    #         (frameWidth, frameHeight//2 - 50), (0, 255, 255), 2)
    # cv.circle(frame, (top+(bottom-top)//2, top +
    #                   (bottom-top)//2), 3, (0, 0, 255), -1)

    # counter = []
    # if (top+(bottom-top)//2 in range(frameHeight//2 - 2, frameHeight//2 + 2)):
    #     coun += 1
    #     # print(coun)

    #     counter.append(coun)
    # if (left+(right-left)//2 in range(frameWidth//2 - 2, frameWidth//2 + 2)):
    #     coun += 1
    #     # print(coun)

    #     counter.append(coun)

    # label = 'Pedestrians: '.format(str(counter))
    # cv.putText(frame, label, (0, 30),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

# Remove the bounding boxes with low confidence using non-maxima suppression


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

    # global totalDown
    # global totalUp
    # global totalLeft
    global total
    global count
    global checklen
    global today
    global checktotal
    
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():

        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        # หากไม่มีวัตถุที่ติดตามได้ให้สร้างขึ้นมา
        if to is None:
            to = TrackableObject(objectID, centroid)
            

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        # มิฉะนั้นจะมีวัตถุที่ติดตามได้เพื่อให้เราสามารถใช้ประโยชน์ได้
        # เพื่อกำหนดทิศทาง
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            # y = [c[1] for c in to.centroids]
            # direction = centroid[1] - np.mean(y)
            # to.centroids.append(centroid)

            # # check to see if the object has been counted or not
            # if not to.counted:
            #     # if the direction is negative (indicating the object
            #     # is moving up) AND the centroid is above the center
            #     # line, count the object

            # if direction < 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
            #     totalUp += 1
            #     to.counted = True
            # # if the direction is positive (indicating the object
            # # is moving down) AND the centroid is below the
            # # center line, count the object
            # elif direction > 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
            #     totalDown += 1
            #     to.counted = True

            x = [c[0] for c in to.centroids]
            direction = centroid[0] - np.mean(x)
            to.centroids.append(centroid)
            # y = [c[1] for c in to.centroids]
            # direction = centroid[1] - np.mean(y)
            # to.centroids.append(centroid)

            # check to see if the object has been counted or not

            # if the direction is negative (indicating the object
            # is moving up) AND the centroid is above the center
            # line, count the object

            # if direction < 0 and centroid[0] in range(frameWidth//2 - 30, frameWidth//2 + 30):
            #     totalLeft += 1
            #     to.counted = True
            # if the direction is positive (indicating the object
            # is moving down) AND the centroid is below the
            # center line, count the object

# direction < 0 and
            now = datetime.now()
            if(total == 0):
                if centroid[0] in range(frameWidth//2-minus, frameWidth//2 + pluslineWidth):
                    total += 1
                    timeminute.append(now)
                    rountTimedelay.append(now)
                    current_time = now.strftime("%H:%M:%S")
                    average.append(current_time)
                    to.counted = True
# direction < pluslineWidth and
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

                        # delta = now-timeminute[0]
                        # if delta.seconds >= 30:
                        #     print("0.30 Min")
                        #     Update 't' variable to new time
                        #     t = dt.datetime.now()

                        # current_time = now.strftime("%H:%M:%S")
                        # print("Current Time =", current_time)
                        average.append(current_time)

                        # print(timeminute[0].strftime("%H:%M:%S"))
                        # print(average[0])
                        # if (now.minute % 1):
                        #     print(now.minute)
                        # delta = now-t
                        if delta.seconds >= 10:
                            print("10 Min")
                            t = datetime.now()
                        # if( average[total-1] <= )
              

                        # -------นับคนทุก10วิ------
                        # if(len(average) % 10 == 0):
                            # s = datetime.strptime(
                            # average[total-10], '%H:%M:%S')
                            # e = datetime.strptime(
                            # average[total-1], '%H:%M:%S')
                            # print(time_diff(s, e))
                            # t = time_diff(s, e).seconds
                            # av = t/10
                            # print(av, " : เฉลี่ยต่อจำนวน 10 คน")

                            # msg = av
                            # r = requests.post(url, headers=headers,
                            #                   data={'message': msg})

                            # print(count)
                            # print("num_of_graduatescount : ",                                  num_of_graduates[1])
                        if(count < len(num_of_graduates)):
                           
                            # print(fac_id[count])
                            # print(count, len(num_of_graduates))
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

                                    insertcountProc = "UPDATE count_proc SET   start_time=%s  where count_id = %s";
                                    val = [startTime,count_id[count]]
                                    database.mycursor.execute(insertcountProc,val)
                                    database.mydb.commit()
                                # insertcountProc = "INSERT INTO count_proc (fac_id, start_time, end_time, data_stamp, current_person, time_per_person) VALUES( %s, %s,null, %s, 0, 0)";
                                # val = [fac_id[count],startTime,today]
                                # database.mycursor.execute(insertcountProc,val)
                                # database.mydb.commit()
                                # if(count < len(num_of_graduates)):
                                #     num_of_graduates[count] += num_of_graduates[count-1]
                                #     # print(fac_id[count])
                                #     insertfacid = "INSERT INTO count_proc (fac_id, start_time, end_time, data_stamp, current_person, time_per_person) VALUES( %s, %s,null, %s, 0, 0)";
                                #     facid = fac_id[count]
                                #     valfacid = [facid,startTime,today]
                                #     database.mycursor.execute(insertfacid,valfacid)
                                #     database.mydb.commit()
                                    # print(num_of_graduates[count])
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


                                    
                                    insertcountProc = "UPDATE count_proc SET   start_time=%s  where count_id = %s";
                                    val = [startTime,count_id[count]]
                                    database.mycursor.execute(insertcountProc,val)
                                    database.mydb.commit()
                                # insertcountProc = "INSERT INTO count_proc (fac_id, start_time, end_time, data_stamp, current_person, time_per_person) VALUES( %s, %s,null, %s, 0, 0)";
                                # val = [fac_id[count],startTime,today]
                                # database.mycursor.execute(insertcountProc,val)
                                # database.mydb.commit()
                                # if(count < len(num_of_graduates)):
                                #     num_of_graduates[count] += num_of_graduates[count-1]
                                #     # print(fac_id[count])
                                #     insertfacid = "INSERT INTO count_proc (fac_id, start_time, end_time, data_stamp, current_person, time_per_person) VALUES( %s, %s,null, %s, 0, 0)";
                                #     facid = fac_id[count]
                                #     valfacid = [facid,startTime,today]
                                #     database.mycursor.execute(insertfacid,valfacid)
                                #     database.mydb.commit()
                                    # print(num_of_graduates[count])
                            
                            

                        # assert time_diff(s, e) == time_diff(s.time(), e.time())

                                # now = datetime.now()
                                # current_time = now.strftime("%H:%M:%S")
                                # average.append(current_time)
                                # print("Current Time =", current_time)

                                # msg = current_time
                                # r = requests.post(url, headers=headers,
                                #                   data={'message': msg})
                    # print(r.text)
                    # print(direction)
                    # print(centroid[0])
                    # print(frameWidth//2 - 30)
                    # print(frameWidth//2 + 30)
                # else :
                #    print("test")
        # store the trackable object in our dictionary

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
                    # msg = txt
                    # r = requests.post(url, headers=headers,
                    #                   data={'message': msg})
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
                    # insertcountProc = "UPDATE count_proc SET   end_time=%s, current_person=%s, time_per_person=%s where count_id = (SELECT count_id FROM count_proc ORDER BY count_id DESC LIMIT 1)";
                    insertcountProc = "UPDATE count_proc SET   end_time=%s, current_person=%s, time_per_person=%s where count_id = %s";
                    val = [finish,check_num_of_graduates,endt,count_id[count]]
                    database.mycursor.execute(insertcountProc,val)
                    database.mydb.commit()
            
        
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        # text = "ID {}".format(objectID)
        # cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
        # drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        # cv.circle(frame, (centroid[0],centroid[0]), 4, (0, 255, 0), -1)
        # cv.circle(frame, (centroid[0],centroid[1]), 4, (0, 0, 255), -1)
        # cv.circle(frame, (centroid[1],centroid[0]), 4, (255, 0, 0), -1)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        
        # cv.circle(frame, (centroid[0],centroid[1]), 4, (0, 255, 0), -1)
    # construct a tuple of information we will be displaying on the
    # frame
        
    
    # info = [
    #     # ("Up", totalUp),
    #     # ("Down", totalDown),
    #     # ("Left", totalLeft),
    #     ("Count", total),
    # ]

    # loop over the info tuples and draw them on our frame
    # for (i, (k, v)) in enumerate(info):
    #     text = "{}: {}".format(k, v)
    #     cv.putText(frame, text, (10, frameHeight - ((i * 20) + 20)),
    #                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py2.avi"

# if (args.video):
#     # Open the video file
#     if not os.path.isfile(args.video):
#         print("Input video file ", args.video, " doesn't exist")
#         sys.exit(1)
#     cap = cv.VideoCapture(args.video)
#     outputFile = args.video[:-4]+'_yolo_out_py.mp4'
# else:
# Webcam input
# cap = cv.VideoCapture("videos/rmuti.mp4")
cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
# vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
#     cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
cv.namedWindow('Control')


button = [20,60,50,250]
def process_click(event, x, y,flags, params):
    global total
    # check if the click is within the dimensions of the button
    if event == cv.EVENT_LBUTTONDOWN:
        if y > button[0] and y < button[1] and x > button[2] and x < button[3]:   
            total += 1
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

    # Put efficiency informati                                                               on. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    # t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.putText(frame, label, (0, 15),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Write the frame with the detection boxes

    # vid_writer.write(frame.astype(np.uint8))
    cv.imshow(winName, frame)
    cv.imshow('Control', control_image)
    # rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # cv.imshow("test", rgb)
