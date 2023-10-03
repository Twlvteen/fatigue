# 变量声明
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time
import cv2
import dlib
# 通过文件捕获面部特征
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_key = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])
# 初始化各个变量的值
counter = 0
total = 0
t = 0
save = 0
mode = 0
close = 0
last_bpm = 0
prev = 0
# 初始化一些语句防止报错
pct = ""
form = ""


def eye_check(eye):  # 检测眼睛
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear


def shape_np(shape_01, dtype="int"):  # 获取面部各个特征点
    coord = np.zeros((shape_01.num_parts, 2), dtype=dtype)
    for i in range(0, shape_01.num_parts):
        coord[i] = (shape_01.part(i).x, shape_01.part(i).y)
    return coord


def face_detect():  # 面部识别
    gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detect = face_cascade.detectMultiScale(gry, 1.3, 5)
    for (x, y, oo, pp) in faces_detect:
        cv2.rectangle(frame, (x, y), (x + oo, y + pp), (0, 255, 0), 5)


def get_nose_roi(face_points_01):  # 识别鼻子的感兴趣区域
    points = np.zeros((len(face_points_01.parts()), 2))
    for i, part in enumerate(face_points_01.parts()):
        points[i] = (part.x, part.y)
    min_x = int(points[36, 0])
    min_y = int(points[28, 1])
    max_x = int(points[45, 0])
    max_y = int(points[33, 1])
    left = min_x
    right = max_x
    top = min_y + (min_y * 0.02)
    bottom = max_y + (max_y * 0.02)
    return int(left), int(right), int(top), int(bottom)  # 特定公式计算，计算公式来源于网络


def get_forehead_roi(face_points_02):  # 识别额头的感兴趣区域
    points = np.zeros((len(face_points_02.parts()), 2))
    for i, part in enumerate(face_points_02.parts()):
        points[i] = (part.x, part.y)
    min_x = int(points[21, 0])
    min_y = int(min(points[21, 1], points[22, 1]))
    max_x = int(points[22, 0])
    max_y = int(max(points[21, 1], points[22, 1]))
    left = min_x
    right = max_x
    top = min_y - (max_x - min_x)
    bottom = max_y * 0.98
    return int(left), int(right), int(top), int(bottom)  # 特定公式计算，计算公式来源于网络


def get_roi_avg(frame, face_points):  # 获取额头和鼻子的roi,并计算平均值
    fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
    nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)
    fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
    nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
    return get_avg(fh_roi, nose_roi)


def set_bpm(filtered_values, fps, buffer_size, last_bpm):  # 计算bpm值（即心率）
    fft = np.abs(np.fft.rfft(filtered_values))
    freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1)  # 算法来源于网络
    while True:
        max_idx = fft.argmax()
        bps = freqs[max_idx]
        if bps < 0.83 or bps > 3.33:  # 如果bpm太太或太小，即提取的roi不符合心率特征，就不做处理
            fft[max_idx] = 0
        else:
            bpm = bps * 60.0  # 如果bpm合理，就按60s计算，得出每分钟的心率
            break
    if last_bpm > 0:
        bpm = (last_bpm * 0.9) + (bpm * 0.1)
    return bpm


def get_avg(roi1, roi2):  # 取两数/数列的平均值
    roi1_green = roi1[:, :, 1]
    roi2_green = roi2[:, :, 1]
    avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
    return avg


# 初始化dlib库，使用里面的部分内容
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_key["left_eye"]
(rStart, rEnd) = face_key["right_eye"]
# 初始化一些数列，防止后面报错
storage = []
roi_avg_values = []
graph_values = []

cap = cv2.VideoCapture(0)  # 启用摄像头
while True:
    ret, frame = cap.read()
    current = time.time()
    if not ret:  # 如果没启动则关闭
        break
    view = np.array(frame)
    fps = 1/(current - prev)  # 计算fps的值
    prev = current
    face_detect()  # 调用自定义函数识别面部
    (h, w) = frame.shape[:2]
    width = 1200
    r = width / float(w)
    dim = (width, int(h * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 识别灰度
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_check(leftEye)
        rightEAR = eye_check(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        if ear < 0.275:  # 如果ear值小于0.275，则认为眨眼一次
            counter += 1
        if ear < 0.2:  # 如果ear值小于0.2，则认为眼睛闭合，并开始计时闭眼时间
            close += 0.1
        else:
            if counter >= 2:  # 两帧以上都在闭眼，则算作一次眨眼，并暂停画面0.3秒
                total += 1
                time.sleep(0.3)
                t += 0.3
                save = 0
                close = 0
            counter = 0
        save += 0.1
        t += 0.1
        if save >= 8:  # 如果超过8秒不眨眼，则认为视觉疲劳，增加危险等级（详解见文档）
            mode += 2
            save = 0
        if close >= 3:  # 如果闭眼超过3秒，则认为疲劳，增加危险等级（详解见文档）
            mode += 2
            close = 0
        if mode >= 3:  # 不同程度的危险等级判定
            form = "warn3"
        if mode == 2:
            form = "warn2"
        if mode == 1:
            form = "warn1"
        if mode == 0:
            form = "save"
        # 当计算完成后，展示计算的数据
        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "time: {}".format(int(t)), (220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "mode: {}".format(form), (430, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "NoBlink: {}".format(int(save)), (640, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "close: {}".format(int(close)), (850, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 将fps保持在前台展示
    cv2.putText(frame, "fps: {}".format(int(fps)), (220, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 重新进行脸部检测，使用函数提取脸部roi并计算心率
    faces = detector(frame, 0)
    if len(faces) == 1:  # 在仅有一张脸时进行
        face_points = predictor(frame, faces[0])
        roi_avg = get_roi_avg(frame, face_points)
        roi_avg_values.append(roi_avg)
        storage.append(time.time())
        if len(storage) > 500:  # 当数据总量超过需求时，删除多余的数据
            roi_avg_values.pop(0)
            storage.pop(0)
        curr_buffer_size = len(storage)
        if curr_buffer_size > 100:
            temp = storage[-1] - storage[0]
            fps = curr_buffer_size / temp
            graph_values.append(roi_avg_values[-1])
            if len(graph_values) > 50:
                graph_values.pop(0)
                bpm = set_bpm(roi_avg_values, fps, curr_buffer_size, last_bpm)  # 调用函数计算bpm
                pct = int(bpm)
                if pct >= 110:  # 如果pct超过110（即心率超过110）判定为疲劳，增加危险等级（详解见文档）
                    mode += 2
        else:
            pct = str(int(round(float(curr_buffer_size) / 100 * 100.0))) + "%"  # 当数据不够计算心率时，以百分比的形式显示进度
    else:
        pct = "Nofind"  # 当检测不到人脸，或人脸不只一张时显示无法找到
    # 展示计算出的心率（无结果时按特定方式展示）
    cv2.putText(frame, "heart: {}".format(pct), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("text", frame)  # 展示视频
    if cv2.waitKey(25) & 0xFF == ord('q'):  # 按q键退出视频
        break
# 释放资源，结束
cap.release()
cv2.destroyAllWindows()
