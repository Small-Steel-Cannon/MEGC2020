import os
import cv2
import dlib
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import datetime
from RWCSV import rw_csv

detector_face_cut = cv2.CascadeClassifier('F:/data/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('F:/data/shape_predictor_68_face_landmarks.dat')

def face_cut(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector_face_cut.detectMultiScale(gray, 1.1, 5)
    c = 1
    y = 1.1
    while len(faces) == 0 or faces[0][2] * faces[0][3] < 200 * 200:
        print('未检测到人脸')
        if c % 2 == 1:
            x = 5
        else:
            x = 3
        faces = detector_face_cut.detectMultiScale(gray, y, x)
        c += 1
        if c > 10:
            y = 1.1
        if c > 20:
            y = 1.2
        if c > 30:
            y = 1.3
        if c > 40:
            y = 1.4
        if c > 50:
            y = 1.5
    print('检测到人脸')

    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]

    # temp_img = img[y:y + h - 1, x:x + w - 1]
    return x, y, w, h


def face_detector(frame):
    img_gray = frame
    img = frame
    rects = detector(img_gray, 0)
    face_key_point = np.empty([0, 1, 2], dtype=np.float32)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            # print(idx,pos)
            temp_point = np.empty([1, 1, 2], dtype=np.float32)
            temp_point[0, 0] = pos
            face_key_point = np.concatenate([face_key_point, temp_point])
    return face_key_point


def get_point(face_key_point, eyebrow_index):
    temp_point = []
    for i in range(len(eyebrow_index)):
        p = eyebrow_index[i]
        temp_point.append(face_key_point[p][0])
    return temp_point


def extractROI(flow, center, margin):
    ROI_mod = []
    ROI_angle = []
    ROI_flow = []
    for k in range(len(center)):
        x = int(center[k][0] - margin)
        y = int(center[k][1] - margin)
        for i in range(margin * 2):
            for j in range(margin * 2):
                v = flow[x + i][y + j]
                temp_m = np.sqrt(np.dot(np.transpose(v), v))
                temp_a = math.atan2(v[1], v[0])
                ROI_mod.append(temp_m)
                ROI_angle.append(temp_a)
                ROI_flow.append(v)
    return ROI_flow, ROI_mod, ROI_angle


def globalMovment(base_ROI):
    n = len(base_ROI)
    for i in range(n):
        temp_flow = base_ROI[i]
        v = np.sqrt(np.dot(np.transpose(temp_flow), temp_flow))
        if i == 0:
            sum_flow = temp_flow
            sum_mod = v
        else:
            sum_flow += temp_flow
            sum_mod += v
    one_norm = sum_flow / (math.sqrt(math.pow(sum_flow[0], 2) + math.pow(sum_flow[1], 2)))
    mod_mean = sum_mod / n
    global_movment = one_norm * mod_mean
    return global_movment


def removeGlobal(flow, global_movment):
    x = np.zeros_like(flow)
    T = np.full(x.shape, global_movment)
    flow -= T
    return flow


def angle_domin(domin_count):
    pi = np.pi
    dur = pi / (domin_count / 2)
    left = 0
    area = []
    for i in range(int(domin_count / 2)):
        right = left + dur
        area.append([left, right])
        left += dur
    left = -pi
    for i in range(int(domin_count / 2)):
        right = left + dur
        area.append([left, right])
        left += dur
    return area


def getMean(ROI_flow, ROI_mod, ROI_angle):
    domin_count = 6
    n = len(ROI_mod)
    area = angle_domin(domin_count)
    max = 0
    bin = 0
    v_sum = None
    c = 0

    for i in range(len(area)):
        mod_sum = 0
        flow_sum = np.array([0, 0], dtype=np.float32)
        count = 0

        if len(area[i]) == 2:
            left = area[i][0]
            right = area[i][1]
            for j in range(n):
                if left <= ROI_angle[j] < right:
                    count += 1
                    mod_sum += abs(ROI_mod[j])
                    flow_sum[0] = flow_sum[0] + ROI_flow[j][0]
                    flow_sum[1] = flow_sum[1] + ROI_flow[j][1]

        if len(area[i]) == 4:
            left1 = area[i][0]
            right1 = area[i][1]
            left2 = area[i][2]
            right2 = area[i][3]
            for k in range(n):
                if left1 <= ROI_angle[k] <= right1 or left2 <= ROI_angle[k] < right2:
                    count += 1
                    mod_sum += abs(ROI_mod[k])
                    flow_sum[0] = flow_sum[0] + ROI_flow[j][0]
                    flow_sum[1] = flow_sum[1] + ROI_flow[j][1]
        if mod_sum > max:
            max = mod_sum
            bin = i + 1
            v_sum = flow_sum
            c = count

    mod_mean = max / c
    angle_mean = math.atan2(v_sum[1], v_sum[0])
    return mod_mean, angle_mean, bin


def get_page(label_path):
    excel = pd.ExcelFile(label_path)
    pages = excel.sheet_names
    page0 = np.array(pd.read_excel(label_path, pages[0]))
    page1 = np.array(pd.read_excel(label_path, pages[1]))
    page2 = np.array(pd.read_excel(label_path, pages[2]))
    return page0, page1, page2


def generate_label(path, s, code, page0, page1, page2):
    cap = cv2.VideoCapture(path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    column0 = page1[:, 0:1].reshape((-1))
    s = int(s)
    idx = np.argwhere(column0 == s)
    first_idx = idx[0][0]
    convert_s = page1[first_idx][2]

    column0 = page2[:, 0:1].reshape((-1))
    idx = np.argwhere(column0 == int(code[1:]))
    first_idx = idx[0][0]
    convert_code = page2[first_idx][1]

    # print(s, convert_s, code, convert_code)
    column0 = page0[:, 0:1].reshape((-1))
    idx = np.argwhere(column0 == convert_s)
    column1 = page0[:, 1:2].reshape((-1))
    column1 = column1[idx[0][0]: idx[-1][0] + 1]
    base_idx = idx[0][0]
    for i in range(column1.shape[0]):
        column1[i] = column1[i][0:-2]
        if column1[i][-1] == '_':
            column1[i] = column1[i][0:-1]
    # print(column1)
    idx = np.argwhere(column1 == convert_code)
    express_list = []
    for i in range(idx.shape[0]):
        current = idx[i][0] + base_idx
        start = page0[current][2]
        peak = page0[current][3]
        end = page0[current][4]
        if end == 0:
            end = peak
        express_list.append([start, end])
    label_list = np.zeros(total_frame)
    for i in range(len(express_list)):
        start = express_list[i][0]
        end = express_list[i][1]
        for j in range(end - start + 1):
            label_list[start - 1 + j] = 1
    print(express_list)
    return label_list


def solve(video_path, temp_label):
    print(video_path)
    eyebrow_index = np.array([18, 19, 20, 23, 24, 25], dtype=np.int8)
    nose_index = np.array([30], dtype=np.int8)
    mouth_index = np.array([48, 51, 54, 57], dtype=np.int8)
    point_index = np.concatenate([eyebrow_index, nose_index])
    point_index = np.concatenate([point_index, mouth_index])
    base_index = np.array([28], dtype=np.int8)
    point_index = np.concatenate([point_index, base_index])

    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pre_frame = None
    frame = None
    margin = 8
    start = 0
    feature = None
    frame_count = total_frame

    for i in range(frame_count):
        ret, temp_frame = cap.read()
        if i == start:
            x, y, w, h = face_cut(temp_frame)
            temp_img = temp_frame[y:y + h, x:x + w]
            pre_frame = temp_img
            face_key_point = face_detector(temp_img)
        if i > start:
            temp_img = temp_frame[y:y + h, x:x + w]
            frame = temp_img
            face_key_point = face_detector(temp_img)

            flow = calcOpticalFlow(pre_frame, frame)
            center = get_point(face_key_point, point_index)
            base_center = get_point(face_key_point, base_index)
            base_ROI_flow, _, _ = extractROI(flow, base_center, margin)

            global_movment = globalMovment(base_ROI_flow)
            flow = removeGlobal(flow, global_movment)

            for j in range(len(center)):
                ROI_flow, ROI_mod, ROI_angle = extractROI(flow, [center[j]], margin)
                mod_mean, angle_mean, bin = getMean(ROI_flow, ROI_mod, ROI_angle)
                if j == 0:
                    m_a = np.array([mod_mean, angle_mean], dtype=np.float32)
                else:
                    m_a = np.concatenate([m_a, [mod_mean, angle_mean]])
            label = np.array([temp_label[i]])
            m_a = np.concatenate([m_a, label])
            m_a = np.array([m_a])
            if i == 1:
                feature = m_a
            if i > 1:
                feature = np.concatenate([feature, m_a])
            pre_frame = frame

    print(feature.shape)
    return feature


if __name__ == '__main__':
    label_path = ''
    base_path = ''
    dirs = os.listdir(base_path)
    page0, page1, page2 = get_page(label_path)

    for i in range(len(dirs)):
        dir_path = 'F:/feature/' + dirs[i]
        if not os.path.exists(dir_path):
            print('create ' + dir_path)
            os.mkdir(dir_path)
        current_path = base_path + dirs[i]
        files = os.listdir(current_path)
        count = 0
        if i == len(dirs)-1:
            for j in range(len(files)):
                count += 1
                print(count)
                s = files[j][0:2]
                code = files[j][3:7]
                current_path = base_path + dirs[i] + '/' + files[j]
                p0 = page0.copy()
                p1 = page1.copy()
                p2 = page2.copy()

                temp_label = generate_label(current_path, s, code, p0, p1, p2)
                temp_feature = solve(current_path, temp_label)

                current_save_path = dir_path + '/' + files[j][:-4] + '.xlsx'
                rw_csv.save_data_to_excel(temp_feature, current_save_path)

