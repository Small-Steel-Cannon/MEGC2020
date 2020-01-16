import os
import numpy as np
from RWCSV import rw_csv
import datetime
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd


def Threshold_detection(mod_list, center_list, first_interval):
    n = center_list.shape[0]
    detect_list = []
    for i in range(n):
        center = center_list[i]
        if center + 1 < mod_list.shape[0] - 1 and center - 1 > 0:
            if mod_list[center] > 0.3 and mod_list[center + 1] - mod_list[center] < 0 and mod_list[center - 1] - \
                    mod_list[center] < 0:
                # ==========left==========
                left = center
                if left - 1 > 0:
                    while mod_list[left - 1] - mod_list[left] < 0:
                        if left - 1 > 0:
                            left -= 1
                        else:
                            break
                # ==========right=========
                right = center
                if right + 1 < mod_list.shape[0] - 1:
                    while mod_list[right + 1] - mod_list[right] < 0:
                        if right + 1 < mod_list.shape[0] - 1:
                            right += 1
                        else:
                            break
                len = right - left + 1
                if len <= 8:
                    detect_list.append(
                        [left + first_interval + 1, center + first_interval + 1, right + first_interval + 1,
                         mod_list[i]])

    detect_list = np.array(detect_list)
    return detect_list


def wave_detect(y, first_interval):
    threshold = [0.8, 1.5]
    max = np.max(y)
    if max < threshold[0]:
        return None
    threshold[0] = max * 0.8
    threshold[1] = max
    first_select = np.argwhere(y > threshold[0])
    index_list = []
    for i in range(first_select.shape[0]):
        idx = first_select[i][0]
        if y[idx] < threshold[1]:
            index_list.append(idx)
    max_list = np.array(index_list)
    detect_list = Threshold_detection(y, max_list, first_interval)
    return detect_list


def detect(feature, parameter):
    first_interval = 100
    feature = feature[:, 0:-1]
    line_count = feature.shape[1]
    Predict_list = []

    count = 0
    for i in range(line_count):
        if i % 2 == 0:
            current_line = feature[:, i:i + 1]
            current_line = current_line[first_interval:-first_interval]
            current_line = current_line.reshape(-1)
            y_hat = signal.savgol_filter(current_line, parameter, 3)
            detect_list = wave_detect(y_hat, first_interval)
            if not detect_list is None:
                Predict_list.append(detect_list)
    return Predict_list


def get_spot_list(y):
    first = False
    start = None
    end = None
    spot_list = []
    bias = 0
    for i in range(y.shape[0]):
        if y[i] == 1 and first == False:
            first = True
            start = i
        if y[i] == 0 and first == True:
            end = i - 1
            spot_list.append([start + bias + 1, end + bias + 1])
            first = False
        if i == y.shape[0] - 1 and first == True:
            end = i
            spot_list.append([start + 1 + bias, end + 1 + bias])
    return spot_list


def generate_label(Predict_list, predict_path, file_name, frame_count, current_label):
    y = np.zeros([frame_count, ], dtype=np.int8)
    for i in range(len(Predict_list)):
        current_predict = Predict_list[i]
        if current_predict.shape[0] > 30:
            continue
        for k in range(current_predict.shape[0]):
            son_predict = current_predict[k]
            left = int(son_predict[0])
            right = int(son_predict[2])
            length = right - left + 1
            for j in range(length):
                y[left + j - 1] = 1

    fig_path = predict_path + file_name + '/'
    fig_name = file_name + '_before' + '.jpg'
    draw(y, fig_name, fig_path)
    spot_list = get_spot_list(y)
    D = None
    pre_spot = None
    new_spot_list = []
    if len(spot_list) % 2 == 0:
        D = True
    else:
        D = False
    if D:
        for i in range(int(len(spot_list)/2)):
            idx = 0 + i*2
            pre_spot = spot_list[idx]
            current_spot = spot_list[idx+1]
            if current_spot[0] - pre_spot[1] <= 2:
                new_spot = [pre_spot[0], current_spot[1]]
                new_spot_list.append(new_spot)
            else:
                new_spot_list.append(pre_spot)
                new_spot_list.append(current_spot)
    else:
        for i in range(int((len(spot_list)-1) / 2)):
            idx = 0 + i*2
            pre_spot = spot_list[idx]
            current_spot = spot_list[idx+1]
            if current_spot[0] - pre_spot[1] <= 2:
                new_spot = [pre_spot[0], current_spot[1]]
                new_spot_list.append(new_spot)
            else:
                new_spot_list.append(pre_spot)
                new_spot_list.append(current_spot)
        new_spot_list.append(spot_list[-1])
    return new_spot_list


def get_page(label_path):
    excel = pd.ExcelFile(label_path)
    pages = excel.sheet_names
    page0 = np.array(pd.read_excel(label_path, pages[0]))
    page1 = np.array(pd.read_excel(label_path, pages[1]))
    page2 = np.array(pd.read_excel(label_path, pages[2]))
    return page0, page1, page2


def read_label(path, s, code, page0, page1, page2):
    column0 = page1[:, 0:1].reshape((-1))
    s = int(s)
    idx = np.argwhere(column0 == s)
    first_idx = idx[0][0]
    convert_s = page1[first_idx][2]

    column0 = page2[:, 0:1].reshape((-1))
    idx = np.argwhere(column0 == int(code[1:]))
    first_idx = idx[0][0]
    convert_code = page2[first_idx][1]

    column0 = page0[:, 0:1].reshape((-1))
    idx = np.argwhere(column0 == convert_s)
    column1 = page0[:, 1:2].reshape((-1))
    column1 = column1[idx[0][0]: idx[-1][0] + 1]
    base_idx = idx[0][0]
    for i in range(column1.shape[0]):
        column1[i] = column1[i][0:-2]
        if column1[i][-1] == '_':
            column1[i] = column1[i][0:-1]
    idx = np.argwhere(column1 == convert_code)
    express_list = []
    for i in range(idx.shape[0]):
        current = idx[i][0] + base_idx
        start = page0[current][2]
        peak = page0[current][3]
        end = page0[current][4]
        if end == 0:
            end = peak
        if start == -1 or peak == -1 or end == -1:
            continue
        express_list.append([start, end])

    return express_list


def analyse(label_list, spot_list):
    k_count = 0
    FN_count = 0
    TP_count = 0
    FP_count = 0
    TP_lsit = [0] * len(spot_list)
    FN_list = [0] * len(label_list)
    class_list = []
    main_item = None

    for i in range(len(spot_list)):
        tmp_spot = spot_list[i]
        for j in range(len(label_list)):
            tmp_label = label_list[j]
            if tmp_spot[1] >= tmp_label[1]:
                right_range = tmp_spot
                left_range = tmp_label
            if tmp_spot[1] <= tmp_label[1]:
                right_range = tmp_label
                left_range = tmp_spot
            flag = left_range[1] - right_range[0]

            if flag <= 0:
                continue

            if flag > 0:
                k_count += 1

                if left_range[0] >= right_range[0]:
                    inter_length = left_range[1] - left_range[0] + 1
                    union_length = right_range[1] - right_range[0] + 1
                if left_range[0] < right_range[0]:
                    inter_length = left_range[1] - right_range[0] + 1
                    union_length = right_range[1] - left_range[0]

                K = inter_length / union_length

                if K >= 0.5:
                    FN_list[j] = 1
                    tmp_label = label_list[j].copy()
                    tmp_label.append(tmp_spot[0])
                    tmp_label.append(tmp_spot[1])
                    tmp_label.append(K)
                    tmp_item = np.array([tmp_label])

                    if main_item is None:
                        main_item = tmp_item
                        class_list.append('TP')
                    else:
                        main_item = np.concatenate([main_item, tmp_item])
                        class_list.append('TP')
                    TP_count += 1
                    TP_lsit[i] = 1

                if K < 0.5:
                    tmp_label = label_list[j].copy()
                    tmp_label.append(tmp_spot[0])
                    tmp_label.append(tmp_spot[1])
                    tmp_label.append(K)
                    tmp_item = np.array([tmp_label])

                    if main_item is None:
                        main_item = tmp_item
                        class_list.append('FP')
                        FP_count += 1
                    else:
                        main_item = np.concatenate([main_item, tmp_item])
                        class_list.append('FP')
                        TP_lsit[i] = 1
                        FP_count += 1

    for i in range(len(label_list)):
        if FN_list[i] == 0:
            FN_count += 1
            tmp = label_list[i].copy()
            tmp.append(-1)
            tmp.append(-1)
            tmp.append(0)
            tmp = np.array([tmp])
            if main_item is None:
                main_item = tmp
                class_list.append('FN')
            else:
                main_item = np.concatenate([main_item, tmp])
                class_list.append('FN')

    if not (spot_list[0][0] == -1 and spot_list[0][1] == -1):
        for i in range(len(spot_list)):
            if TP_lsit[i] == 0:
                FP_count += 1
                tmp = spot_list[i].copy()
                tmp.insert(0, -1)
                tmp.insert(0, -1)
                tmp.append(0)
                tmp = np.array([tmp])
                if main_item is None:
                    main_item = tmp
                    class_list.append('FP')
                else:
                    main_item = np.concatenate([main_item, tmp])
                    class_list.append('FP')

    FN_count = FN_list.count(0)
    return TP_count, FP_count, FN_count, main_item, class_list, k_count


if __name__ == '__main__':
    label_path = ''
    base_path = ''
    predict_path = ''
    parameter_path = ''
    dirs = os.listdir(base_path)
    feature = None
    K_count = 0
    TP = 0
    FP = 0
    FN = 0
    total_P = 0
    DF = None
    parameter_count = -1
    no_spot = 0
    no_spot_list = []

    parameter_list = rw_csv.read_data_from_excel(parameter_path)
    parameter_list = parameter_list.reshape(-1)
    page0, page1, page2 = get_page(label_path)
    print(datetime.datetime.now())
    print('read feature...')
    feature_list = []
    for i in range(len(dirs)):
        dir_path = base_path + dirs[i]
        files = os.listdir(dir_path)
        for j in range(len(files)):
            parameter_count += 1
            file_path = dir_path + '/' + files[j]
            print(file_path)


            s = files[j][0:2]
            code = files[j][3:7]

            p0 = page0.copy()
            p1 = page1.copy()
            p2 = page2.copy()
            current_label = read_label(file_path, s, code, p0, p1, p2)
            total_P += len(current_label)

            current_feature = rw_csv.read_data_from_excel(file_path)

            Predict_list = detect(current_feature, parameter_list[parameter_count])
            spot_list = generate_label(Predict_list, predict_path, files[j][:-5], current_feature.shape[0],
                                       current_label)

            if not len(spot_list) == 0:
                current_TP, current_FP, current_FN, current_main_item, current_class_list, current_k_count = analyse(
                    current_label,
                    spot_list)
                TP += current_TP
                FP += current_FP
                FN += current_FP
                K_count += current_k_count
                print(current_TP, current_FP, current_FN)

                current_index = [files[j][:-5]] * current_main_item.shape[0]
                df = pd.DataFrame(current_main_item, index=current_index, columns=['T_1', 'T_2', 'P_1', 'P_2', 'K'])
                df.insert(len(df.columns), 'class', current_class_list)
                df["T_1"] = df["T_1"].astype("int")
                df["T_2"] = df["T_2"].astype("int")
                df["P_1"] = df["P_1"].astype("int")
                df["P_2"] = df["P_2"].astype("int")
                if DF is None:
                    DF = df
                else:
                    DF = pd.concat([DF, df], axis=0)

            else:
                spot_list = [[-1, -1]]
                current_TP, current_FP, current_FN, current_main_item, current_class_list, current_k_count = analyse(
                    current_label,
                    spot_list)
                if not current_main_item is None:
                    current_index = [files[j][:-5]] * current_main_item.shape[0]
                    df = pd.DataFrame(current_main_item, index=current_index, columns=['T_1', 'T_2', 'P_1', 'P_2', 'K'])
                    df.insert(len(df.columns), 'class', current_class_list)
                    df["T_1"] = df["T_1"].astype("int")
                    df["T_2"] = df["T_2"].astype("int")
                    df["P_1"] = df["P_1"].astype("int")
                    df["P_2"] = df["P_2"].astype("int")
                    if DF is None:
                        DF = df
                    else:
                        DF = pd.concat([DF, df], axis=0)

                    no_spot += 1
                    no_spot_list.append(files[j])

    P = total_P
    print('P', P)
    FN = P - TP
    print('TP:', TP)
    print('FP:', FP)
    print('FN:', FN)
    print('K_count', K_count)

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * recall * precision / (recall + precision)
    print('recall: ', recall)
    print('precision: ', precision)
    print('f1: ', f1)

    print('no_spot', no_spot)
    print('no_spot_list', no_spot_list)

    if not os.path.exists('F:/ZLW/predict/result/'):
        os.mkdir('F:/predict/result/')
        print('create', 'F:/predict/result/')
    save_path = 'F:/predict/result/' + 'spotting.xlsx'
    rw_csv.save_data_to_excel(DF, save_path)

