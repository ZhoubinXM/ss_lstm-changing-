from scipy.spatial import distance
import numpy as np
import tensorflow as tf
import os
import heapq
import pickle
import matplotlib.pyplot as plt
import time
import cv2
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, LSTM, GRU, Merge
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.merge import Concatenate
from keras.models import load_model
from keras.layers import merge
from keras.layers.core import Permute
from keras.layers.core import RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import History
from keras.utils import plot_model, multi_gpu_model
from keras import backend as k
from keras.applications.vgg16 import VGG16
from keras import optimizers

from train.utils import circle_group_model_input, log_group_model_input, group_model_input
from train.utils import get_traj_like, get_obs_pred_like
from train.utils import person_model_input, model_expected_ouput, preprocess
from train.vehicle_utils import preprocess_vehicle,get_traj_like_vehicle,get_obs_vehicle_like,vehicle_model_input,\
    vehicle_model_expected_ouput,veh2ped_group_model_input,veh2ped_circle_group_model_input

import train.data_process as dp
import train.veh_data_process as vdp


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 只有编号为1的GPU对程序是可见的，在代码中gpu[0]指的就是这块儿GPU
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.5
k.tensorflow_backend.set_session(tf.Session(config=config))


def calculate_FDE(test_label, predicted_output, test_num, show_num):
    total_FDE = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        total_FDE[i] = distance.euclidean(predicted_result_temp[-1], label_temp[-1])  # 计算两个阵列之间的欧几里德距离

    show_FDE = heapq.nsmallest(show_num, total_FDE)   # 在total_FDE中找到最小的show_num的数
    show_FDE = np.reshape(show_FDE, [show_num, 1])

    return np.average(show_FDE)

def calculate_ADE(test_label, predicted_output, test_num, predicting_frame_num, show_num):
    total_ADE = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        ADE_temp = 0.0
        for j in range(predicting_frame_num):
            ADE_temp += distance.euclidean(predicted_result_temp[j], label_temp[j])
        ADE_temp = ADE_temp / predicting_frame_num
        total_ADE[i] = ADE_temp

    show_ADE = heapq.nsmallest(show_num, total_ADE)
    show_ADE = np.reshape(show_ADE, [show_num, 1])

    return np.average(show_ADE)

def CNN(img_rows, img_cols, img_channels=3):
    model = Sequential()
    img_shape = (img_rows, img_cols, img_channels)
    model.add(Conv2D(96, kernel_size=11, strides=4, input_shape=img_shape, padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=5, strides=1, padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(BatchNormalization(momentum=0.8))
    #    model.add(Conv2D(384, kernel_size=3, strides=1, padding="same"))
    #    model.add(Conv2D(384, kernel_size=3, strides=1, padding="same"))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))

    return model

# Parameter's Define
observed_frame_num = 8
predicting_frame_num = 12
train_data_num = 2  # 训练的数据

hidden_size = 128
tsteps = observed_frame_num#8
batch_size = 20

dimensions_1 = [720, 576] #[720,576] #eth hotel
# dimensions_1 = [224, 224]
dimensions_2 = [640, 480]  #eth univ
veh_neighborhood_size=128
grid_veh_size=4
neighborhood_size = 32
grid_size = 4
neighborhood_radius = 32
grid_radius = 4
# grid_radius_1 = 4
grid_angle = 45
circle_map_weights = [1, 2, 3, 4, 4, 3, 2, 1]
circle_map_weights=np.array(circle_map_weights)

opt = optimizers.RMSprop(lr=0.003)

# Starting load the datadut

PersonInput=[]
PersonExpectOutput=[]
Veh2PedInput=[]
Veh2PedExpectOutput=[]
GroupInput=[]

for datadir in range(train_data_num):
    data_dir=r'C:\Users\asus\Desktop\lstm\ss_lstm(changing)\datadut\0%s' %(datadir+1)
    veh_data, numveh = preprocess_vehicle(data_dir)
    raw_data, numPeds = preprocess(data_dir)
    check_veh = vdp.veh_DataProcesser(data_dir, observed_frame_num, predicting_frame_num)
    check = dp.DataProcesser(data_dir, observed_frame_num, predicting_frame_num)
    obs_veh = check_veh.obs_veh
    obs = check.obs
    pred_veh = check_veh.pred_veh
    pred = check.pred

    vehicle_input_raw = vehicle_model_input(obs_veh, observed_frame_num)
    person_input_raw = person_model_input(obs, observed_frame_num)
    group_circle_raw = circle_group_model_input(obs, observed_frame_num, neighborhood_size, dimensions_1,
                                              neighborhood_radius, grid_radius, grid_angle, circle_map_weights,
                                              raw_data)  # 圆形
    group_grid_1 = group_model_input(obs, observed_frame_num, neighborhood_size, dimensions_1, grid_size,
                                     raw_data)  # 矩形区域只写一个做存档
    # 车辆对行人的影响数据
    gruop_grid_veh2ped_raw = veh2ped_circle_group_model_input(obs, observed_frame_num, dimensions_1,
                                                            veh_neighborhood_size, grid_radius, grid_angle, raw_data,
                                                            veh_data,circle_map_weights)  # 圆形区域，若要矩形区域改成veh2ped_grid_model_input
    vehicle_expect_output_raw = vehicle_model_expected_ouput(pred_veh, predicting_frame_num) # 车辆
    expected_ouput_raw = model_expected_ouput(pred, predicting_frame_num)  # 行人

    group_input_raw = group_circle_raw
    veh2ped_group_input_raw = gruop_grid_veh2ped_raw

    PersonInput.append(person_input_raw)
    PersonExpectOutput.append(expected_ouput_raw)
    GroupInput.append(group_input_raw)
    Veh2PedInput.append(veh2ped_group_input_raw)
    Veh2PedExpectOutput.append(vehicle_expect_output_raw)


def all_run(epochs, predicting_frame_num, min_loss):

    person_input = np.concatenate(PersonInput)
    expected_ouput = np.concatenate(PersonExpectOutput)
    group_input = np.concatenate(GroupInput)
    vehicle_expect_output = np.concatenate(Veh2PedExpectOutput)
    veh2ped_group_input = np.concatenate(Veh2PedInput)
    # test_input = [group_input_10, person_input_10, veh2ped_group_input_10]
    # test_output = expected_ouput_10

    scene_scale = CNN(dimensions_1[1], dimensions_1[0])
    scene_scale.add(RepeatVector(tsteps))
    scene_scale.add(GRU(hidden_size,
                        input_shape=(tsteps, 512),
                        batch_size=batch_size,#all_run(epochs, predicting_frame_num, leave_dataset_index, map_index, show_num, min_loss)
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2))

    group_model = Sequential()
    group_model.add(Dense(hidden_size, activation='relu', input_shape=(tsteps, 64)))#全连接层
    group_model.add(GRU(hidden_size,
                        input_shape=(tsteps, int(neighborhood_radius / grid_radius) * int(360 / grid_angle)),
                        batch_size=batch_size,
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2))
    veh2ped_group_model = Sequential()
    veh2ped_group_model.add(Dense(hidden_size, activation='relu', input_shape=(tsteps, 256)))  # 全连接层
    veh2ped_group_model.add(GRU(hidden_size,
                        input_shape=(tsteps, int(veh_neighborhood_size / grid_radius) * int(360 / grid_angle)),
                        batch_size=batch_size,
                        return_sequences=False,
                        stateful=False,
                        dropout=0.2))
    person_model = Sequential() # 定义时间序列
    person_model.add(Dense(hidden_size, activation='relu', input_shape=(tsteps, 2)))
    person_model.add(GRU(hidden_size,
                         input_shape=(tsteps, 2),
                         batch_size=batch_size,
                         return_sequences=False,
                         stateful=False,
                         dropout=0.2))


    #myself
    vehicle_model=Sequential()
    vehicle_model.add(Dense(hidden_size, activation='relu', input_shape=(tsteps, 2)))
    vehicle_model.add(GRU(hidden_size,
                         input_shape=(tsteps, 2),
                         batch_size=batch_size,
                         return_sequences=False,
                         stateful=False,
                         dropout=0.2))
    # model.add(Merge([scene_scale, group_model, person_model], mode='sum'))
    model = Sequential()
    model.add(Merge([ group_model,
                      person_model,veh2ped_group_model], mode='sum'))
    model.add(RepeatVector(predicting_frame_num))
    model.add(GRU(128,
                input_shape=(predicting_frame_num, 2),
                batch_size=batch_size,
                return_sequences=True,
                stateful=False,
                dropout=0.2))
    model.add(TimeDistributed(Dense(2)))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer=opt)
    # parallel=multi_gpu_model(model,gpus=2)
    print(model.summary())
    for i in range(epochs):
        # history = model.fit([scene_input, group_input, person_input], expected_ouput,
                            #batch_size=batch_size,
                            #epochs=1,
                            #verbose=1,
                            #shuffle=False)
        history = model.fit([group_input, person_input,veh2ped_group_input], expected_ouput,
                            batch_size=batch_size,
                            epochs=1,
                            verbose=1,
                            shuffle=False)
        loss = history.history['loss']
        if loss[0] < min_loss:
            break
        else:
            continue
        model.reset_states()
        # parallel.reset_states()


    # model.save('ss_map_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'testing_seg.h5')
    # plot_model(model, to_file='model.png')

    # model.save('testing_SS_LSTM_logmap_Zara1_1000epoc_batchsize_20.h5')
    model.save_weights("model_weights_1000epochs_090402.h5")
    print('The model has been saved in h5 files')
    print('Now u can predicted the reslut for runing the predicting.py pythonfile.')


all_run(1000, predicting_frame_num, 0)















