from scipy.spatial import distance
import numpy as np
import tensorflow as tf
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
from keras import optimizers
import time
import cv2
from train.utils import circle_group_model_input, log_group_model_input, group_model_input
from train.utils import get_traj_like, get_obs_pred_like
from train.utils import person_model_input, model_expected_ouput, preprocess
from train.vehicle_utils import preprocess_vehicle,get_traj_like_vehicle,get_obs_vehicle_like,vehicle_model_input,vehicle_model_expected_ouput,veh2ped_group_model_input,veh2ped_circle_group_model_input
from keras.callbacks import History
import heapq
from train import data_process as dp
from train import veh_data_process as vdp
import predicting_vdp as pvdp
import predicting_dp as pdp
import os
from keras.utils import plot_model, multi_gpu_model
import pickle
import matplotlib.pyplot as plt
from keras import backend as k
from keras.applications.vgg16 import VGG16
from vizualize_trajectories import visualize_trajectories

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.5
k.tensorflow_backend.set_session(tf.Session(config=config))

def calculate_FDE(test_label, predicted_output, test_num, show_num):
    total_FDE = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        total_FDE[i] = distance.euclidean(predicted_result_temp[-1], label_temp[-1])  # 欧式距离公式 (两点间的距离公式)

    show_FDE = heapq.nsmallest(show_num, total_FDE)  # 取total_FDE里面最小的show_num的数据

    show_FDE = np.reshape(show_FDE, [show_num, 1])

    return np.average(show_FDE)                  # 计算数据的平均值

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

# Parameters Defination
observed_frame_num = 8
predicting_frame_num = 4
hidden_size = 128
tsteps = observed_frame_num#8
dimensions_1 = [720, 576] #[720,576] #eth hotel
# dimensions_1 = [224, 224]
dimensions_2 = [640, 480]  #eth univ
img_width_1 = 720
img_height_1 = 576
img_width_2 = 640
img_height_2 = 480
batch_size = 20
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


# Predicting Datadut Loading
data_dir_10=r'C:\Users\asus\Desktop\lstm\ss_lstm(changing)\datadut\010'
# data_dir_10=r'/home/lianli/zhoubin_work/zhoubin_work/lstm项目/ss_lstm_0715/datadut/010'
veh_data_10,numveh_10=preprocess_vehicle(data_dir_10)

# use to anti-normalization
raw_data_10, numPeds_10,ranges_x,ranges_y,min_x,min_y = preprocess(data_dir_10)

check_veh_10=pvdp.veh_DataProcesser(data_dir_10,observed_frame_num,predicting_frame_num)
check_10 = pdp.DataProcesser(data_dir_10,observed_frame_num,predicting_frame_num)
obs_veh_10=check_veh_10.obs_veh
obs_10=check_10.obs
pred_veh_10=check_veh_10.pred_veh
pred_10=check_10.pred

vehicle_input_10=vehicle_model_input(obs_veh_10,observed_frame_num)
person_input_10 = person_model_input(obs_10, observed_frame_num)
group_circle_10 = circle_group_model_input(obs_10, observed_frame_num, neighborhood_size, dimensions_1,neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data_10)
gruop_grid_veh2ped_10=veh2ped_circle_group_model_input(obs_10, observed_frame_num, dimensions_1, veh_neighborhood_size,grid_radius, grid_angle,raw_data_10,veh_data_10,circle_map_weights)
vehicle_expect_output_10= vehicle_model_expected_ouput(pred_veh_10, predicting_frame_num)
expected_ouput_10 = model_expected_ouput(pred_10, predicting_frame_num)



def all_run(epochs, predicting_frame_num, leave_dataset_index, map_index, show_num, min_loss):
    group_input_10 = group_circle_10
    veh2ped_group_input_10 = gruop_grid_veh2ped_10
    test_input = [group_input_10, person_input_10, veh2ped_group_input_10]
    test_output = expected_ouput_10
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
    person_model = Sequential()#定义时间序列
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
    model.load_weights("model_weights_1000epochs_07150203.h5")
    print('Predicting...')
    predicted_output = model.predict(test_input, batch_size=batch_size)

    # shuchujieguo dao txt files
    # data = open("person.txt","w+")
    print(predicted_output)
    print('text_output')
    print(test_output)
    print('test_input')
    print(person_input_10)

    # anti-normalization
    for list_1 in predicted_output:
        for i in list_1:
            i[0]=((i[0]+1)*ranges_x)/2+min_x
            i[1] = ((i[1] + 1) * ranges_y) / 2 + min_y

    for list_2 in test_output:
        for j in list_2:
            j[0]=((j[0]+1)*ranges_x)/2+min_x
            j[1]=((j[1]+1)*ranges_y)/2+min_y

    # data.close()
    # with open("person.txt","w") as fi:
    #     for k in person_input_10:
    #         fi.write('test_input'+'\n')
    #         fi.write(str(k) + '\n')
    #     for j in test_output:
    #         fi.write('test_output'+'\n')
    #         fi.write(str(j)+'\n')
    #     for i in predicted_output:
    #         fi.write('perdicted_output' + '\n')
    #         fi.write(str(i) + '\n')
    #
    #     print('The output has been saved in the txt files.')

    # np.savetxt('person.txt',(person_input_10),fmt="%3D")

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------

    # x y 是包括检测和输出的所有数据
    x_concat_test = np.concatenate([person_input_10, test_output], axis=1)   # 按照行方向拼接矩阵  三维矩阵 见text.py
    x_concat_model = np.concatenate([person_input_10, predicted_output], axis=1)

    # 将预测坐标保存至person_n.txt文件
    writefile=open('person_n.txt','w')
    writefile.write('拼接的测试数据：\n\n'+x_concat_test)
    writefile.write('\n\n拼接的预测数据：\n\n'+x_concat_model)

    # out_fig_dir = os.path.join('/home/lianli/zhoubin_work/zhoubin_work/ss-lstm_0715', "figs")
    # os.makedirs(out_fig_dir, exist_ok=True)
    #
    # fig = visualize_trajectories(x_concat_test, x_concat_model,8, 12)
    # fig_file = "01_person_n.png"
    # fig.savefig(fig_file)
    # plt.close()

    print('Predicting Done!')
    print('Calculating Predicting Error...')
    mean_FDE = calculate_FDE(test_output, predicted_output, len(test_output), show_num)
    mean_ADE = calculate_ADE(test_output, predicted_output, len(test_output), 12, show_num)
    all_FDE = calculate_FDE(test_output, predicted_output, len(test_output), len(test_output))
    all_ADE = calculate_ADE(test_output, predicted_output, len(test_output), 12, len(test_output))
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'mean ADE:', mean_ADE)
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'mean FDE:', mean_FDE)
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'all ADE:', all_ADE)
    print('ssmap_' + str(map_index) + '_ETHUCY_' + str(leave_dataset_index) + 'all FDE:', all_FDE)

    # 输出误差结果到txt文件
    with open('AFDE.txt','w+') as f:
        f.write('mean_ADE:\n'+str(mean_ADE)+'\n\n')
        f.write('mean_FDE:\n'+str(mean_FDE)+'\n\n')
        f.write('all_ADE:\n'+str(all_ADE)+'\n\n')
        f.write('all_FDE:\n'+str(all_FDE)+'\n\n')
# 轨迹可视化(画图)
    for i in range (len(test_output)):
        plt.plot(predicted_output[i][:, 0], predicted_output[i][:, 1], "y+")
        plt.plot(test_output[i][:, 0], test_output[i][:, 1], "g+")
        plt.plot(person_input_10[i][:,0],person_input_10[i][:,1],"r+")

    plt.title('')
    plt.legend()
    plt.show()


    # for i in range (len(x_concat_model)):
    #     plt.plot(x_concat_model[i][:, 0], x_concat_model[i][:, 1], "y+", label='predicted_SS_LSTM')
    #     plt.plot(x_concat_test[i][:, 0], x_concat_test[i][:, 1], "g+", label='test_SS_LSTM')
    # plt.title(i)
    # plt.legend()
    # plt.show()

    return predicted_output, mean_ADE, mean_FDE, all_ADE, all_FDE


all_run(1000, predicting_frame_num, 0, 1, 1, 0)