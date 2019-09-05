# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:50:59 2017

@author: 21992674
"""
import numpy as np
import os
import math

from occupancy import get_rectangular_occupancy_map
#from occupancy import NYGC_rectangular_occupancy_map
from occupancy import get_circle_occupancy_map, log_circle_occupancy_map,get_veh_rectangular_occupancy_map,get_veh_circle_occupancy_map
from train.utils import person_model_input


# NYGC processing
# def file2matrix(filename):
#     data = np.loadtxt(filename, dtype=int)
#     data = np.reshape(data, [-1, 3])
#     return data


# def get_coord_from_txt(filename, ped_ID):
#     data = file2matrix(filename)
#     coord = []
#     for i in range(len(data)):
#         coord.append([ped_ID, data[i][-1], data[i][0], data[i][1]])
#     coord = np.reshape(coord, [-1, 4])
#     return coord


# def select_trajectory(data, frame_num):
#     if len(data) >= frame_num:
#         return True
#     else:
#         return False


# def get_all_trajectory(total_pedestrian_num):
#     data = []
#
#     for i in range(total_pedestrian_num):
#         filename = str(i + 1).zfill(6) + '.txt'
#         filepath = './data/ETHhotel/annotation/' + filename
#         ped_ID = i + 1
#         data.append(get_coord_from_txt(filepath, ped_ID))

    # return data


def preprocess_vehicle(data_dir):
    file_path = os.path.join(data_dir, 'roundabout_traj_veh_filtered.csv')
    # # file_path = data_dir + ''+'pixel_pos.csv'
    data = np.genfromtxt(file_path, delimiter=',')
    #xinjia

    data = np.transpose(data)
    data=[data[12,1:], data[1,1:], data[14,1:], data[15,1:]]
    x = 2 * (data[3] - min(data[3])) / (max(data[3]) - min(data[3])) - 1
    y = 2 * (data[2] - min(data[2])) / (max(data[2]) - min(data[2])) - 1
    numvehicles = np.size(np.unique(data[1]))#wogaileyixia
    ranges_x=(max(data[3]) - min(data[3]))
    ranges_y=(max(data[2]) - min(data[2]))
    min_x= min(data[3])
    min_y=min(data[2])

    data = [data[0], data[1], y, x]

    return data, numvehicles,ranges_x,ranges_y,min_x,min_y

def get_traj_like_vehicle(data, numvehicles):
    '''
    reshape data format from [frame_ID, ped_ID, y-coord, x-coord]
    to pedestrian_num * [ped_ID, frame_ID, x-coord, y-coord]
    '''
    traj_data_vehicle = []

    for vehIndex in range(numvehicles):
        traj = []
        for i in range(len(data[:,1])):
            if data[i][1] == vehIndex + 1:
                #wogaileyixia
                traj.append([data[1][i], data[0][i], data[-1][i], data[-2][i]])
        traj = np.reshape(traj, [-1, 4])

        traj_data_vehicle.append(traj)

    return traj_data_vehicle


# def get_traj_like_pixel(data, numPeds, dimension):
#     '''
#     reshape data format from [frame_ID, ped_ID, y-coord, x-coord]
#     to pedestrian_num * [ped_ID, frame_ID, x-coord, y-coord]
#     '''
#     traj_data = []
#     a = dimension[0]
#     b = dimension[1]
#
#     for pedIndex in range(numPeds):
#         traj = []
#         for i in range(len(data[:,1])):
#             if data[1][i] == pedIndex + 1:
#                 traj.append([data[1][i], data[0][i], data[-1][i] * a, data[-2][i] * b])
#         traj = np.reshape(traj, [-1, 4])
#
#         traj_data.append(traj)
#
#     return traj_data

def get_obs_vehicle_like(data, observed_frame_num, predicting_frame_num):
    """
    get input observed data and output predicted data
    """

    obs = []
    pred = []
    count = 0

    for vehIndex in range(len(data[:,1])):

        if len(data[:,vehIndex]) >= observed_frame_num + predicting_frame_num:
            obs_vehIndex = []
            pred_vehIndex = []
            count += 1
            for i in range(observed_frame_num):
                obs_vehIndex.append(data[vehIndex][i])
            for j in range(predicting_frame_num):
                pred_vehIndex.append(data[vehIndex][j + observed_frame_num])

            obs_pedIndex = np.reshape(obs_vehIndex, [observed_frame_num, 4])
            pred_pedIndex = np.reshape(pred_vehIndex, [predicting_frame_num, 4])

            obs.append(obs_vehIndex)
            pred.append(pred_vehIndex)

    obs_veh = np.reshape(obs, [count, observed_frame_num, 4])
    pred_veh = np.reshape(pred, [count, predicting_frame_num, 4])

    return obs_veh, pred_veh
#改到这里啦！

def vehicle_model_input(obs, observed_frame_num):
    vehicle_model_input = []
    for vehIndex in range(len(obs)):
        vehicle_vehIndex = []
        for i in range(observed_frame_num):
            vehicle_vehIndex.append([obs[vehIndex][i][-2], obs[vehIndex][i][-1]])
        vehicle_vehIndex = np.reshape(vehicle_vehIndex, [observed_frame_num, 2])

        vehicle_model_input.append(vehicle_vehIndex)

    vehicle_model_input = np.reshape(vehicle_model_input, [len(obs), observed_frame_num, 2])

    return vehicle_model_input


def vehicle_model_expected_ouput(pred, predicting_frame_num):
    vehicle_model_expected_ouput = []
    for vehIndex in range(len(pred)):
        vehicle_vehIndex = []
        for i in range(predicting_frame_num):
            vehicle_vehIndex.append([pred[vehIndex][i][-2], pred[vehIndex][i][-1]])
        vehicle_vehIndex = np.reshape(vehicle_vehIndex, [predicting_frame_num, 2])

        vehicle_model_expected_ouput.append(vehicle_vehIndex)

    vehicle_model_expected_ouput = np.reshape(vehicle_model_expected_ouput, [len(pred), predicting_frame_num, 2])

    return vehicle_model_expected_ouput

#改到这里啦！
def veh2ped_group_model_input(obs_ped, observed_frame_num, neighborhood_size, dimensions, grid_size, raw_data,vehicle_data,):
    veh2ped_group_model_input = []
    vehicle=vehicle_model_input
    ped=person_model_input

    for pedIndex in range(len(obs_ped)):
        group_vehIndex = []
        for i in range(observed_frame_num):
            o_map_vehIndex = get_veh_rectangular_occupancy_map(obs_ped[pedIndex][i][1], obs_ped[pedIndex][i][0], dimensions,
                                                           neighborhood_size, grid_size, raw_data,vehicle_data)
            o_map_vehIndex = np.reshape(o_map_vehIndex, [int(neighborhood_size / grid_size) ** 2, ])
            group_vehIndex.append(o_map_vehIndex)
        group_vehIndex = np.reshape(group_vehIndex, [observed_frame_num, int(neighborhood_size / grid_size) ** 2])

        veh2ped_group_model_input.append(group_vehIndex)


    veh2ped_group_model_input = np.reshape(veh2ped_group_model_input, [-1, observed_frame_num, int(neighborhood_size / grid_size) ** 2])

    return veh2ped_group_model_input


def veh2ped_circle_group_model_input(obs, observed_frame_num,  dimensions, neighborhood_radius, grid_radius,
                             grid_angle, raw_data,vehicle_data,circle_map_weights):                  # ,circle_map_weights
    veh2ped_group_model_input = []

    for vehIndex in range(len(obs)):
        group_vehIndex = []
        for i in range(observed_frame_num):
            o_map_vehIndex = get_veh_circle_occupancy_map(obs[vehIndex][i][1], obs[vehIndex][i][0], dimensions,
                                                      neighborhood_radius, grid_radius, grid_angle, raw_data,vehicle_data)
            o_map_vehIndex = np.reshape(o_map_vehIndex, [-1, ])
            group_vehIndex.append(o_map_vehIndex)
        group_vehIndex = np.reshape(group_vehIndex, [observed_frame_num, -1])

        veh2ped_group_model_input.append(group_vehIndex)
        veh2ped_group_model_input=veh2ped_group_model_input

    veh2ped_group_model_input = np.reshape(veh2ped_group_model_input, [len(veh2ped_group_model_input), observed_frame_num, -1])

    return veh2ped_group_model_input


def log_group_model_input(obs, observed_frame_num, neighborhood_size, dimensions, neighborhood_radius, grid_radius,
                          grid_angle, circle_map_weights, raw_data):
    group_model_input = []

    for pedIndex in range(len(obs)):
        group_pedIndex = []
        for i in range(observed_frame_num):
            o_map_pedIndex = log_circle_occupancy_map(obs[pedIndex][i][1], obs[pedIndex][i][0], dimensions,
                                                      neighborhood_radius, grid_radius, grid_angle, raw_data)
            o_map_pedIndex = np.reshape(o_map_pedIndex, [-1, ])
            group_pedIndex.append(o_map_pedIndex)
        group_pedIndex = np.reshape(group_pedIndex, [observed_frame_num, -1])

        group_model_input.append(group_pedIndex)

    group_model_input = np.reshape(group_model_input, [len(group_model_input), observed_frame_num, -1])

    return group_model_input

