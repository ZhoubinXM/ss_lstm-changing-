
"""
Created on Mon Jul 31 14:23:47 2017

@author: Hao Xue
"""

import numpy as np
import math


def get_rectangular_occupancy_map(frame_ID, ped_ID, dimensions, neighborhood_size, grid_size, data):
    """
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    该功能计算每帧的每个行人的矩形占用图。
     此占用地图用于组级LSTM。
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)标量值表示所考虑的邻域的大小
        grid_size : Scalar value representing the size of the grid discretization (4)标量值表示网格离散化的大小
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    """
    #    width_bound, height_bound = neighborhood_size/(width*1.0), neighborhood_size/(height*1.0)
    #    width_grid_bound, height_grid_bound = grid_size/(width*1.0), grid_size/(height*1.0)

    o_map = np.zeros((int(neighborhood_size / grid_size), int(neighborhood_size / grid_size)))
    #    o_map = np.zeros((int(neighborhood_size/grid_size)**2))

    ped_list = []
    data=np.array(data)
    # search for all peds in the same frame
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])

    if len(ped_list) == 0:
        print('no pedestrian in this frame!')
    elif len(ped_list) == 1:
        print('only one pedestrian in this frame!')
        return o_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-1], ped_list[pedIndex][-2]
                width_low, width_high = current_x - neighborhood_size / 2, current_x + neighborhood_size / 2
                height_low, height_high = current_y - neighborhood_size / 2, current_y + neighborhood_size / 2
                current_index = pedIndex
        for otherIndex in range(len(ped_list)):
            if otherIndex != current_index:
                other_x, other_y = ped_list[otherIndex][-1], ped_list[otherIndex][-2]
                if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                    continue
                cell_x = int(np.floor((other_x - width_low) / grid_size))
                cell_y = int(np.floor((other_y - height_low) / grid_size))

                o_map[cell_x, cell_y] += 1
        #                o_map[cell_x + cell_y*grid_size] = 1

        return o_map


#
def cal_angle(current_x, current_y, other_x, other_y):
    p0 = [other_x, other_y]
    p1 = [current_x, current_y]
    p2 = [current_x + 0.1, current_y]
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle_degree = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return angle_degree

def get_veh_rectangular_occupancy_map(frame_ID, ped_ID, dimensions, veh_neighborhood_size, grid_veh_size, data,vehicle_data):
    """
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    该功能计算每帧的每个行人的矩形占用图。
     此占用地图用于组级LSTM。
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)标量值表示所考虑的邻域的大小
        grid_size : Scalar value representing the size of the grid discretization (4)标量值表示网格离散化的大小
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    """
    #    width_bound, height_bound = neighborhood_size/(width*1.0), neighborhood_size/(height*1.0)
    #    width_grid_bound, height_grid_bound = grid_size/(width*1.0), grid_size/(height*1.0)

    veh_o_map = np.zeros((int(veh_neighborhood_size / grid_veh_size), int(veh_neighborhood_size / grid_veh_size)))
    #    o_map = np.zeros((int(neighborhood_size/grid_size)**2))

    ped_list = []
    veh_list=[]
    data=np.array(data)
    vehicle_data=np.array(vehicle_data)
    # search for all peds in the same frame
    for i in range(len(vehicle_data[0])):
        if vehicle_data[0][i] == frame_ID:
            veh_list.append(vehicle_data[:, i])
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])
    # veh_list = np.reshape(veh_list, [-1, 4])

    if len(veh_list) == 0:
        print('no vehicle in this frame!')
        return veh_o_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-1], ped_list[pedIndex][-2]
                width_low, width_high = current_x - veh_neighborhood_size / 2, current_x + veh_neighborhood_size / 2
                height_low, height_high = current_y - veh_neighborhood_size / 2, current_y + veh_neighborhood_size / 2
                current_index = pedIndex
        for otherIndex in range(len(veh_list)):
                other_x, other_y = veh_list[otherIndex][-1], veh_list[otherIndex][-2]
                if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                    continue
                veh_cell_x = int(np.floor((other_x - width_low) / grid_veh_size))
                veh_cell_y = int(np.floor((other_y - height_low) / grid_veh_size))

                veh_o_map[veh_cell_x, veh_cell_y] += 1
        #                o_map[cell_x + cell_y*grid_size] = 1

        return veh_o_map


#

def get_veh_circle_occupancy_map(frame_ID, ped_ID, dimensions, neighborhood_radius, grid_radius, grid_angle, data, vehicle_data):
    '''
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    '''
    width, height = dimensions[0], dimensions[1]
    neighborhood_bound = neighborhood_radius / (min(width, height) * 1.0)
    grid_bound = grid_radius / (min(width, height) * 1.0)
    veh_o_map = np.zeros((int(neighborhood_radius / grid_radius), int(360 / grid_angle)))
    #    o_map = np.zeros((int(neighborhood_size/grid_size)**2))
    ped_list = []
    veh_list = []
    veh_list_last=[]
    list_last=[]
    data = np.array(data)
    f=[]
    vehicle_data = np.array(vehicle_data)

    # search for all peds in the same frame
    for i in range(len(vehicle_data[0])):
        if vehicle_data[0][i] == frame_ID:
            veh_list.append(vehicle_data[:, i])
        if vehicle_data[0][i] == 1:
            veh_list_last.append(vehicle_data[:, i])
        if vehicle_data[0][i-1] == frame_ID:
            veh_list_last.append(vehicle_data[:, i-1])
    for j in range(len(data[0])):
        if data[0][j] == frame_ID:
            ped_list.append(data[:, j])
        if data[0][j] == 1:
           list_last.append(vehicle_data[:, j])
        if data[0][j-1] == frame_ID:
           list_last.append(data[:, j-1])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])
    list_last = np.reshape(list_last, [-1, 4])
    veh_list = np.reshape(veh_list, [-1, 4])
    veh_list_last = np.reshape(veh_list_last, [-1, 4])
    if len(veh_list) == 0:
        print('no vehicle in this frame!')
        return veh_o_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-1], ped_list[pedIndex][-2]
                current_index = pedIndex
        for otherIndex in range(len(veh_list)):
            if otherIndex != current_index:
                other_x, other_y = veh_list[otherIndex][-1], veh_list[otherIndex][-2]
                other_distance = math.sqrt((other_x - current_x) ** 2 + (other_y - current_y) ** 2)
                angle = cal_angle(current_x, current_y, other_x, other_y)
                if other_distance >= neighborhood_bound:
                    continue
                last_x, last_y = list_last[pedIndex][-1], list_last[pedIndex][-2]
                other_x_last, other_y_last = veh_list_last[otherIndex][-1], veh_list[otherIndex][-2]
                other_distance_last = math.sqrt((other_x_last - last_x) ** 2 + (other_y_last - last_y) ** 2)
                f=(other_distance_last/other_distance)
                veh_cell_x = int(np.floor(other_distance / grid_bound))
                veh_cell_y = int(np.floor(angle / grid_angle))

                veh_o_map[veh_cell_x, veh_cell_y] += 1*f

        return veh_o_map

def get_circle_occupancy_map(frame_ID, ped_ID, dimensions, neighborhood_radius, grid_radius, grid_angle, data):
    '''
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    '''
    width, height = dimensions[0], dimensions[1]
    neighborhood_bound = neighborhood_radius / (min(width, height) * 1.0)
    grid_bound = grid_radius / (min(width, height) * 1.0)
    o_map = np.zeros((int(neighborhood_radius / grid_radius), int(360 / grid_angle)))
    #    o_map = np.zeros((int(neighborhood_size/grid_size)**2))
    ped_list = []
    fs=[]
    data = np.array(data)


    # search for all peds in the same frame
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])

    if len(ped_list) == 0:
        print('no pedestrian in this frame!')
    elif len(ped_list) == 1:
        print('only one pedestrian in this frame!')
        return o_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-1], ped_list[pedIndex][-2]
                current_index = pedIndex
        for otherIndex in range(len(ped_list)):
            if otherIndex != current_index:
                other_x, other_y = ped_list[otherIndex][-1], ped_list[otherIndex][-2]
                other_distance = math.sqrt((other_x - current_x) ** 2 + (other_y - current_y) ** 2)
                angle = cal_angle(current_x, current_y, other_x, other_y)
                if other_distance >= neighborhood_bound:
                    continue
                #改方向系数
                # if  data[0][i]==0:
                #     continue
                # last_x, last_y = ped_list[pedIndex-1][-1], ped_list[pedIndex-1][-2]
                # other_x_last, other_y_last = ped_list[otherIndex-1][-1], ped_list[otherIndex-1][-2]
                # other_distance_last = math.sqrt((other_x_last - last_x) ** 2 + (other_y_last - last_y) ** 2)
                # f=other_distance_last/other_distance
                # # fmin=min(f)
                # # fmax=max(f)
                # # fs=(f-fmin)/(fmax-fmin)
                cell_x = int(np.floor(other_distance / grid_bound))
                cell_y = int(np.floor(angle / grid_angle))

                o_map[cell_x, cell_y] += 1

        return o_map

def log_circle_occupancy_map(frame_ID, ped_ID, dimensions, neighborhood_radius, grid_radius, grid_angle, data):
    """
    This function computes occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    """
    width, height = dimensions[0], dimensions[1]
    o_map = np.zeros((8, 8))
    #    o_map = np.zeros((int(neighborhood_size/grid_size)**2))

    ped_list = []
    data=np.array(data)
    # search for all peds in the same frame
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])

    if len(ped_list) == 0:
        print('no pedestrian in this frame!')
    elif len(ped_list) == 1:
        print('only one pedestrian in this frame!')
        return o_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-1], ped_list[pedIndex][-2]
                current_index = pedIndex
        for otherIndex in range(len(ped_list)):
            if otherIndex != current_index:
                other_x, other_y = ped_list[otherIndex][-1], ped_list[otherIndex][-2]
                other_distance = math.sqrt(
                    (other_x * width - current_x * width) ** 2 + (other_y * height - current_y * height) ** 2)
                log_distance = math.log2(other_distance)
                angle = cal_angle(current_x, current_y, other_x, other_y)
                if other_distance >= 8:
                    continue
                cell_x = int(np.floor(log_distance))
                cell_y = int(np.floor(angle / grid_angle))

                o_map[cell_x, cell_y] += 1

        return o_map



