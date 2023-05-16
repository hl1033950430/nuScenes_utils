import math
import numpy as np
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap


# 获取半径范围内的所有道路，离散化
def get_lanes_in_radius(x: float, y: float, radius: float,
                        discretization_meters: float,
                        map_api):
    lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    lanes = map_api.discretize_lanes(lanes, discretization_meters)

    return lanes


def rotate(angle, pointP, pointQ):
    x2, y2 = pointQ  # 旋转中心点
    x1, y1 = pointP  # 旋转前的点
    x = x2 + (x1 - x2) * math.cos(angle) - (y1 - y2) * math.sin(angle)
    y = y2 + (x1 - x2) * math.sin(angle) + (y1 - y2) * math.cos(angle)
    return [x, y]


def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


# 获取前进方向左边的点
def get_left_point(target_x, target_y, yaw, offset=4):
    # yaw = correct_yaw(yaw)
    x, y = target_x, target_y + offset
    x, y = rotate(yaw, (x, y), (target_x, target_y))
    return x, y


# 获取前进方向右边的点
def get_right_point(target_x, target_y, yaw, offset=4):
    # yaw = correct_yaw(yaw)
    x, y = target_x, target_y - offset
    x, y = rotate(yaw, (x, y), (target_x, target_y))
    return x, y


# 获取道路 lane 中，与 (x, y) 最近的点
def get_closet_pos(lane, x, y):
    if len(lane) == 0:
        return -1
    return np.argmin(np.array([(pos[0] - x) ** 2 + (pos[1] - y) ** 2 for i, pos in enumerate(lane)]))


# 获取车辆前方的道路，并离散化
def get_front_lane(nusc_map, target_x, target_y):
    result = []
    # 获取车辆附近所有的车道
    lanes = get_lanes_in_radius(target_x, target_y, 40, 1, nusc_map)
    target_lane = nusc_map.get_closest_lane(target_x, target_y, radius=2)
    # 车辆可能没在道路上
    if target_lane == '':
        return result
    target_lane = lanes[target_lane]
    result.extend(target_lane)
    # 车道，获取与车道相连的车道
    end_point_x, end_point_y = target_lane[-1][0], target_lane[-1][1]
    has_next = True
    while has_next:
        start_point_x, start_point_y, min_distance = -1, -1, 1
        next_lane = ''
        for lane in lanes.values():
            distance = (lane[0][0] - end_point_x) ** 2 + (lane[0][1] - end_point_y) ** 2
            if distance <= min_distance:
                start_point_x, start_point_y, min_distance = lane[0][0], lane[0][1], distance
                next_lane = lane
        if next_lane == '':
            has_next = False
        else:
            result.extend(next_lane)
            end_point_x, end_point_y = next_lane[-1][0], next_lane[-1][1]
    return result


# 获取附近的车道
def get_neighbor_lane(nusc_map, target_x, target_y, offset):
    # 获取当前车道的位置
    lane = get_front_lane(nusc_map, target_x, target_y)
    pos = get_closet_pos(lane, target_x, target_y)
    target_x, target_y, yaw = lane[pos][0], lane[pos][1], lane[pos][2]
    # 当前车道位置，附近的点
    x, y = offset(target_x, target_y, yaw)
    # 根据点获取最近的车道，要求与当前道路不同且方向与当前车道方向一致
    neighbor_lane = get_front_lane(nusc_map, x, y)
    return neighbor_lane


# 左侧的车道
def get_left_lane(nusc_map, target_x, target_y):
    return get_neighbor_lane(nusc_map, target_x, target_y, get_left_point)


# 右侧的车道
def get_right_lane(nusc_map, target_x, target_y):
    return get_neighbor_lane(nusc_map, target_x, target_y, get_right_point)


# 获取附近所有车辆
def get_surrounding_vehicle(helper, nusc_map, instance_token, sample_token):
    target_ann = helper.get_sample_annotation(instance_token, sample_token)
    target_x, target_y, target_yaw = target_ann['translation']
    result = {}
    min_distance = {}

    # 获取当前车辆所在的道路、左右侧的道路
    lane = get_front_lane(nusc_map, target_x, target_y)
    left_lane = get_left_lane(nusc_map, target_x, target_y)
    right_lane = get_right_lane(nusc_map, target_x, target_y)
    if len(lane) == 0:
        return result

    # 获取周围车辆
    threshold_within_lane = 2
    target_pos = get_closet_pos(lane, target_x, target_y)
    annotations = helper.get_annotations_for_sample(sample_token)
    annotations = [ann for ann in annotations if
                   ann['category_name'].startswith('vehicle') and ann['token'] != target_ann['token']]
    for ann in annotations:
        x, y, yaw = ann['translation']
        pos = get_closet_pos(lane, x, y)
        pos_on_left = get_closet_pos(left_lane, x, y)
        pos_on_right = get_closet_pos(right_lane, x, y)
        distance_to_lane = distance(x, y, lane[pos][0], lane[pos][1])
        distance_to_left_lane = distance(x, y, left_lane[pos_on_left][0], left_lane[pos_on_left][1])
        distance_to_right_lane = distance(x, y, right_lane[pos_on_right][0], right_lane[pos_on_right][1])

        # 前车
        if distance_to_lane <= threshold_within_lane and pos - target_pos >= 0:
            # 只获取最近的
            if min_distance.get('front') is None or min_distance.get('front') > (pos - target_pos):
                result['front'] = ann
                min_distance['front'] = pos - target_pos

        # 左前车
        if distance_to_left_lane <= threshold_within_lane and pos_on_left - target_pos >= 0:
            # 只获取最近的
            if min_distance.get('left_front') is None or min_distance.get('left_front') > (pos_on_left - target_pos):
                result['left_front'] = ann
                min_distance['left_front'] = pos_on_left - target_pos

        # 右前车
        if distance_to_right_lane <= threshold_within_lane and pos_on_right - target_pos >= 0:
            # 只获取最近的
            if min_distance.get('right_front') is None or min_distance.get('right_front') > (pos_on_right - target_pos):
                result['right_front'] = ann
                min_distance['right_front'] = pos_on_right - target_pos

        # 左后车
        if distance_to_left_lane <= threshold_within_lane and target_pos - pos_on_left >= 0:
            # 只获取最近的
            if min_distance.get('left_back') is None or min_distance.get('left_back') > (target_pos - pos_on_left):
                result['left_back'] = ann
                min_distance['left_back'] = target_pos - pos_on_left

        # 右后车
        if distance_to_right_lane <= threshold_within_lane and target_pos - pos_on_right >= 0:
            # 只获取最近的
            if min_distance.get('right_back') is None or min_distance.get('right_back') > (target_pos - pos_on_right):
                result['right_back'] = ann
                min_distance['right_back'] = target_pos - pos_on_right

    return result
