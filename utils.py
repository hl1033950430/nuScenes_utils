import math

import numpy as np
import torch.nn.functional
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

# 获取半径范围内的所有道路，离散化
from pyquaternion import Quaternion


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


def point_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


# 将坐标与目标车辆坐标系对齐
# 目标车辆的坐标系：以车辆中心点为原点，车辆前进方向为 y 轴
def align_coordinate(target_x, target_y, target_yaw, x, y, yaw):
    # 相对坐标
    relative_x = x - target_x
    relative_y = y - target_y
    # 旋转
    rotate_yaw = np.pi / 2 - target_yaw
    x, y = rotate(rotate_yaw, (relative_x, relative_y), (0, 0))
    yaw += rotate_yaw
    return x, y, yaw


def lane_with_align_coordinate(lane, target_x, target_y, target_yaw):
    return [align_coordinate(target_x, target_y, target_yaw, item[0], item[1], item[2]) for item in lane]


# 获取前进方向左边的点
def get_left_point(target_x, target_y, yaw, offset=3):
    # yaw = correct_yaw(yaw)
    x, y = target_x, target_y + offset
    x, y = rotate(yaw, (x, y), (target_x, target_y))
    return x, y


# 获取前进方向右边的点
def get_right_point(target_x, target_y, yaw, offset=3):
    # yaw = correct_yaw(yaw)
    x, y = target_x, target_y - offset
    x, y = rotate(yaw, (x, y), (target_x, target_y))
    return x, y


# 获取道路 lane 中，与 (x, y) 最近的点
def get_closest_pos(lane, x, y):
    if len(lane) == 0:
        return -1
    return np.argmin(np.array([(pos[0] - x) ** 2 + (pos[1] - y) ** 2 for i, pos in enumerate(lane)]))


def quaternion_yaw(q) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


# 根据 annotation 里的 rotation 字段，获取正确的角度值
def get_correct_yaw(rotation):
    return quaternion_yaw(Quaternion(rotation))


# 根据位置和角度获取最近的车道
def get_closest_lane(nusc_map, target_x, target_y, target_yaw=None, radius=2):
    min_distance = 9999
    min_yaw_distance = np.pi
    result = ''
    yaw_threshold = np.pi / 4
    # 获取车辆附近所有的车道
    lanes = get_lanes_in_radius(target_x, target_y, 40, 1, nusc_map)
    for key in lanes:
        lane = lanes[key]
        for index, pos in enumerate(lane):
            distance = point_distance(pos[0], pos[1], target_x, target_y)
            yaw_distance = abs(get_lane_start_yaw(lane[index:]) - target_yaw)
            if distance <= radius and (target_yaw is None or yaw_distance <= yaw_threshold):
                # 满足距离和角度
                if yaw_distance < min_yaw_distance:
                    # 且是角度最符合的
                    min_yaw_distance = yaw_distance
                    min_distance = distance
                    result = key
                elif yaw_distance == min_yaw_distance and distance < min_distance:
                    # 如果角度相等，考虑距离
                    min_distance = distance
                    result = key
    return result


# 获取车辆前方的道路，并离散化
def get_front_lane(nusc_map, x, y, yaw=None, _cache={}):
    key = 'get_front_lane_{}_{}_{}'.format(nusc_map.map_name, x, y)
    if key in _cache:
        return _cache[key]

    result = []
    # 获取车辆附近所有的车道
    lanes = get_lanes_in_radius(x, y, 60, 1, nusc_map)
    target_lane_token = get_closest_lane(nusc_map, x, y, yaw, radius=2)
    # 车辆可能没在道路上
    if target_lane_token == '':
        return result
    target_lane = lanes[target_lane_token]
    result.extend(target_lane)
    # 车道，获取与车道相连的车道
    end_point_x, end_point_y, end_point_yaw = target_lane[-1][0], target_lane[-1][1], target_lane[-1][2]
    has_next = True
    while has_next:
        start_point_x, start_point_y, min_distance, min_yaw_distance = -1, -1, 1, np.pi
        next_lane = ''
        # 寻找下一段车道的起点
        for lane in lanes.values():
            distance = (lane[0][0] - end_point_x) ** 2 + (lane[0][1] - end_point_y) ** 2
            # 角度取前3个离散点的平均角度
            yaw_distance = abs(get_lane_start_yaw(lane) - end_point_yaw)
            if distance <= 1 and yaw_distance <= (np.pi / 4) and yaw_distance <= min_yaw_distance:
                start_point_x, start_point_y, min_yaw_distance = lane[0][0], lane[0][1], yaw_distance
                next_lane = lane
        if next_lane == '':
            has_next = False
        else:
            result.extend(next_lane)
            end_point_x, end_point_y = next_lane[-1][0], next_lane[-1][1]

    _cache[key] = result
    return result


def get_front_lane_with_target_coordinate(nusc_map, x, y, yaw=None, target_x=0, target_y=0, target_yaw=0):
    front_lane = get_front_lane(nusc_map, x, y, yaw)
    return lane_with_align_coordinate(front_lane, target_x, target_y, target_yaw)


# 获取道路开头的角度，取前面几个点的平均角度
def get_lane_start_yaw(lane):
    if len(lane) == 0:
        return 0
    prefix = min(len(lane), 6)
    sum_yaw = 0
    for i in range(prefix):
        sum_yaw += lane[i][2]
    return sum_yaw / prefix


# 获取附近的车道
# x, y : 目标位置附近的车道
# target_lane_token : 目标车道旁边的车道
def get_neighbor_lane(nusc_map, x, y, current_lane, offset):
    if len(current_lane) == 0:
        return []
    # 获取当前车道的位置
    pos = get_closest_pos(current_lane, x, y)
    x, y, yaw = current_lane[pos][0], current_lane[pos][1], current_lane[pos][2]
    # 当前车道位置，附近的点
    x, y = offset(x, y, yaw)
    # 根据点获取最近的车道，要求与当前道路不同且方向与当前车道方向一致
    neighbor_lane = get_front_lane(nusc_map, x, y, yaw)
    return neighbor_lane


# 左侧的车道
def get_left_lane(nusc_map, x, y, current_lane, _cache={}):
    key = 'get_left_lane_{}_{}_{}'.format(nusc_map.map_name, x, y)
    if key in _cache:
        return _cache[key]

    result = get_neighbor_lane(nusc_map, x, y, current_lane, get_left_point)

    _cache[key] = result
    return result


def get_left_lane_with_target_coordinate(nusc_map, x, y, current_lane, target_x, target_y, target_yaw):
    left_lane = get_neighbor_lane(nusc_map, x, y, current_lane, get_left_point)
    return lane_with_align_coordinate(left_lane, target_x, target_y, target_yaw)


# 右侧的车道
def get_right_lane(nusc_map, x, y, current_lane, _cache={}):
    key = 'get_right_lane_{}_{}_{}'.format(nusc_map.map_name, x, y)
    if key in _cache:
        return _cache[key]

    result = get_neighbor_lane(nusc_map, x, y, current_lane, get_right_point)

    _cache[key] = result
    return result


def get_right_lane_with_target_coordinate(nusc_map, x, y, current_lane, target_x, target_y, target_yaw):
    right_lane = get_neighbor_lane(nusc_map, x, y, current_lane, get_right_point)
    return lane_with_align_coordinate(right_lane, target_x, target_y, target_yaw)


# 获取附近所有车辆
def get_surrounding_vehicle(helper, nusc_map, instance_token, sample_token, _cache={}):
    key = 'get_surrounding_vehicle_{}_{}'.format(instance_token, sample_token)
    if key in _cache:
        return _cache[key]

    target_ann = helper.get_sample_annotation(instance_token, sample_token)
    target_x, target_y, target_yaw = target_ann['translation']
    result = {}
    min_distance = {}

    # 获取当前车辆所在的道路、左右侧的道路
    lane = get_front_lane(nusc_map, target_x, target_y, get_correct_yaw(target_ann['rotation']))
    left_lane = get_left_lane(nusc_map, target_x, target_y, lane)
    right_lane = get_right_lane(nusc_map, target_x, target_y, lane)
    if len(lane) == 0:
        return result

    # 获取周围车辆
    threshold_within_lane = 2
    target_pos = get_closest_pos(lane, target_x, target_y)
    annotations = helper.get_annotations_for_sample(sample_token)
    annotations = [ann for ann in annotations if
                   ann['category_name'].startswith('vehicle') and ann['token'] != target_ann['token']]
    for ann in annotations:
        x, y, yaw = ann['translation']
        pos = get_closest_pos(lane, x, y)
        relative_pos = pos - target_pos
        pos_on_left = get_closest_pos(left_lane, x, y)
        pos_on_right = get_closest_pos(right_lane, x, y)
        distance_to_lane = point_distance(x, y, lane[pos][0], lane[pos][1])

        # 前车
        if distance_to_lane <= threshold_within_lane and pos - target_pos >= 0:
            # 只获取最近的
            if min_distance.get('front') is None or min_distance.get('front') > (pos - target_pos):
                result['front'] = ann
                min_distance['front'] = pos - target_pos

        # 左前车、左后车
        if pos_on_left != -1:
            distance_to_left_lane = point_distance(x, y, left_lane[pos_on_left][0], left_lane[pos_on_left][1])
            if distance_to_left_lane <= threshold_within_lane and relative_pos >= 0:
                # 只获取最近的
                if min_distance.get('left_front') is None or min_distance.get('left_front') > relative_pos:
                    result['left_front'] = ann
                    min_distance['left_front'] = relative_pos
            if distance_to_left_lane <= threshold_within_lane and relative_pos <= 0:
                # 只获取最近的
                if min_distance.get('left_back') is None or min_distance.get('left_back') > (-relative_pos):
                    result['left_back'] = ann
                    min_distance['left_back'] = (-relative_pos)

        # 右前车、右后车
        if pos_on_right != -1:
            distance_to_right_lane = point_distance(x, y, right_lane[pos_on_right][0], right_lane[pos_on_right][1])
            if distance_to_right_lane <= threshold_within_lane and relative_pos >= 0:
                # 只获取最近的
                if min_distance.get('right_front') is None or min_distance.get('right_front') > relative_pos:
                    result['right_front'] = ann
                    min_distance['right_front'] = relative_pos
            if distance_to_right_lane <= threshold_within_lane and relative_pos <= 0:
                # 只获取最近的
                if min_distance.get('right_back') is None or min_distance.get('right_back') > (-relative_pos):
                    result['right_back'] = ann
                    min_distance['right_back'] = (-relative_pos)

    _cache[key] = result
    return result


# 未来是否进行了变道
def get_lane_change(helper, nusc_map, instance_token, sample_token):
    # 当前道路
    target_ann = helper.get_sample_annotation(instance_token, sample_token)
    target_x, target_y = target_ann['translation'][0], target_ann['translation'][1]
    target_yaw = get_correct_yaw(target_ann['rotation'])
    front_lane = get_front_lane(nusc_map, target_x, target_y, target_yaw)
    left_lane = get_left_lane(nusc_map, target_x, target_y, front_lane)
    right_lane = get_right_lane(nusc_map, target_x, target_y, front_lane)
    # 获取未来六秒的最后一个时间戳
    future_anns = helper.get_future_for_agent(instance_token, sample_token, 6, False, False)
    future_ann = future_anns[-1]
    future_x, future_y = future_ann['translation'][0], future_ann['translation'][1]

    # 哪条车道距离最近
    front_distance, left_distance, right_distance = 100, 100, 100
    for x, y, yaw in front_lane:
        front_distance = min(front_distance, point_distance(x, y, future_x, future_y))
    for x, y, yaw in left_lane:
        left_distance = min(left_distance, point_distance(x, y, future_x, future_y))
    for x, y, yaw in right_lane:
        right_distance = min(right_distance, point_distance(x, y, future_x, future_y))

    # 判断是否进行变道
    threshold = 2
    keep_lane, turn_left, turn_right = False, False, False
    label = 'keep_lane'
    if front_distance <= threshold or (front_distance <= left_distance and front_distance <= right_distance):
        keep_lane = True
    elif left_distance <= right_distance:
        turn_left = True
        label = 'turn_left'
    else:
        turn_right = True
        label = 'turn_right'
    return keep_lane, turn_left, turn_right, label


# 将车辆的类型转换为 one hot 编码
def type_with_one_hot_encoding(type):
    type_list = get_type_list()
    index = type_list.index(type)
    one_hot_encoding = torch.nn.functional.one_hot(torch.tensor(index), num_classes=len(type_list))
    return one_hot_encoding.tolist()


def get_type_list():
    return ['vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction',
            'vehicle.emergency.police', 'vehicle.truck']
