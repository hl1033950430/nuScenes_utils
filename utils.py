import numpy as np


# 获取半径范围内的所有道路，离散化
def get_lanes_in_radius(x: float, y: float, radius: float,
                        discretization_meters: float,
                        map_api):
    lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    lanes = map_api.discretize_lanes(lanes, discretization_meters)

    return lanes


# 获取道路 lane 中，与 (x, y) 最近的点
def get_closet_pos(lane, x, y):
    return np.argmin(np.array([(pos[0] - x) ** 2 + (pos[1] - y) ** 2 for i, pos in enumerate(lane)]))


# 获取坐标所在的前方道路，离散化
def get_front_lane(nusc_map, target_x, target_y):
    result = []
    # 获取附近所有的车道
    lanes = get_lanes_in_radius(target_x, target_y, 40, 1, nusc_map)
    target_lane = nusc_map.get_closest_lane(target_x, target_y, radius=2)
    # 附近没有道路
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


# 获取目标车辆行驶道路上的前方车辆
def get_front_vehicle(helper, instance_token, sample_token):
    target_ann = helper.get_sample_annotation(instance_token, sample_token)
    target_x, target_y, target_yaw = target_ann['translation']
    # 获取当前车辆所在的道路
    lane = get_front_lane(target_ann)
    # 道路可能不存在
    if len(lane) == 0:
        return ''
    # 获取当前车辆所在道路的位置
    target_pos = get_closet_pos(lane, target_x, target_y)
    # 获取场景下其他的所有车辆，若车辆距离道路在阈值范围内，则说明车辆在道路上
    threshold_within_lane = 2
    annotations = helper.get_annotations_for_sample(sample_token)
    annotations = [ann for ann in annotations if
                   ann['category_name'].startswith('vehicle') and ann['token'] != target_ann['token']]
    front_vehicle = ''
    min_distance_to_target_vehicle = 1000
    for ann in annotations:
        x, y, yaw = ann['translation']
        pos = get_closet_pos(lane, x, y)
        distance_to_lane = ((lane[pos][0] - x) ** 2 + (lane[pos][1] - y) ** 2) ** (1 / 2)
        distance_to_target_vehicle = pos - target_pos
        if distance_to_lane <= threshold_within_lane and 0 < distance_to_target_vehicle <= 50:
            # 与目标车辆在同一车道上，且在目标车辆前方
            if distance_to_target_vehicle < min_distance_to_target_vehicle:
                # 只考虑与目标车辆靠的最近的
                min_distance_to_target_vehicle = distance_to_target_vehicle
                front_vehicle = ann
    return front_vehicle
