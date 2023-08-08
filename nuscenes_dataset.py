import torch
from torch.utils.data import Dataset
import tqdm


class LaneChangeDataset(Dataset):

    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.load_data()

    def __len__(self):
        return len(self.target_vehicle_vector)

    def __getitem__(self, index):
        return self.target_vehicle_vector[index], self.surrounding_vehicle_vector[index], self.labels[index]

    def load_data(self):
        print("load data {}".format(self.dataroot))
        datasource = torch.load(self.dataroot)
        data_size = len(datasource)
        # 历史时间长度
        time_horizon = max([len(item['past_state']) for item in datasource])

        # 目标车辆
        target_vehicle_vector = torch.zeros(data_size, time_horizon, 14)

        # 周围车辆 front(0) left_front(1) left_back(2) right_front(3) right_back(4)
        surrounding_vehicle_directions = ['front', 'left_front', 'left_back', 'right_front', 'right_back']
        surrounding_vehicle_vector = torch.zeros(data_size, time_horizon, 5, 15).float()

        for data_index, data in enumerate(tqdm.tqdm(datasource)):
            past_state = data['past_state']
            this_time_horizon = len(past_state)
            # 越往前时间越早
            past_state = list(reversed(past_state))
            for time_stamp in range(time_horizon - this_time_horizon, time_horizon):
                state = past_state[time_stamp - (time_horizon - this_time_horizon)]
                # 周围车辆
                for direction_index, direction in enumerate(surrounding_vehicle_directions):
                    if direction in state:
                        vehicle = state[direction]
                        feature = [vehicle['x'], vehicle['y'], vehicle['yaw'], vehicle['velocity'],
                                   vehicle['acceleration'], vehicle['distance']]
                        feature.extend(vehicle['type'])
                        feature = torch.tensor(feature)
                        surrounding_vehicle_vector[data_index][time_stamp][direction_index][:] = feature[:]
                # 目标车辆
                vehicle = state['target']
                feature = [vehicle['x'], vehicle['y'], vehicle['yaw'], vehicle['velocity'], vehicle['acceleration']]
                feature.extend(data['vehicle_type'])
                feature = torch.tensor(feature)
                target_vehicle_vector[data_index][time_stamp][:] = feature[:]

        lane_change_type = ['keep_lane', 'turn_left', 'turn_right']
        labels = [data['lane_change_label'] for data in datasource]
        labels = [lane_change_type.index(item) for item in labels]
        labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=len(lane_change_type)).float()

        self.target_vehicle_vector = target_vehicle_vector
        self.surrounding_vehicle_vector = surrounding_vehicle_vector
        self.labels = labels