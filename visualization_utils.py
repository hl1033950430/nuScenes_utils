from utils import NuscenesUtil
import cv2 as cv

class VisualizationUtil():

    @staticmethod
    def transfer_to_image_coordinate(x, y, image_size=500):
        return int(x * 5 + image_size / 2), image_size - int(y * 5 + image_size / 2)

    # 画出一条轨
    # [[x1,y1], [x2,y2]....]
    @staticmethod
    def draw_a_trajectory(base_image, trajectory, color):
        start_point = [0, 0]
        for point in trajectory:
            VisualizationUtil.draw_line(base_image, start_point[0], start_point[1], point[0], point[1], color)
            start_point = point
        return

    # 画出一组轨迹
    # [[[x1,y1], [x2,y2]....], [[x1,y1], [x2,y2]....], ...]
    @staticmethod
    def draw_a_set_of_trajectories(base_image, trajectories, colors):
        for index in range(len(trajectories)):
            trajectory = trajectories[index]
            color = colors[index]
            VisualizationUtil.draw_a_trajectory(trajectory, color)
        return

    # 将线段画在图片上，并将坐标转换为图片的坐标
    @staticmethod
    def draw_line(base_image, start_x, start_y, end_x, end_y, color):
        start_x, start_y = VisualizationUtil.transfer_to_image_coordinate(start_x, start_y)
        end_x, end_y = VisualizationUtil.transfer_to_image_coordinate(end_x, end_y)
        cv.line(base_image, [start_x, start_y], [end_x, end_y], color, thickness=5)

    # 展示图片
    @staticmethod
    def show_image(image):
        image = cv.cvtColor(image.astype("uint8"), cv.COLOR_RGB2BGR)
        cv.imshow('image', image)
        cv.waitKey(0)
        cv.destroyAllWindows()