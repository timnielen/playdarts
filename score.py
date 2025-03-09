import cv2
import numpy as np

board_radius = 170
inner_bull_radius = 12.7 / 2
outer_bull_radius = 32 / 2
second_ring_radius = 107
ring_width = 8
points_list = np.array([13,4,18,1,20,5,12,9,14,11,8,16,7,19,3,17,2,15,10,6])
positions = np.array([
    [0,1], #top
    [0,-1], #bottom
    [-1,0], #left
    [1,0], #right
])
def get_score(images, num_points, xy):
    bs, width, height, channels = images.shape
    image_positions = xy[:, :4]
    all_points= []
    for index in range(bs):
        src_pts = image_positions.numpy().astype(np.float32)  # (4, 2)
        dst_pts = positions.astype(np.float32)  # (4, 2)
        
        # Get the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        transformed_xy = np.ones((num_points[index], 3))
        transformed_xy[:, :2] = xy[index, :num_points[index]]
        transformed_xy = transformed_xy @ M.transpose(1,0)
        transformed_xy /= transformed_xy[:, 2, None]
        
        darts = transformed_xy[4:, :2]
        radius = np.linalg.norm(darts, axis=1) * board_radius
        angles = np.arctan2(darts[:, 1], darts[:, 0])
        angles[angles < 0] += 2*np.pi
        sector = (angles / (2*np.pi) * 20).astype(np.int32)
        points = points_list[sector]
        for dart in range(num_points[index]-4):
            r = radius[dart]
            if r < inner_bull_radius:
                points[dart] = 50
                continue
            if r < outer_bull_radius:
                points[dart] = 25
                continue
            if r < second_ring_radius and r > second_ring_radius - ring_width:
                points[dart] *= 3
                continue
            if r < board_radius and r > board_radius - ring_width:
                points[dart] *= 2
                continue
            if r > board_radius:
                points[dart] = 0
        all_points.append(points)
        # Show the transformed image
    return all_points