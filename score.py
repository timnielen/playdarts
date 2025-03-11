import cv2
import numpy as np

board_radius = 170
inner_bull_radius = 14 / 2
outer_bull_radius = 33 / 2
second_ring_radius = 107
ring_width = 10
fields = [13,4,18,1,20,5,12,9,14,11,8,16,7,19,3,17,2,15,10,6, 26,  8, 36,  2, 40, 10, 24, 18, 28, 22, 16, 32, 14, 38,  6, 34,  4,
       30, 20, 12, 39, 12, 54,  3, 60, 15, 36, 27, 42, 33, 24, 48, 21, 57,  9, 51,  6,
       45, 30, 18, 25, 50, 0]
positions = np.array([
    [0,1], #top
    [0,-1], #bottom
    [-1,0], #left
    [1,0], #right
])
def get_score(corners, darts):
    src_pts = np.array(corners).astype(np.float32)  # (4, 2)
    dst_pts = positions.astype(np.float32)  # (4, 2)
    
    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed_darts = np.ones((darts.shape[0], 3))
    transformed_darts[:, :2] = darts
    transformed_darts = transformed_darts @ M.transpose(1,0)
    transformed_darts /= transformed_darts[:, 2, None]
    radius = np.linalg.norm(transformed_darts[:, :2], axis=1) * board_radius
    angles = np.arctan2(transformed_darts[:, 1], transformed_darts[:, 0])
    angles[angles < 0] += 2*np.pi
    sectors = (angles / (2*np.pi) * 20).astype(np.int32)
    
    for dart in range(sectors.shape[0]):
        r = radius[dart]
        if r < inner_bull_radius:
            sectors[dart] = len(fields)-2
        elif r < outer_bull_radius:
            sectors[dart] = len(fields)-3
        elif r < second_ring_radius and r > second_ring_radius - ring_width:
            sectors[dart] += 20
        elif r < board_radius and r > board_radius - ring_width:
            sectors[dart] += 40
        elif r > board_radius:
            sectors[dart] = len(fields)-1
            continue
    
    points = np.array(fields)[sectors]
        # Show the transformed image
    return points, sectors