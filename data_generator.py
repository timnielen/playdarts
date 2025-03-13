import numpy as np
board_radius = 170
inner_bull_radius = 14 / 2
outer_bull_radius = 33 / 2
second_ring_radius = 107
ring_width = 10
fields = [13,  4, 18,  1, 20,  5, 12,  9, 14, 11,  8, 16,  7, 19,  3, 17,  2, 15, 10,  6, 
          39, 12, 54,  3, 60, 15, 36, 27, 42, 33, 24, 48, 21, 57,  9, 51,  6, 45, 30, 18, 
          26,  8, 36,  2, 40, 10, 24, 18, 28, 22, 16, 32, 14, 38,  6, 34,  4, 30, 20, 12, 
          25, 50,  0]

class Player:
    def __init__(self, kind=2, accuracy=0.5, min_field=0, max_field=len(fields)-1):
        self.kind = kind
        self.accuracy =  accuracy
        self.min_field = min_field
        self.max_field = max_field
    def generate_dart_locations(self, num_darts):
        #initialize result
        darts = np.zeros((num_darts, 2))
        
        # generate target fields depending on the kind of player (exclude the 0 field )
        target_fields = np.random.randint(self.min_field, self.max_field, size=num_darts, dtype=np.int32)
        for i in range(self.kind):
            if i+1 == num_darts:
                break
            target_fields[i+1] = target_fields[i]
        
        
        for dart in range(num_darts):
            target_field = target_fields[dart]
            if target_field >= len(fields)-3: #if it is bull, second bull or outside the board the angle is arbitrary
                angle_min = 0
                angle_max = 2*np.pi
            else:
                sector_size = (2*np.pi) / 20
                angle_min = (target_field % 20) * sector_size + sector_size/2
                angle_max = angle_min + sector_size
            target_angle = angle_min + np.random.rand() * (angle_max - angle_min)
            if target_field < 20:
                target_radius = (second_ring_radius + (board_radius-ring_width)) / 2
            elif target_field < 40:
                target_radius = ((second_ring_radius-ring_width) + second_ring_radius)  / 2
            elif target_field < 60:
                target_radius = ((board_radius-ring_width) + board_radius) / 2
            elif target_field == len(fields)-3:
                target_radius = (inner_bull_radius + outer_bull_radius) / 2
            else:
                target_radius = 0
            
            x = np.cos(target_angle)
            y = np.sin(target_angle)
            target = target_radius * np.array([x,y])
            darts[dart] = np.random.normal(loc=target, scale=self.accuracy)
            
        return darts
                