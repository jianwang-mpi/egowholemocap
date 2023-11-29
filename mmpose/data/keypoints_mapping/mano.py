import copy

mano_original_joint_names = [
    'right_wrist',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
]

# add finger tips
mano_joint_names = [
    'right_wrist',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'right_thumb',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_index',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_middle',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_ring',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_pinky']

mano_right_hand_joint_names = copy.deepcopy(mano_joint_names)
mano_left_hand_joint_names = copy.deepcopy(mano_joint_names)
mano_left_hand_joint_names = [name.replace('right', 'left') for name in mano_left_hand_joint_names]

mano_skeleton = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
                    (1, 2), (2, 3), (3, 4),
                    (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20)]