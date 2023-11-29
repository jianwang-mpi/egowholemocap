mo2cap2_joint_names = [
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
]

mo2cap2_chain = [
    [0, 1, 2, 3],
    [0, 4, 5, 6],
    [1, 7, 8, 9, 10],
    [4, 11, 12, 13, 14],
    [7, 11]
]

mo2cap2_parents = [
    (0, None),
    (1, 0),
    (2, 1),
    (3, 2),
    (4, 0),
    (5, 4),
    (6, 5),
    (7, 1),
    (8, 7),
    (9, 8),
    (10, 9),
    (11, 4),
    (12, 11),
    (13, 12),
    (14, 13),
]

