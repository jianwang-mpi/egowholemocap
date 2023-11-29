import numpy as np

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in
                         ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                          'left_foot', 'right_foot', ]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS - 1))
HML_ROOT_MASK = np.concatenate(([True] * (1 + 2 + 1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK_w_global = np.concatenate(([True] * (1 + 2 + 1),
                                               HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                               HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                               HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                               [True] * 4))
HML_UPPER_BODY_MASK_wo_global = ~HML_LOWER_BODY_MASK_w_global

HML_LOWER_BODY_MASK_wo_global = np.concatenate(([False] * (1 + 2 + 1),
                                                HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                                HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                                HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                                [False] * 4))
HML_UPPER_BODY_MASK_w_global = ~HML_LOWER_BODY_MASK_wo_global

# -------------------------- for mo2cap2 joints ---------------------

MO2CAP2_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ["neck",
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
                                                                "left_foot"]]

NUM_MO2CAP2_JOINTS = len(MO2CAP2_BODY_JOINTS)
MO2CAP2_JOINTS_BINARY = np.array([i in MO2CAP2_BODY_JOINTS for i in range(NUM_HML_JOINTS)])

MO2CAP2_BODY_MASK = np.concatenate(
    (
        [True] * (1 + 2 + 1),
        MO2CAP2_JOINTS_BINARY[1:].repeat(3),
        np.array([False] * (NUM_HML_JOINTS - 1)).repeat(6),
        np.array([False] * NUM_HML_JOINTS).repeat(3),
        [False] * 4)
)

lower_body_confidence1 = 0.96
lower_body_confidence2 = 0.96
MO2CAP2_JOINTS_CONFIDENCE_SOFT = MO2CAP2_JOINTS_BINARY.astype(float) * 1
left_knee_id = HML_JOINT_NAMES.index("left_knee")
right_knee_id = HML_JOINT_NAMES.index("right_knee")
left_ankle_id = HML_JOINT_NAMES.index("left_ankle")
right_ankle_id = HML_JOINT_NAMES.index("right_ankle")
left_foot_id = HML_JOINT_NAMES.index("left_foot")
right_foot_id = HML_JOINT_NAMES.index("right_foot")
MO2CAP2_JOINTS_CONFIDENCE_SOFT[left_knee_id:left_knee_id + 1] = lower_body_confidence2
MO2CAP2_JOINTS_CONFIDENCE_SOFT[right_knee_id:right_knee_id + 1] = lower_body_confidence1
MO2CAP2_JOINTS_CONFIDENCE_SOFT[left_ankle_id:left_ankle_id + 1] = lower_body_confidence2
MO2CAP2_JOINTS_CONFIDENCE_SOFT[right_ankle_id:right_ankle_id + 1] = lower_body_confidence1
MO2CAP2_JOINTS_CONFIDENCE_SOFT[left_foot_id:left_foot_id + 1] = lower_body_confidence2
MO2CAP2_JOINTS_CONFIDENCE_SOFT[right_foot_id:right_foot_id + 1] = lower_body_confidence1
MO2CAP2_BODY_MASK_SOFT = np.concatenate(
    (
        [2] * (1 + 2 + 1),
        MO2CAP2_JOINTS_CONFIDENCE_SOFT[1:].repeat(3),
        np.array([0] * (NUM_HML_JOINTS - 1)).repeat(6),
        np.array([0] * NUM_HML_JOINTS).repeat(3),
        [0] * 4)
).astype(float)

# -----------------------mo2cap2 human ml chain

MO2CAP2_TREE_IN_HUMANML = [
    [HML_JOINT_NAMES.index(name) for name in ["neck", "right_shoulder", "right_elbow", "right_wrist",]],
    [HML_JOINT_NAMES.index(name) for name in ["neck", "left_shoulder", "left_elbow", "left_wrist",]],
    [HML_JOINT_NAMES.index(name) for name in ["right_shoulder", 'right_hip', 'right_knee', 'right_ankle', 'right_foot']],
    [HML_JOINT_NAMES.index(name) for name in ["left_shoulder", 'left_hip', 'left_knee', 'left_ankle', 'left_foot']],
    [HML_JOINT_NAMES.index(name) for name in [ 'left_hip', 'right_hip',]],
]