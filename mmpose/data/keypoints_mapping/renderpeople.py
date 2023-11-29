render_people_orginal_joint_names = [
    'hip',
    'upperleg_l',
    'upperleg_r',
    'spine_01',
    'lowerleg_l',
    'lowerleg_r',
    'spine_02',
    'foot_l',
    'foot_r',
    'spine_03',
    'ball_l',
    'ball_r',
    'foot_end_l',
    'foot_end_r',
    'neck',
    'shoulder_l',
    'shoulder_r',
    'head',
    'upperarm_l',
    'upperarm_r',
    'lowerarm_l',
    'lowerarm_r',
    'hand_l',
    'hand_r',
    'head_end',
    'jaw',
    'jaw_end',
    'eye_l',
    'eye_end_l',
    'eye_r',
    'eye_end_r',
    'mouth_l',
    'mouth_r',
    'index_01_l',
    'index_02_l',
    'index_03_l',
    'index_end_l',
    'middle_01_l',
    'middle_02_l',
    'middle_03_l',
    'middle_end_l',
    'pinky_01_l',
    'pinky_02_l',
    'pinky_03_l',
    'pinky_end_l',
    'ring_01_l',
    'ring_02_l',
    'ring_03_l',
    'ring_end_l',
    'thumb_01_l',
    'thumb_02_l',
    'thumb_03_l',
    'thumb_end_l',
    'index_01_r',
    'index_02_r',
    'index_03_r',
    'index_end_r',
    'middle_01_r',
    'middle_02_r',
    'middle_03_r',
    'middle_end_r',
    'pinky_01_r',
    'pinky_02_r',
    'pinky_03_r',
    'pinky_end_r',
    'ring_01_r',
    'ring_02_r',
    'ring_03_r',
    'ring_end_r',
    'thumb_01_r',
    'thumb_02_r',
    'thumb_03_r',
    'thumb_end_r',
]

render_people_to_smplx_joint = {
    'hip': "pelvis",
    'upperleg_l': "left_hip",
    'upperleg_r': "right_hip",
    'spine_01': "renderpeople_spine1", # note: spine position might be different
    'lowerleg_l': "left_knee",
    'lowerleg_r': "right_knee",
    'spine_02': "renderpeople_spine2",
    'foot_l': "left_ankle",
    'foot_r': "right_ankle",
    'spine_03': "renderpeople_spine3",
    'ball_l': "left_foot",
    'ball_r': "right_foot",
    'foot_end_l': 'left_foot_end',
    'foot_end_r': 'right_foot_end',
    'neck': "neck",
    'shoulder_l': "renderpeople_left_shoulder",
    'shoulder_r': "renderpeople_right_shoulder",
    # 'head': "renderpeople_head",    # note: head position might be different
    'head': "head",
    'upperarm_l': "left_shoulder",
    'upperarm_r': "right_shoulder",
    'lowerarm_l': "left_elbow",
    'lowerarm_r': "right_elbow",
    'hand_l': "left_wrist",
    'hand_r': "right_wrist",
    'head_end': "renderpeople_head_end",
    'jaw': "renderpeople_jaw",
    'jaw_end': "renderpeople_jaw_end",
    'eye_l': "renderpeople_left_eye_smplx",
    'eye_end_l': "renderpeople_left_eye_end_smplx",
    'eye_r': "renderpeople_right_eye_smplx",
    'eye_end_r': "renderpeople_right_eye_end_smplx",
    'mouth_l': "renderpeople_left_mouth",
    'mouth_r': "renderpeople_right_mouth",
    'index_01_l': "left_index1",
    'index_02_l': "left_index2",
    'index_03_l': "left_index3",
    'index_end_l': "left_index",
    'middle_01_l': "left_middle1",
    'middle_02_l': "left_middle2",
    'middle_03_l': "left_middle3",
    'middle_end_l': "left_middle",
    'pinky_01_l': "left_pinky1",
    'pinky_02_l': "left_pinky2",
    'pinky_03_l': "left_pinky3",
    'pinky_end_l': "left_pinky",
    'ring_01_l': "left_ring1",
    'ring_02_l': "left_ring2",
    'ring_03_l': "left_ring3",
    'ring_end_l': "left_ring",
    'thumb_01_l': "left_thumb1",
    'thumb_02_l': "left_thumb2",
    'thumb_03_l': "left_thumb3",
    'thumb_end_l': "left_thumb",
    'index_01_r': "right_index1",
    'index_02_r': "right_index2",
    'index_03_r': "right_index3",
    'index_end_r': "right_index",
    'middle_01_r': "right_middle1",
    'middle_02_r': "right_middle2",
    'middle_03_r': "right_middle3",
    'middle_end_r': "right_middle",
    'pinky_01_r': "right_pinky1",
    'pinky_02_r': "right_pinky2",
    'pinky_03_r': "right_pinky3",
    'pinky_end_r': "right_pinky",
    'ring_01_r': "right_ring1",
    'ring_02_r': "right_ring2",
    'ring_03_r': "right_ring3",
    'ring_end_r': "right_ring",
    'thumb_01_r': "right_thumb1",
    'thumb_02_r': "right_thumb2",
    'thumb_03_r': "right_thumb3",
    'thumb_end_r': "right_thumb",
}
# convert the render people joint names to smplx joint names
render_people_joint_names = [render_people_to_smplx_joint[key] for key in render_people_orginal_joint_names]

render_people_left_hand_names = [
    'index_01_l',
    'index_02_l',
    'index_03_l',
    'index_end_l',
    'middle_01_l',
    'middle_02_l',
    'middle_03_l',
    'middle_end_l',
    'pinky_01_l',
    'pinky_02_l',
    'pinky_03_l',
    'pinky_end_l',
    'ring_01_l',
    'ring_02_l',
    'ring_03_l',
    'ring_end_l',
    'thumb_01_l',
    'thumb_02_l',
    'thumb_03_l',
    'thumb_end_l',
]