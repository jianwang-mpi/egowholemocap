beat_original_joint_names = [
    'Hips',
    'Spine',
    'Spine1',
    'Spine2',
    'Spine3',
    'Neck',
    'Neck1',
    'Head',
    'HeadEnd',
    'HeadEnd_end',
    'RightShoulder',
    'RightArm',
    'RightForeArm',
    'RightHand',
    'RightHandMiddle1',
    'RightHandMiddle2',
    'RightHandMiddle3',
    'RightHandMiddle4',
    'RightHandMiddle4_end',
    'RightHandRing',
    'RightHandRing1',
    'RightHandRing2',
    'RightHandRing3',
    'RightHandRing4',
    'RightHandRing4_end',
    'RightHandPinky',
    'RightHandPinky1',
    'RightHandPinky2',
    'RightHandPinky3',
    'RightHandPinky4',
    'RightHandPinky4_end',
    'RightHandIndex',
    'RightHandIndex1',
    'RightHandIndex2',
    'RightHandIndex3',
    'RightHandIndex4',
    'RightHandIndex4_end',
    'RightHandThumb1',
    'RightHandThumb2',
    'RightHandThumb3',
    'RightHandThumb4',
    'RightHandThumb4_end',
    'LeftShoulder',
    'LeftArm',
    'LeftForeArm',
    'LeftHand',
    'LeftHandMiddle1',
    'LeftHandMiddle2',
    'LeftHandMiddle3',
    'LeftHandMiddle4',
    'LeftHandMiddle4_end',
    'LeftHandRing',
    'LeftHandRing1',
    'LeftHandRing2',
    'LeftHandRing3',
    'LeftHandRing4',
    'LeftHandRing4_end',
    'LeftHandPinky',
    'LeftHandPinky1',
    'LeftHandPinky2',
    'LeftHandPinky3',
    'LeftHandPinky4',
    'LeftHandPinky4_end',
    'LeftHandIndex',
    'LeftHandIndex1',
    'LeftHandIndex2',
    'LeftHandIndex3',
    'LeftHandIndex4',
    'LeftHandIndex4_end',
    'LeftHandThumb1',
    'LeftHandThumb2',
    'LeftHandThumb3',
    'LeftHandThumb4',
    'LeftHandThumb4_end',
    'RightUpLeg',
    'RightLeg',
    'RightFoot',
    'RightForeFoot',
    'RightToeBase',
    'RightToeBaseEnd',
    'RightToeBaseEnd_end',
    'LeftUpLeg',
    'LeftLeg',
    'LeftFoot',
    'LeftForeFoot',
    'LeftToeBase',
    'LeftToeBaseEnd',
    'LeftToeBaseEnd_end'
]

beat_original_joint_names_to_smplx = {
    'Hips': 'beat_hips',
    'Spine': 'pelvis',
    'Spine1': 'beat_spine1',
    'Spine2': 'beat_spine2',
    'Spine3': 'beat_spine3',
    'Neck': 'neck',
    'Neck1': 'beat_neck1',
    'Head': 'head',
    'HeadEnd': 'beat_head_end',
    'HeadEnd_end': 'beat_head_end_end',
    'RightShoulder': 'beat_right_collar',
    'RightArm':  'right_shoulder',
    'RightForeArm': 'right_elbow',
    'RightHand': 'right_wrist',
    'RightHandMiddle1': 'right_middle1',
    'RightHandMiddle2': 'right_middle2',
    'RightHandMiddle3': 'right_middle3',
    'RightHandMiddle4': 'right_middle',
    'RightHandMiddle4_end': 'beat_right_middle_end',
    'RightHandRing': 'beat_right_hand_ring',
    'RightHandRing1': 'right_ring1',
    'RightHandRing2': 'right_ring2',
    'RightHandRing3': 'right_ring3',
    'RightHandRing4': 'right_ring',
    'RightHandRing4_end': 'beat_right_ring_end',
    'RightHandPinky': 'beat_right_pinky',
    'RightHandPinky1': 'right_pinky1',
    'RightHandPinky2': 'right_pinky2',
    'RightHandPinky3': 'right_pinky3',
    'RightHandPinky4': 'right_pinky',
    'RightHandPinky4_end': 'beat_right_pinky_end',
    'RightHandIndex': 'beat_right_index',
    'RightHandIndex1': 'right_index1',
    'RightHandIndex2': 'right_index2',
    'RightHandIndex3': 'right_index3',
    'RightHandIndex4': 'right_index',
    'RightHandIndex4_end': 'beat_right_index_end',
    'RightHandThumb1': 'right_thumb1',
    'RightHandThumb2': 'right_thumb2',
    'RightHandThumb3': 'right_thumb3',
    'RightHandThumb4': 'right_thumb',
    'RightHandThumb4_end': 'beat_right_thumb_end',
    'LeftShoulder': 'beat_left_collar',
    'LeftArm': 'left_shoulder',
    'LeftForeArm': 'left_elbow',
    'LeftHand': 'left_wrist',
    'LeftHandMiddle1': 'left_middle1',
    'LeftHandMiddle2': 'left_middle2',
    'LeftHandMiddle3': 'left_middle3',
    'LeftHandMiddle4': 'left_middle',
    'LeftHandMiddle4_end': 'beat_left_middle_end',
    'LeftHandRing': 'beat_left_hand_ring',
    'LeftHandRing1': 'left_ring1',
    'LeftHandRing2': 'left_ring2',
    'LeftHandRing3': 'left_ring3',
    'LeftHandRing4': 'left_ring',
    'LeftHandRing4_end': 'beat_left_ring_end',
    'LeftHandPinky': 'beat_left_pinky',
    'LeftHandPinky1': 'left_pinky1',
    'LeftHandPinky2': 'left_pinky2',
    'LeftHandPinky3': 'left_pinky3',
    'LeftHandPinky4': 'left_pinky',
    'LeftHandPinky4_end': 'beat_left_pinky_end',
    'LeftHandIndex': 'beat_left_index',
    'LeftHandIndex1': 'left_index1',
    'LeftHandIndex2': 'left_index2',
    'LeftHandIndex3': 'left_index3',
    'LeftHandIndex4': 'left_index',
    'LeftHandIndex4_end': 'beat_left_index_end',
    'LeftHandThumb1': 'left_thumb1',
    'LeftHandThumb2': 'left_thumb2',
    'LeftHandThumb3': 'left_thumb3',
    'LeftHandThumb4': 'left_thumb',
    'LeftHandThumb4_end': 'beat_left_thumb_end',
    'RightUpLeg': 'right_hip',
    'RightLeg': 'right_knee',
    'RightFoot': 'beat_right_foot',
    'RightForeFoot': 'right_ankle',
    'RightToeBase': 'right_foot',
    'RightToeBaseEnd': 'beat_right_toe_base_end',
    'RightToeBaseEnd_end': 'beat_right_toe_base_end_end',
    'LeftUpLeg': 'left_hip',
    'LeftLeg': 'left_knee',
    'LeftFoot': 'beat_left_foot',
    'LeftForeFoot': 'left_ankle',
    'LeftToeBase': 'left_foot',
    'LeftToeBaseEnd': 'beat_left_toe_base_end',
    'LeftToeBaseEnd_end': 'beat_left_toe_base_end_end'
}

beat_joint_names = [beat_original_joint_names_to_smplx[key] for key in beat_original_joint_names]