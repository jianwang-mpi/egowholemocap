#  Copyright Jian Wang @ MPI-INF (c) 2023.
from .egobody_dataset import EgoBodyDataset
from .ego_smplx_dataset import EgoSMPLXDataset
from .renderpeople_motion_dataset import RenderpeopleMotionDataset
from .beat_dataset import BEATDataset
from .studio_motion_dataset import MocapStudioMotionDataset
from .fullbody_motion_test_dataset import FullBodyMotionTestDataset
from .fullbody_ego_motion_eval_dataset import FullBodyEgoMotionEvalDataset
from .smplx_features_dataset import EgoSMPLXFeaturesDataset
from .fullbody_ego_motion_test_dataset import FullBodyEgoMotionTestDataset
from .fullbody_egopw_motion_test_dataset import FullBodyEgoPwMotionTestDataset

__all__ = ['EgoBodyDataset', 'EgoSMPLXDataset', 'RenderpeopleMotionDataset', 'BEATDataset',
           'MocapStudioMotionDataset', 'FullBodyMotionTestDataset', 'EgoSMPLXFeaturesDataset',
           'FullBodyEgoMotionTestDataset', 'FullBodyEgoPwMotionTestDataset', 'FullBodyEgoMotionEvalDataset']