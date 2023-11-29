#  Copyright Jian Wang @ MPI-INF (c) 2023.

import torch
import torch.nn as nn
from mmpose.models.ego_previous_methods.mo2cap2.pose_net import get_pose_net

from mmpose.models.builder import POSENETS
from mmpose.models.ego_previous_methods.mo2cap2.depth_module import DepthModule


@POSENETS.register_module()
class Mo2Cap2Network(nn.Module):
    def __init__(self, phase, pose_net_module_path, pose_net_zoom_module_path, depth_net_module_path):
        """
        Args:
            phase: phase should be one of ['pose', 'pose_zoom', 'depth', 'all']
        """
        super(Mo2Cap2Network, self).__init__()

        assert phase in ['pose', 'pose_zoom', 'depth', 'all']
        self.phase = phase

        self.heatmap_net = get_pose_net(model_path=pose_net_module_path)

        self.heatmap_net_zoom = get_pose_net(model_path=pose_net_zoom_module_path)

        self.depth_net = DepthModule(input_features=1024 * 2 + 2048 * 2)

        self.depth_net.load_state_dict(torch.load(depth_net_module_path))

    def init_weights(self, pretrained=None):
        pass

    def forward(self, img):
        x, layer3_mid, layer4 = self.heatmap_net.forward_2D_pose(img, slice=False)
        y, layer3_zoom_mid, layer4_zoom = self.heatmap_net_zoom.forward_2D_pose_zoom(img, slice=False)
        layer4 = torch.nn.functional.interpolate(layer4, size=(layer3_mid.shape[2], layer3_mid.shape[3]))
        layer4_zoom = torch.nn.functional.interpolate(layer4_zoom, size=(layer3_zoom_mid.shape[2], layer3_zoom_mid.shape[3]))
        depth = self.depth_net(layer3_mid, layer4, layer3_zoom_mid, layer4_zoom)
        return depth
