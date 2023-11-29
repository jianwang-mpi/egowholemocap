import numpy as np

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.models.builder import POSENETS
from mmpose.models.builder import build_loss
from mmpose.models.detectors.base import BasePose
from mmpose.models.expose.config import cfg
from mmpose.models.expose.models.smplx_net import SMPLXNet
from mmpose.models.expose.utils.checkpointer import Checkpointer


@POSENETS.register_module()
class SMPLXNetWrapper(BasePose):
    """
    Wrapper of SMPLX net and register with mmcv
    """

    def __init__(self, exp_cfg_path, smplx_loss, pretrained):
        super(SMPLXNetWrapper, self).__init__()

        cfg.merge_from_file(exp_cfg_path)
        self.smplx_net = SMPLXNet(cfg)

        self.smplx_loss = build_loss(smplx_loss)

        self.pretrained = pretrained

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
            print('init weights with pretrained model')
        if isinstance(self.pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self.smplx_net, self.pretrained, strict=False, logger=logger)
            checkpointer = Checkpointer(self.smplx_net, save_dir=pretrained)
            checkpointer.load_checkpoint()

    def forward_train(self, img, img_metas, **kwargs):
        output = self.smplx_net(img, targets=None)
        # re-organize the output dict
        output_body_stages = output['body']
        num_stages = output_body_stages['num_stages']
        output_body_dict = output_body_stages[f'stage_{num_stages - 1:02d}']
        pred_keypoints_3d = output_body_dict['joints']
        pose = output_body_dict['body_pose']
        beta = output_body_dict['betas']


        pred_dict = {
            'keypoints_3d': pred_keypoints_3d,
            'pose': pose,
            'beta': beta
        }
        # solved: why the target is still here?
        target = kwargs
        # if return loss
        losses = self.smplx_loss(pred_dict, target)
        keypoint_accuracy = self.get_accuracy(pred_dict, target)
        losses.update(keypoint_accuracy)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        self.smplx_net = self.smplx_net.eval()
        output = self.smplx_net(img, targets=None)
        # re-organize the output dict
        output_body_stages = output['body']
        num_stages = output_body_stages['num_stages']
        output_body_dict = output_body_stages[f'stage_{num_stages - 1:02d}']
        pred_keypoints_3d = output_body_dict['joints']
        # global_orient = output_body_dict['global_orient']
        # pose = output_body_dict['body_pose']
        # beta = output_body_dict['betas']
        pred_dict = {
            'keypoints_pred': pred_keypoints_3d,
            'img_metas': img_metas,
            'output_body_dict': output_body_dict
        }
        return pred_dict

    def forward(self, img,
                img_metas=None,
                return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def show_result(self, **kwargs):
        """
        reproject smplx model over egocentric image?
        Rendering? With what framework? Pytorch3D?
        Args:
            **kwargs:

        Returns:

        """
        pass

    def get_accuracy(self, output, target):
        """
        Get mpjpe accuracy
        Args:
            output:
            keypoints_3d:
            target_weight:
        Returns:

        """
        accuracy = dict()

        pred_keypoints_3d = output['keypoints_3d']
        N, K, _ = pred_keypoints_3d.shape
        gt_keypoints_3d = target['keypoints_3d']
        joints_3d_visible = target['keypoints_3d_visible']

        # should calculate the conf of each joint!
        mpjpe = keypoint_mpjpe(
            pred_keypoints_3d.detach().cpu().numpy(),
            gt_keypoints_3d.detach().cpu().numpy(),
            mask=joints_3d_visible.detach().cpu().numpy().astype(np.bool), alignment='none')
        accuracy['mpjpe'] = float(mpjpe)

        return accuracy
