import torch
import torch.nn as nn

from ..builder import LOSSES
from mmpose.core.evaluation.mesh_eval import compute_similarity_transform_torch


@LOSSES.register_module()
class PAMPJPELoss(nn.Module):
    """MPJPE (Mean Per Joint Position Error) loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None,):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N,K,D]):
                Weights across different joint types.
        """
        # align the target and the output
        target_aligned_list = []
        for i in range(len(target)):
            target_aligned_i = compute_similarity_transform_torch(target[i], output[i])
            # data_name_i = kwargs['data_name'][i]
            # if data_name_i == 'mo2cap2':
            #     target_aligned_i = compute_similarity_transform_torch(target[i], output[i])
            # else:
            #     target_aligned_i = target[i]
            target_aligned_list.append(target_aligned_i)
        target_aligned = torch.stack(target_aligned_list)
        # print('output')
        # print(output)
        # print('target')
        # print(target)
        if self.use_target_weight:
            assert target_weight is not None
            loss = torch.mean(
                torch.norm((output - target_aligned) * target_weight, dim=-1))
        else:
            loss = torch.mean(torch.norm(output - target_aligned, dim=-1))

        if loss.item() > 1 or loss.item() < 0:
            print('output')
            print(output)
            print('target')
            print(target)
            print('loss')
            print(loss)
        return loss * self.loss_weight