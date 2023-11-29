# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from mmpose.datasets.datasets.diffusion.keypoints_to_hml3d import recover_from_ric
from mmpose.models import POSENETS
from mmpose.models.detectors.base import BasePose
from mmpose.models.diffusion_mdm.utils import dist_util
from mmpose.models.diffusion_mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mmpose.models.diffusion_mdm.utils.parser_util import edit_args
from mmpose.models.diffusion_mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mmpose.utils.visualization.visualize_mdm_results import render_mdm_results
from mmpose.models.diffusion_mdm.data_loaders import humanml_utils



@POSENETS.register_module()
class MDMEdit(BasePose):


    def __init__(self, model_path, max_frames=196):
        super(MDMEdit, self).__init__()
        self.args = edit_args()
        self.model_path = model_path
        name = os.path.basename(os.path.dirname(self.model_path))
        niter = os.path.basename(self.model_path).replace('model', '').replace('.pt', '')
        self.out_path = os.path.join(os.path.dirname(self.model_path),
                                     'edit_{}_{}_seed{}'.format(name, niter, torch.seed()))

        self.model, self.diffusion = create_model_and_diffusion(self.args, data=None)

        print(f"Loading checkpoints from [{self.model_path}]...")
        state_dict = torch.load(self.model_path, map_location='cpu')
        load_model_wo_clip(self.model, state_dict)

        self.model = ClassifierFreeSampleModel(self.model)  # wrapping model with the classifier-free sampler
        # self.model.eval()  # disable random masking

        self.args.guidance_param = 0.  # disable guidance
        self.max_frames = max_frames

    def forward_train(self, img, img_metas, **kwargs):
        pass

    def forward_test(self, img, img_metas, **kwargs):
        pass



    def forward(self, data, mask, lengths, mean, std, img_metas=None, return_loss=True, **kwargs):
        batch_size, seq_len, feature_size = data.shape
        assert self.max_frames == seq_len
        # should not reshape here!
        data = torch.unsqueeze(data, dim=-1)
        input_motions = torch.permute(data, (0, 2, 3, 1))
        assert input_motions.shape == (batch_size, feature_size, 1, seq_len)
        mask = torch.permute(mask, (0, 2, 3, 1))
        # mask = mask.reshape(batch_size, feature_size, 1, seq_len)
        gt_frames_per_sample = {}
        model_kwargs = {
            'y': {
                'mask': mask,
                'lengths': lengths,
                'text': '',
            }
        }
        self.args.guidance_param = 0.  # disable guidance
        model_kwargs['y']['inpainted_motion'] = input_motions

        # model_kwargs['y']['inpainting_mask'] = torch.ones_like(data, dtype=torch.bool,
        #                                                        device=data.device)  # True means use gt motion
        # for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
        #     start_idx, end_idx = int(self.args.prefix_end * length), int(self.args.suffix_start * length)
        #     print('start idx is {}, end idx is {}'.format(start_idx, end_idx))
        #     gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, self.max_frames))
        #     model_kwargs['y']['inpainting_mask'][i, :, :, start_idx: end_idx] = False  # do inpainting in those frames

        model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_UPPER_BODY_MASK_w_global, dtype=torch.bool,
                                                            device=input_motions.device)  # True is upper body data
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])

        lower_body_mask = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK_wo_global, dtype=torch.bool,
                                       device=input_motions.device)  # True is lower body data
        lower_body_mask = lower_body_mask.unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])

        # add noise to the lower body
        noise = torch.randn_like(input_motions) * 0.2
        input_motions = input_motions + noise * lower_body_mask
        # input_motions[:, 60:63, :, 100:110] += 100
        model_kwargs['y']['inpainted_motion'] = input_motions

        all_motions = []
        all_lengths = []

        for rep_i in range(self.args.num_repetitions):
            print(f'### Start sampling [repetitions #{rep_i}]')

            # add CFG scale to batch
            model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * self.args.guidance_param

            sample_fn = self.diffusion.p_sample_loop

            sample = sample_fn(
                self.model,
                (batch_size, self.model.njoints, self.model.nfeats, self.max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            n_joints = 22 if sample.shape[1] == 263 else 21
            # sample shape: (batch_size, 263, 1, max_frames)
            sample = sample.permute(0, 2, 3, 1)
            # sample shape: (batch_size, 1, max_frames, 263)
            sample = sample * std + mean  # this should be in the dataset!
            sample = recover_from_ric(sample, n_joints)
            # sample shape: (batch_size, 1, max_frames, 22, 3)
            sample = sample.view(-1, *sample.shape[2:])

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * batch_size} samples")

        total_num_samples = self.args.num_repetitions * self.args.num_samples
        all_motions = np.concatenate(all_motions, axis=0)
        print('all motions shape: ', all_motions.shape)
        all_motions = all_motions[:total_num_samples]
        all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
        print(f'all lengths shape: {all_lengths.shape}')
        results = {'motions': all_motions, 'lengths': all_lengths, 'input_motions': input_motions}
        return results

    def show_result(self, **kwargs):
        result_motions = kwargs['results']
        input_motions = kwargs['inputs']
        save_dir = kwargs['save_dir']

        render_mdm_results(input_motions, result_motions, save_dir)



