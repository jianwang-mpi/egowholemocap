#  Copyright Jian Wang @ MPI-INF (c) 2023.
import os
import smplx

class SMPLXForward:
    """
    Forward SMPLX model to get 3D joints and mesh vertices
    """

    def __init__(self, smplx_model_dir, use_pca=False, num_pca_comps=12):
        super(SMPLXForward, self).__init__()

        smplx_male_model_path = os.path.join(smplx_model_dir, 'SMPLX_MALE.npz')
        self.male_smplx = smplx.create(smplx_male_model_path, model_type='smplx', num_betas=10,
                                       gender='male', batch_size=1, use_pca=use_pca,
                                       num_pca_comps=num_pca_comps)
        smplx_female_model_path = os.path.join(smplx_model_dir, 'SMPLX_FEMALE.npz')
        self.female_smplx = smplx.create(smplx_female_model_path, model_type='smplx', num_betas=10,
                                         gender='female', batch_size=1, use_pca=use_pca,
                                         num_pca_comps=num_pca_comps)

    def __call__(self, results: dict) -> dict:
        gender = results['smplx_input']['gender']
        if gender == 'male':
            smplx_model = self.male_smplx
        else:
            assert gender == 'female'
            smplx_model = self.female_smplx

        smplx_output = smplx_model(**results['smplx_input'])
        results['smplx_output'] = smplx_output
        return results