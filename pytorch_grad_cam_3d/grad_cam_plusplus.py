import numpy as np
from pytorch_grad_cam_3d.base_cam import BaseCAM

# https://arxiv.org/abs/1710.11063


class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(model, target_layers, use_cuda,
                                              reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        img_ndims = input_tensor.ndim - 2
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        #activations = activations.reshape((*activations.shape[:2],-1))
        sum_activations = np.sum(activations, axis=tuple(range(2,input_tensor.ndim)))
        sum_activations = sum_activations.reshape( sum_activations.shape +(1,)*img_ndims)
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        
        #weights = np.sum(weights, axis=(2, 3))
        weights = np.sum(weights, axis=tuple(range(2,input_tensor.ndim)))
        return weights
