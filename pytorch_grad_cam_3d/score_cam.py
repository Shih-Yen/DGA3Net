import torch
import tqdm
from pytorch_grad_cam_3d.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None):
        super(ScoreCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform=reshape_transform,
                                       uses_gradients=False)

        if len(target_layers) > 0:
            print("Warning: You are using ScoreCAM with target layers, "
                  "however ScoreCAM will ignore them.")

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            # upsample = torch.nn.UpsamplingBilinear2d(
            #     size=input_tensor.shape[-2:])

            #
            img_ndims = input_tensor.ndim - 2
            ###
            interp_modes = {1:'linear',2:'bilinear',3:'trilinear'}
            interp_mode = interp_modes[img_ndims]
            upsample = torch.nn.Upsample(
                size=input_tensor.shape[2:],align_corners=True,mode =interp_mode) ###
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]
            #new_shape = maxs.shape +(1,)*img_ndims
            maxs = maxs.view( maxs.shape +(1,)*img_ndims)
            mins = mins.view( mins.shape +(1,)*img_ndims)
            #maxs, mins = maxs[:, :, None, None,None], mins[:, :, None, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            # input_tensors = input_tensor[:, None,
            #                              :, :,:] * upsampled[:, :, None, :, :,:]
            input_tensors = input_tensor.unsqueeze(1) * upsampled.unsqueeze(2)
            

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item() for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
