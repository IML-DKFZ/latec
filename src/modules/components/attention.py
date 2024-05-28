import argparse
import torch
import numpy as np
from numpy import *

# # compute rollout between attention layers
# def compute_rollout_attention(all_layer_matrices, start_layer=0):
#     # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
#     num_tokens = all_layer_matrices[0].shape[1]
#     batch_size = all_layer_matrices[0].shape[0]
#     eye = (
#         torch.eye(num_tokens)
#         .expand(batch_size, num_tokens, num_tokens)
#         .to(all_layer_matrices[0].device)
#     )
#     all_layer_matrices = [
#         all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))
#     ]
#     matrices_aug = [
#         all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
#         for i in range(len(all_layer_matrices))
#     ]
#     joint_attention = matrices_aug[start_layer]
#     for i in range(start_layer + 1, len(matrices_aug)):
#         joint_attention = matrices_aug[i].bmm(joint_attention)
#     return joint_attention


def rescale_attention_3D(tensor, height=7, width=7, z=7, scale_factor=4, dim=28):
    atten = tensor.reshape(1, 1, height, width, z)
    atten = torch.nn.functional.interpolate(
        atten, scale_factor=scale_factor, mode="trilinear"
    )
    atten = atten.reshape(1, dim, dim, dim).detach().cpu().numpy()
    atten = (atten - atten.min()) / (atten.max() - atten.min())
    return atten


def rescale_attention_2D(tensor, height=14, width=14):
    atten = tensor.reshape(1, 1, height, width)
    atten = torch.nn.functional.interpolate(atten, scale_factor=16, mode="bilinear")
    atten = atten.reshape(224, 224).detach().cpu().numpy()
    atten = (atten - atten.min()) / (atten.max() - atten.min())
    return atten


def rescale_attention_1D(tensor, height=14, width=14):
    atten = torch.nn.functional.interpolate(tensor, scale_factor=4, mode="linear")
    # atten = atten.reshape(224, 224).detach().cpu().numpy()
    atten = (atten - atten.min()) / (atten.max() - atten.min())
    return atten


class AttentionLRP:
    def __init__(self, model, modality):
        self.model = model
        self.model.eval()
        self.modality = modality

    def attribute(
        self,
        inputs,
        target=None,
        method="transformer_attribution",
        is_ablation=False,
        start_layer=0,
    ):
        list = []
        for i in range(inputs.shape[0]):
            output = self.model(inputs[i].unsqueeze(0))
            kwargs = {"alpha": 1}
            if target == None:
                target = np.argmax(output.cpu().data.numpy(), axis=-1)

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            idx = target.cpu() if len(target.shape) == 0 else target[i].cpu()
            one_hot[0, idx] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(inputs.device)
            one_hot = torch.sum(one_hot * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            atten = self.model.relprop(
                torch.tensor(one_hot_vector).to(inputs.device),
                method=method,
                is_ablation=is_ablation,
                start_layer=start_layer,
                **kwargs,
            )

            if method != "full":
                if self.modality == "image":
                    atten = np.repeat(
                        np.expand_dims(rescale_attention_2D(atten), (0, 1)), 3, axis=1
                    ).squeeze()
                elif self.modality == "volume":
                    atten = rescale_attention_3D(atten)
                elif self.modality == "point_cloud":
                    atten = np.repeat(
                        np.expand_dims(atten.detach().cpu().numpy(), 0),
                        3,
                        axis=0,
                    ).squeeze()
            else:
                if self.modality == "image":
                    atten = np.repeat(
                        np.expand_dims(atten.detach().cpu().numpy(), 1), 3, axis=1
                    ).squeeze()
                elif self.modality == "volume" or self.modality == "point_cloud":
                    atten = atten.detach().cpu().numpy()

                atten = np.maximum(atten, 0.0)

            list.append(atten)

        return np.array(list)


# class Baselines:
#     def __init__(self, model):
#         self.model = model
#         self.model.eval()

#     def generate_cam_attn(self, input, index=None):
#         output = self.model(input, register_hook=True)
#         if index == None:
#             index = np.argmax(output.cpu().data.numpy())

#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0][index] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         one_hot = torch.sum(one_hot * output)

#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)
#         #################### attn
#         grad = self.model.blocks[-1].attn.get_attn_gradients()
#         cam = self.model.blocks[-1].attn.get_attention_map()
#         cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
#         grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
#         grad = grad.mean(dim=[1, 2], keepdim=True)
#         cam = (cam * grad).mean(0).clamp(min=0)
#         cam = (cam - cam.min()) / (cam.max() - cam.min())

#         return cam
#         #################### attn

#     def generate_rollout(self, input, start_layer=0):
#         self.model(input)
#         blocks = self.model.blocks
#         all_layer_attentions = []
#         for blk in blocks:
#             attn_heads = blk.attn.get_attention_map()
#             avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
#             all_layer_attentions.append(avg_heads)
#         rollout = compute_rollout_attention(
#             all_layer_attentions, start_layer=start_layer
#         )
#         return rollout[:, 0, 1:]

#     def raw_attention(self, input):
#         self.model(input)
#         blocks = self.model.blocks

#         attn_heads = blocks[-1].attn.get_attention_map()
#         avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()

#         return avg_heads
