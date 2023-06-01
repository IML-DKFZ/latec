import numpy as np
import torch
from scipy.integrate import trapezoid
from torchvision.transforms import GaussianBlur
from abc import ABCMeta, abstractmethod


class BaseEvaluation(metaclass=ABCMeta):
    """
    Base class for all evaluation methods
    get attribution map and img as input, returns a dictionary contains
    evaluation result
    """

    @abstractmethod
    def evaluate(self, heatmap, *args, **kwargs) -> dict:
        raise NotImplementedError


class InsertionDeletion(BaseEvaluation):
    def __init__(self, pixel_batch_size=10, sigma=5.0):
        self.sigma = sigma
        self.pixel_batch_size = pixel_batch_size
        self.gaussian_blurr = GaussianBlur(9, sigma)

    @torch.no_grad()
    def evaluate(self, model, x_batch, y_batch, a_batch):  # noqa
        """# TODO to add docs
        Args:
            heatmap (Tensor): heatmap with shape (H, W) or (3, H, W).
            input_tensor (Tensor): image with shape (3, H, W).
            target (int): class index of the image.
        Returns:
            dict[str, Union[Tensor, np.array, float]]: a dictionary
                containing following fields:
                - del_scores: ndarray,
                - ins_scores:
                - del_input:
                - ins_input:
                - ins_auc:
                - del_auc:
        """
        self.classifier = model
        insertion_auc_total = []
        deletion_auc_total = []

        x_batch_f = torch.tensor(x_batch)
        y_batch_f = torch.tensor(y_batch)
        a_batch_f = torch.tensor(a_batch)

        for i in range(x_batch.shape[0]):
            x_batch = x_batch_f[i, :]
            y_batch = y_batch_f[i]
            a_batch = a_batch_f[i, :]

            # sort pixel in attribution
            num_pixels = torch.numel(a_batch)
            _, indices = torch.topk(a_batch.flatten(), num_pixels)
            indices = np.unravel_index(indices.cpu().numpy(), a_batch.size())

            # apply deletion game
            deletion_perturber = PixelPerturber(x_batch, torch.zeros_like(x_batch))
            deletion_scores = self._procedure_perturb(
                deletion_perturber, num_pixels, indices, y_batch
            )

            # apply insertion game
            if len(x_batch.shape) > 3:
                blurred_input = self.gaussian_blurr(x_batch)
            else:
                blurred_input = x_batch + (torch.randn_like(x_batch) * self.sigma)

            insertion_perturber = PixelPerturber(blurred_input, x_batch)
            insertion_scores = self._procedure_perturb(
                insertion_perturber, num_pixels, indices, y_batch
            )

            # calculate AUC
            insertion_auc = trapezoid(insertion_scores, dx=1.0 / len(insertion_scores))
            deletion_auc = trapezoid(deletion_scores, dx=1.0 / len(deletion_scores))

            insertion_auc_total.append(insertion_auc)
            deletion_auc_total.append(deletion_auc)

        return np.array(insertion_auc_total), np.array(deletion_auc_total)

    def _procedure_perturb(self, perturber, num_pixels, indices, target):
        """# TODO to add docs
        Args:
            perturber (PixelPerturber):
            num_pixels (int):
            indices (tuple):
            target (int):
        Returns:
            np.ndarray:
        """
        scores_after_perturb = []
        replaced_pixels = 0
        while replaced_pixels < num_pixels:
            perturbed_inputs = []
            for i in range(80):
                batch = min(num_pixels - replaced_pixels, self.pixel_batch_size)

                # perturb # of pixel_batch_size pixels
                for pixel in range(batch):
                    perturb_index = (
                        indices[0][replaced_pixels + pixel],
                        indices[1][replaced_pixels + pixel],
                    )

                    # perturb input using given pixels
                    perturber.perturb(perturb_index[0], perturb_index[1])
                perturbed_inputs.append(perturber.get_current())
                replaced_pixels += batch
                if replaced_pixels == num_pixels:
                    break

            # get score after perturb
            device = next(self.classifier.parameters()).device
            perturbed_inputs = torch.stack(perturbed_inputs)

            if perturbed_inputs.shape[1] <= 3:
                logits = self.classifier(perturbed_inputs.to(device))
            else:
                logits = self.classifier(perturbed_inputs.unsqueeze(1).to(device))
            score_after = torch.softmax(logits, dim=-1)[:, target]
            scores_after_perturb = np.concatenate(
                (scores_after_perturb, score_after.detach().cpu().numpy())
            )
        return scores_after_perturb


class Perturber:
    def perturb(self, r: int, c: int):
        """perturb a tile or pixel"""
        raise NotImplementedError

    def get_current(self) -> np.ndarray:
        """get current img with some perturbations"""
        raise NotImplementedError

    def get_idxes(self, hmap: np.ndarray, reverse=False) -> list:
        # TODO: might not needed, we determine perturb priority outside
        #  perturber
        """return a sorted list with shape length NUM_CELLS of
        which pixel/cell index to blur first"""
        raise NotImplementedError

    def get_grid_shape(self) -> tuple:
        """return the shape of the grid, i.e. the max r, c values"""
        raise NotImplementedError


class PixelPerturber(Perturber):
    def __init__(self, inp: torch.Tensor, baseline: torch.Tensor):
        self.current = inp.clone()
        self.baseline = baseline

    def perturb(self, r: int, c: int):
        if len(self.baseline.shape) > 2:
            self.current[:, r, c] = self.baseline[:, r, c]
        else:
            self.current[r, c] = self.baseline[r, c]

    def get_current(self) -> torch.Tensor:
        return self.current

    def get_grid_shape(self) -> tuple:
        return self.current.shape
