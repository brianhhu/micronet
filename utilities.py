import math
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler


class Cutout(object):
    def __init__(self, sz, random_pixel=False):
        self._sz = sz
        self.random_pixel = random_pixel

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = int(np.clip(y - self._sz / 2, 0, h))
        y2 = int(np.clip(y + self._sz / 2, 0, h))
        x1 = int(np.clip(x - self._sz / 2, 0, w))
        x2 = int(np.clip(x + self._sz / 2, 0, w))

        if self.random_pixel:
            img[:, y1:y2, x1:x2].random_(to=256)
            img[:, y1:y2, x1:x2] /= 255.0
        else:
            img[:, y1:y2, x1:x2].fill_(0.0)
        return img


def RandomPixelPad(img, padding=4):
    """
    Note: only supports integer padding
    """
    def pad_function(vec, pad_width, *_, **__):
        vec[:pad_width[0]] = np.random.randint(0, 256, size=pad_width[0])
        vec[vec.size-pad_width[1]:] = np.random.randint(0, 256, size=pad_width[1])
        return vec

    img = np.rollaxis(np.asarray(img), 2, 0)
    img = np.stack([np.pad(i, padding, pad_function) for i in img], axis=2)
    img = img.astype('uint8')

    return Image.fromarray(img)


# Source: https://github.com/pytorch/pytorch/pull/11104
class CosineAnnealingRestartsLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule with warm restarts, where :math:`\eta_{max}` is set to the
    initial learning rate, :math:`T_{cur}` is the number of epochs since the
    last restart and :math:`T_i` is the number of epochs in :math:`i`-th run
    (after performing :math:`i` restarts). If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2} \eta_{mult}^i (\eta_{max}-\eta_{min})
        (1 + \cos(\frac{T_{cur}}{T_i - 1}\pi))
        T_i = T T_{mult}^i
    Notice that because the schedule is defined recursively, the learning rate
    can be simultaneously modified outside this scheduler by other operators.
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that in the
    paper the :math:`i`-th run takes :math:`T_i + 1` epochs, while in this
    implementation it takes :math:`T_i` epochs only. This implementation
    also enables updating the range of learning rates by multiplicative factor
    :math:`\eta_{mult}` after each restart.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Length of the initial run (in number of epochs).
        eta_min (float): Minimum learning rate. Default: 0.
        T_mult (float): Multiplicative factor adjusting number of epochs in
            the next run that is applied after each restart. Default: 2.
        eta_mult (float): Multiplicative factor of decay in the range of
            learning rates that is applied after each restart. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T, eta_min=0, T_mult=2.0, eta_mult=1.0, last_epoch=-1):
        self.T = T
        self.eta_min = eta_min
        self.eta_mult = eta_mult

        if T_mult < 1:
            raise ValueError('T_mult should be >= 1.0.')
        self.T_mult = T_mult

        super(CosineAnnealingRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        if self.T_mult == 1:
            i_restarts = self.last_epoch // self.T
            last_restart = i_restarts * self.T
        else:
            # computation of the last restarting epoch is based on sum of geometric series:
            # last_restart = T * (1 + T_mult + T_mult ** 2 + ... + T_mult ** i_restarts)
            i_restarts = int(math.log(1 - self.last_epoch * (1 - self.T_mult) / self.T,
                                      self.T_mult))
            last_restart = int(self.T * (1 - self.T_mult **
                                         i_restarts) / (1 - self.T_mult))

        if self.last_epoch == last_restart:
            T_i1 = self.T * self.T_mult ** (i_restarts - 1)  # T_{i-1}
            lr_update = self.eta_mult / self._decay(T_i1 - 1, T_i1)
        else:
            T_i = self.T * self.T_mult ** i_restarts
            t = self.last_epoch - last_restart
            lr_update = self._decay(t, T_i) / self._decay(t - 1, T_i)

        return [lr_update * (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    @staticmethod
    def _decay(t, T):
        """Cosine decay for step t in run of length T, where 0 <= t < T."""
        return 0.5 * (1 + math.cos(math.pi * t / T))
