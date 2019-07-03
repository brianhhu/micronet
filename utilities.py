import numpy as np


class Cutout(object):
    def __init__(self, sz):
        self._sz = sz

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = int(np.clip(y - self._sz / 2, 0, h))
        y2 = int(np.clip(y + self._sz / 2, 0, h))
        x1 = int(np.clip(x - self._sz / 2, 0, w))
        x2 = int(np.clip(x + self._sz / 2, 0, w))
        img[:, y1:y2, x1:x2].fill_(0.0)
        return img
