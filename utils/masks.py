# This code snippet is adapted from the LaMa project by SAIC-AIM:
# https://github.com/advimman/lama/blob/main/saicinpainting/training/data/masks.py
# License: Apache License 2.0
# Copyright (c) 2021 SAIC AI Lab, Samsung AI Center


from enum import Enum
import random
import cv2
import numpy as np
import torch


class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part


class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_len=10, min_width=5, min_times=0,
                               max_times=10, draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = min_len + np.random.randint(max_len)
            brush_w = min_width + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    
    result = mask[None, ...]
    return result


class RandomIrregularMaskEmbedder:
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_len=60, min_width=20, min_times=0, max_times=10,
                 ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_len = min_len
        self.min_width = min_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(img.shape[1:], max_angle=self.max_angle,
                                          max_len=cur_max_len, max_width=cur_max_width,
                                          min_len=self.min_len, min_width=self.min_width,
                                          min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)


def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=1, max_times=3):
    
    height, width = shape
    union_mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for _ in range(times):
        mask_height = random.randint(1, height)
        mask_width = random.randint(1, width)
        
        start_h = random.randint(0, height - mask_height)
        start_w = random.randint(0, width - mask_width)

        union_mask[start_h:start_h + mask_height, start_w:start_w + mask_width] = 1
    
    return union_mask[None, ...]


class RandomRectangleMaskEmbedder:
    def __init__(self, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=1, max_times=3, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None, nb_times=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        if nb_times is not None:
            min_times = nb_times
            max_times = nb_times
        else:
            min_times = self.min_times
            max_times = cur_max_times
        return make_random_rectangle_mask(img.shape[1:], margin=self.margin, bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size, min_times=min_times,
                                          max_times=max_times)


class FullMaskEmbedder:
    def __init__(self, **kwargs):
        pass

    def __call__(self, img, iter_i=None, raw_image=None, **kwargs):
        mask = np.ones((img.shape[-2], img.shape[-1]), np.float32)
        return mask[None, ...]


class CocoSegmentationMaskEmbedder:
    def __init__(self):
        ## Empty class
        pass

    def __call__(self, masks):
        return masks


class MixedMaskEmbedder:
    def __init__(self, irregular_proba=1 / 4, irregular_kwargs=None,
                 box_proba=1 / 4, box_kwargs=None,
                 full_proba=1 / 4, full_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 segm_proba=1 / 4, segm_kwargs=None,
                 invert_proba=0.5,
                 **kwargs
                 ):
        self.probas = []
        self.gens = []

        self.probas.append(irregular_proba)
        if irregular_kwargs is None:
            irregular_kwargs = {'max_angle': 4, 'max_len': 50, 'max_width': 20, 'min_len': 50, 'min_width': 20,
                                'min_times': 1, 'max_times': 5}
        else:
            irregular_kwargs = dict(irregular_kwargs)
        irregular_kwargs['draw_method'] = DrawMethod.LINE
        self.gens.append(RandomIrregularMaskEmbedder(**irregular_kwargs))

        self.probas.append(box_proba)
        if box_kwargs is None:
            box_kwargs = {'margin': 10, 'bbox_min_size': 30, 'bbox_max_size': 100, 'min_times': 1, 'max_times': 3}
        else:
            box_kwargs = dict(box_kwargs)
        self.gens.append(RandomRectangleMaskEmbedder(**box_kwargs))

        # full mask
        self.probas.append(full_proba)
        self.gens.append(FullMaskEmbedder())

        self.probas.append(segm_proba)
        if segm_kwargs is None:
            segm_kwargs = {}
        self.gens.append(CocoSegmentationMaskEmbedder(**segm_kwargs))

        # draw a lot of random squares
        if squares_proba > 0:
            self.probas.append(squares_proba)
            if squares_kwargs is None:
                squares_kwargs = {'max_angle': 4, 'max_width': 30, 'min_width': 30, 'min_times': 1, 'max_times': 5}
            else:
                squares_kwargs = dict(squares_kwargs)
            squares_kwargs['draw_method'] = DrawMethod.SQUARE
            self.gens.append(RandomIrregularMaskEmbedder(**squares_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()

        self.invert_proba = invert_proba

    def __call__(self,
                 imgs,
                 masks=None,
                 iter_i=None,
                 raw_image=None,
                 verbose=False,
                 ) -> torch.Tensor:
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        
        if isinstance(gen, CocoSegmentationMaskEmbedder):
            result = gen(masks)
        else:
            result = gen(imgs[0], iter_i=iter_i, raw_image=raw_image)
            result = np.repeat(result[np.newaxis, :], imgs.shape[0], axis=0)
            result = torch.from_numpy(result).float()
        
        if self.invert_proba > 0 and (np.random.random() < self.invert_proba) and not isinstance(gen, FullMaskEmbedder):
            result = 1 - result
        
        if verbose:
            print(f"kind = {kind}, result = {result.mean().item()}")
        
        return result


def get_mask_embedder(kind, **kwargs):
    if kind is None:
        kind = "mixed"
    if kwargs is None:
        kwargs = {}

    if kind == "mixed":
        cl = MixedMaskEmbedder
    elif kind == "full":
        cl = FullMaskEmbedder
    else:
        raise NotImplementedError(f"No such embedder kind = {kind}")
    return cl(**kwargs)
