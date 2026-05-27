import cv2
import numpy as np
import random

from albumentations import Compose as Compose_albu
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    GridDistortion,
    RandomBrightnessContrast,
    RandomGamma,
    Crop,
    LongestMaxSize,
    ShiftScaleRotate,
    #Flip,
    Rotate,
    VerticalFlip,
    RandomRotate90,
    Resize
)


def to_numpy(data):
    image, label = data['image'], data['label']
    data['image'] = np.array(image)
    if data['label'] is not None:
        data['label'] = np.array(label)
    return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class MedicalTransform:
    def __init__(self, output_size, roi_error_range=0, use_roi=False):
        if isinstance(output_size, (tuple, list)):
            self._output_size = output_size  # (h, w)
        else:
            self._output_size = (output_size, output_size)
        
        self._roi_error_range = roi_error_range
        self._type = 'train'
        self.use_roi = use_roi
    
    def train(self):
        self._type = 'train'
        return self
    
    def eval(self):
        self._type = 'eval'
        return self
    
    def __call__(self, data):
        data = to_numpy(data)
        img, label = data['image'], data['label']
        #print(len(img.shape))

        
        is_3d = True if len(img.shape) == 4 else False
        
        max_size = max(self._output_size[0], self._output_size[1])
        
        if self._type == 'train':
            task = [
                HorizontalFlip(p=0.5),  # 随机垂直翻转
                RandomBrightnessContrast(p=0.5),  # 随机更改亮度对比度
                RandomGamma(p=0.5),  # 随机Gama变换
                GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.5),  # 随机网格失真
                # 随机旋转
                #LongestMaxSize(max_size, p=1),  # 缩放图像
                #PadIfNeeded(self._output_size[0], self._output_size[1], cv2.BORDER_CONSTANT, value=0, p=1),  # 填充
                Resize(self._output_size[0], self._output_size[1]),
                ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT,
                                 value=0, p=0.5)  # 随机平移缩放旋转图片
            ]
        else:
            task = [
                #LongestMaxSize(max_size, p=1),
                #PadIfNeeded(self._output_size[0], self._output_size[1], cv2.BORDER_CONSTANT, value=0, p=1)
                Resize(self._output_size[0], self._output_size[1]),
            ]
        
        if self.use_roi:
            assert 'roi' in data.keys() and len(data['roi']) is not 0
            roi = data['roi']
            min_y = 0
            max_y = img.shape[0]
            min_x = 0
            max_x = img.shape[1]
            min_x = max(min_x, roi['min_x'] - self._roi_error_range)
            max_x = min(max_x, roi['max_x'] + self._roi_error_range)
            min_y = max(min_y, roi['min_y'] - self._roi_error_range)
            max_y = min(max_y, roi['max_y'] + self._roi_error_range)
            
            crop = [Crop(min_x, min_y, max_x, max_y, p=1)]
            task = crop + task
        
        aug = Compose_albu(task)
        if not is_3d:
            aug_data = aug(image=img, mask=label)
            data['image'], data['label'] = aug_data['image'], aug_data['mask']
        
        else:
            img = img.squeeze().transpose((1,2,0))
            label = label.squeeze().transpose((1,2,0))
            keys = {}
            targets = {}
            for i in range(1, img.shape[2]):
                keys.update({f'image{i}': 'image'})
                keys.update({f'mask{i}': 'mask'})
                targets.update({f'image{i}': img[:, :, i]})
                targets.update({f'mask{i}': label[:, :, i]})
            aug.add_targets(keys)
            
            targets.update({'image': img[:, :, 0]})
            targets.update({'mask': label[:, :, 0]})
            
            aug_data = aug(**targets)
            imgs = [aug_data['image']]
            labels = [aug_data['mask']]
            
            for i in range(1, img.shape[2]):
                imgs.append(aug_data[f'image{i}'])
                labels.append(aug_data[f'mask{i}'])
            
            img = np.stack(imgs, axis=-1)
            label = np.stack(labels, axis=-1)
            data['image'] = img
            data['label'] = label
        
        return data
    
    @property
    def roi_error_range(self):
        return self._roi_error_range
    
    @property
    def output_size(self):
        return self._output_size


class MedicalTransformTwotask:
    def __init__(self, output_size, roi_error_range=0, use_roi=False):
        if isinstance(output_size, (tuple, list)):
            self._output_size = output_size  # (h, w)
        else:
            self._output_size = (output_size, output_size)

        self._roi_error_range = roi_error_range
        self._type = 'train'
        self.use_roi = use_roi

    def train(self):
        self._type = 'train'
        return self

    def eval(self):
        self._type = 'eval'
        return self

    def __call__(self, data):
        data = to_numpy(data)
        img, label, mask1 = data['image'], data['label'], data['mask1']
        # print(len(img.shape))

        is_3d = True if len(img.shape) == 4 else False

        max_size = max(self._output_size[0], self._output_size[1])

        if self._type == 'train':
            task = [
                HorizontalFlip(p=0.5),  # 随机垂直翻转
                RandomBrightnessContrast(p=0.5),  # 随机更改亮度对比度
                RandomGamma(p=0.5),  # 随机Gama变换
                GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.5),  # 随机网格失真
                # 随机旋转
                # LongestMaxSize(max_size, p=1),  # 缩放图像
                # PadIfNeeded(self._output_size[0], self._output_size[1], cv2.BORDER_CONSTANT, value=0, p=1),  # 填充
                Resize(self._output_size[0], self._output_size[1]),
                ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT,
                                 value=0, p=0.5)  # 随机平移缩放旋转图片
            ]
        else:
            task = [
                # LongestMaxSize(max_size, p=1),
                # PadIfNeeded(self._output_size[0], self._output_size[1], cv2.BORDER_CONSTANT, value=0, p=1)
                Resize(self._output_size[0], self._output_size[1]),
            ]

        aug = Compose_albu(task, additional_targets={"mask1": "mask", "label": "mask"})
        aug_data = aug(image=img, mask1=mask1, label=label)
        data['image'], data['label'], data['mask1'] = aug_data['image'], aug_data['label'], aug_data['mask1']

        return data

    @property
    def roi_error_range(self):
        return self._roi_error_range

    @property
    def output_size(self):
        return self._output_size

class MedicalTransformCompose:
    """
    只做普通的四种增强，翻转和旋转90度
    """
    def __init__(self, output_size, roi_error_range=0, use_roi=False):
        if isinstance(output_size, (tuple, list)):
            self._output_size = output_size  # (h, w)
        else:
            self._output_size = (output_size, output_size)

        self._roi_error_range = roi_error_range
        self._type = 'train'
        self.use_roi = use_roi

        self.total_task = [
            HorizontalFlip(p=1),
            VerticalFlip(p=1),
            RandomRotate90(p=1)
        ]

    def train(self):
        self._type = 'train'
        return self

    def eval(self):
        self._type = 'eval'
        return self

    def __call__(self, data):
        data = to_numpy(data)
        img, label = data['image'], data['label']

        max_size = max(self._output_size[0], self._output_size[1])

        if self._type == 'train':
            task = [
                self.total_task[random.randint(0,2)],
                Resize(self._output_size[0], self._output_size[1])
            ]
        else:
            task = [
                LongestMaxSize(max_size, p=1),
                Resize(self._output_size[0], self._output_size[1])
            ]



        aug = Compose_albu(task)
        aug_data = aug(image=img, mask=label)
        data['image'], data['label'] = aug_data['image'], aug_data['mask']

        return data

    @property
    def roi_error_range(self):
        return self._roi_error_range

    @property
    def output_size(self):
        return self._output_size


class MedicalTransformComposeTwoMask:
    """
    只做普通的四种增强，翻转和旋转90度
    """
    def __init__(self, output_size, roi_error_range=0, use_roi=False):
        if isinstance(output_size, (tuple, list)):
            self._output_size = output_size  # (h, w)
        else:
            self._output_size = (output_size, output_size)

        self._roi_error_range = roi_error_range
        self._type = 'train'
        self.use_roi = use_roi

        self.total_task = [
            HorizontalFlip(p=1),
            VerticalFlip(p=1),
            RandomRotate90(p=1)
        ]

    def train(self):
        self._type = 'train'
        return self

    def eval(self):
        self._type = 'eval'
        return self

    def __call__(self, data):
        data = to_numpy(data)
        img, label,mask1 = data['image'], data['label'],data['mask1']

        max_size = max(self._output_size[0], self._output_size[1])

        if self._type == 'train':
            task = [
                self.total_task[random.randint(0,2)],
                Resize(self._output_size[0], self._output_size[1])
            ]
        else:
            task = [
                LongestMaxSize(max_size, p=1),
                Resize(self._output_size[0], self._output_size[1])
            ]



        aug = Compose_albu(task, additional_targets={"mask1":"mask", "label":"mask"})
        aug_data = aug(image=img, mask1=mask1, label=label)
        data['image'], data['label'],data['mask1'] = aug_data['image'], aug_data['label'], aug_data['mask1']

        return data

    @property
    def roi_error_range(self):
        return self._roi_error_range

    @property
    def output_size(self):
        return self._output_size


class MedicalTransformResize:
    """
    只做普通的四种增强，翻转和旋转90度
    """
    def __init__(self, output_size, roi_error_range=0, use_roi=False):
        if isinstance(output_size, (tuple, list)):
            self._output_size = output_size  # (h, w)
        else:
            self._output_size = (output_size, output_size)

        self._roi_error_range = roi_error_range
        self._type = 'train'
        self.use_roi = use_roi

    def train(self):
        self._type = 'train'
        return self

    def eval(self):
        self._type = 'eval'
        return self

    def __call__(self, data):
        data = to_numpy(data)
        img, label = data['image'], data['label']

        task = [Resize(self._output_size[0], self._output_size[1])]

        aug = Compose_albu(task)
        aug_data = aug(image=img, mask=label)
        data['image'], data['label'] = aug_data['image'], aug_data['mask']

        return data

    @property
    def roi_error_range(self):
        return self._roi_error_range

    @property
    def output_size(self):
        return self._output_size


class MedicalTransformComposeRandom:
    """
    只做普通的四种增强，翻转和旋转90度
    """
    def __init__(self, output_size, roi_error_range=0, use_roi=False):
        if isinstance(output_size, (tuple, list)):
            self._output_size = output_size  # (h, w)
        else:
            self._output_size = (output_size, output_size)

        self._roi_error_range = roi_error_range
        self._type = 'train'
        self.use_roi = use_roi

        self.total_task = [
            HorizontalFlip(p=1),
            VerticalFlip(p=1),
            RandomRotate90(p=1)
        ]

    def train(self):
        self._type = 'train'
        return self

    def eval(self):
        self._type = 'eval'
        return self

    def __call__(self, data, index):
        data = to_numpy(data)
        img, label = data['image'], data['label']

        max_size = max(self._output_size[0], self._output_size[1])

        if self._type == 'train':
            task = [
                self.total_task[index]
            ]
        else:
            task = [
                LongestMaxSize(max_size, p=1),
                PadIfNeeded(self._output_size[0], self._output_size[1], cv2.BORDER_CONSTANT, value=0, p=1)
            ]



        aug = Compose_albu(task)
        aug_data = aug(image=img, mask=label)
        data['image'], data['label'] = aug_data['image'], aug_data['mask']

        return data

    @property
    def roi_error_range(self):
        return self._roi_error_range

    @property
    def output_size(self):
        return self._output_size

class CropTransform:
    def __init__(self, output_size, roi_error_range=0, use_roi=False):
        if isinstance(output_size, (tuple, list)):
            self._output_size = output_size  # (h, w)
        else:
            self._output_size = (output_size, output_size)

        self._roi_error_range = roi_error_range
        self._type = 'train'
        self.use_roi = use_roi

    def train(self):
        self._type = 'train'
        return self

    def eval(self):
        self._type = 'eval'
        return self

    def get_liver_list(self, label):
        index = np.where((label == 1) | (label == 2))
        x = index[0]
        y = index[1]
        if len(x) > 1:
            sed = np.random.randint(0, len(x) - 1)
        else:
            sed = 0
        origion_x = label.shape[0]
        origion_y = label.shape[1]
        cen_x = x[sed]
        cen_y = y[sed]
        target_x = self.output_size[0]
        target_y = self.output_size[1]
        x_min = max(0, int(cen_x - target_x/2))
        x_max = min(origion_x-1, x_min + target_x - 1)
        y_min = max(0, int(cen_y - target_y/2))
        y_max = min(origion_y-1, y_min + target_y -1)
        return(x_min, x_max, y_min, y_max)

    def __call__(self, data):
        data = to_numpy(data)
        img, label = data['image'], data['label']
        (min_x, max_x, min_y,  max_y) = self.get_liver_list(label)

        max_size = max(self._output_size[0], self._output_size[1])

        if self._type == 'train':
            task = [
                Crop(min_x, min_y, max_x, max_y, p=1),
                #HorizontalFlip(p=0.5),  # 随机垂直翻转
                #RandomBrightnessContrast(p=0.5),  # 随机更改亮度对比度
                #RandomGamma(p=0.5),  # 随机Gama变换
                #GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.5),  # 随机网格失真
                Flip(p=0.8), #随机翻转
                Rotate(limit=90, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), #随机旋转
                LongestMaxSize(max_size, p=1),  # 缩放图像
                PadIfNeeded(self._output_size[0], self._output_size[1], cv2.BORDER_CONSTANT, value=0, p=1),  # 填充
                #ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT,
                #                 value=0, p=0.5)  # 随机平移缩放旋转图片
            ]
        else:
            task = [
                Crop(min_x, min_y, max_x, max_y, p=1),
                LongestMaxSize(max_size, p=1),
                PadIfNeeded(self._output_size[0], self._output_size[1], cv2.BORDER_CONSTANT, value=0, p=1)
            ]


        aug = Compose_albu(task)
        aug_data = aug(image=img, mask=label)
        data['image'], data['label'] = aug_data['image'], aug_data['mask']
        return data

    @property
    def roi_error_range(self):
        return self._roi_error_range

    @property
    def output_size(self):
        return self._output_size