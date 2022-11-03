from albumentations import ImageOnlyTransform
import albumentations as A
import cv2


class SolidCrop(ImageOnlyTransform):
    
    def __init__(self, position_left_top, position_right_bottom):
        super(SolidCrop, self).__init__(True, 1.)
        self.x1, self.y1 = position_left_top
        self.x2, self.y2 = position_right_bottom
    
    def apply(self, img, **params):
        img = img[self.y1:self.y2, self.x1:self.x2]
        return img