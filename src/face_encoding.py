import abc
from typing import List, Tuple

import dlib
import numpy as np

from src.util import Util


class FaceEncoding(metaclass=abc.ABCMeta):
    """
    人脸特征值提取face encoding抽象类
    """

    @staticmethod
    @abc.abstractmethod
    def encoding_frame_box(frame_box) -> List[Tuple[str, str, str, str]]:
        """
        获取frame_box中头像和对应box位置的特征码

        :param frame_box: frame_box,包含了图片和box信息
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def encoding(img, box):
        """
        获取图片中头像的特征码

        :param img: 图片,h,w,s三维信息
        :param box: 头像坐标
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def encoding_img(face_img):
        """
        获取图片中第一个头像的特征码(言外之意,图片本来就是头像图片)

        :param face_img: 图片,h,w,s三维信息
        """
        pass


class FaceEncodingFrFe(FaceEncoding):
    @staticmethod
    def encoding_frame_box(frame_box):
        if frame_box.img is not None and len(frame_box.img) > 0 and frame_box.box is not None and len(
                frame_box.box) > 0:
            return FaceEncodingFrFe.encoding(frame_box.img, frame_box.box)
        return FaceEncodingFrFe.encoding_img(frame_box.img)

    @staticmethod
    def encoding(img, box):
        fl_box = Util.cv_to_fl_box(box)
        face_encodings = face_recognition.face_encodings(img, [fl_box])
        return np.array(face_encodings[0]) if face_encodings else None

    @staticmethod
    def encoding_img(face_img):
        face_encodings = face_recognition.face_encodings(face_img)
        return np.array(face_encodings[0]) if face_encodings else None


class FaceEncodingDlibReg(FaceEncoding):
    face_detector = dlib.get_frontal_face_detector()
    shape = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")
    face_encoding = dlib.face_recognition_model_v1("../model/dlib_face_recognition_resnet_model_v1.dat")

    @staticmethod
    def cv_box_to_dlib(box):
        x, y, w, h = box
        rectangle = dlib.rectangle(x, y, x + w, y + h)
        return rectangle

    @staticmethod
    def encoding_frame_box(frame_box):
        if frame_box.img is not None and len(frame_box.img) > 0 and frame_box.box is not None and len(
                frame_box.box) > 0:
            return FaceEncodingDlibReg.encoding(frame_box.img, frame_box.box)
        return FaceEncodingDlibReg.encoding_img(frame_box.img)

    @staticmethod
    def encoding(img, box):
        rectangle = FaceEncodingDlibReg.cv_box_to_dlib(box)
        shape = FaceEncodingDlibReg.shape(img, rectangle)
        face_descriptor = FaceEncodingDlibReg.face_encoding.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)

    @staticmethod
    def encoding_img(face_img):
        boxes = FaceEncodingDlibReg.face_detector(face_img, 1)
        if not boxes:
            return None
        box = boxes[0]
        shape = FaceEncodingDlibReg.shape(face_img, box)
        face_descriptor = FaceEncodingDlibReg.face_encoding.compute_face_descriptor(face_img, shape)
        return np.array(face_descriptor)


class FaceEncodingFactory(object):
    """
    face encoding 的工厂类
    """
    face_encoding_construct = {"FR_FE": FaceEncodingFrFe,
                               "DLIB_REG": FaceEncodingDlibReg}

    @staticmethod
    def get_instance(encoding_method):
        """
        获取encoding_method对应的encoding实例

        :param encoding_method: 人脸特征码提取方法
        :return: 特征码提取方法实例
        """
        return FaceEncodingFactory.face_encoding_construct.get(encoding_method)()


