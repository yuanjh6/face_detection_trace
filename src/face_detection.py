import abc
from typing import List, Tuple
import cv2
import os
import dlib


class FaceDetection(metaclass=abc.ABCMeta):
    """
    人脸检测FaceDetection抽象类
    """

    @abc.abstractmethod
    def detection(self, frame) -> List[Tuple[str, str, str, str]]:
        """
        图片中所有的人脸矩形框坐标

        :param frame: 图片,h,w,s三维数组格式
        :return: List[tuple],每个tuple都是(int,int,int,int)形式的opencv的矩形坐标
        """
        pass


class FaceDetectionFrFoc(FaceDetection):
    @staticmethod
    def fl_to_cv_box(rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        坐标系转换

        将face_recognition的face_locations坐标转为opencv坐标系坐标

        :param rect: face_locations中的坐标
        :return:opencv中的坐标系
        """
        top, right, bottom, left = rect
        x = left
        y = top
        w = right - left
        h = bottom - top
        return x, y, w, h

    def detection(self, frame) -> List[Tuple[str, str, str, str]]:
        boxes = face_recognition.face_locations(frame)
        boxes = [FaceDetectionFrFoc.fl_to_cv_box(box) for box in boxes]
        return boxes


class FaceDetectionDlibFro(FaceDetection):
    face_detector = dlib.get_frontal_face_detector()

    @staticmethod
    def dlib_box_to_cv(rectangle: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        坐标系转换

        将dlib的rectangle坐标转为opencv坐标系坐标

        :param rectangle: dlib的rectangle坐标
        :return:opencv中的坐标系
        """
        x, y, w, h = rectangle.left(), rectangle.top(), rectangle.right() - rectangle.left(), rectangle.bottom() - rectangle.top()
        return x, y, w, h

    def detection(self, frame):
        rectangles = FaceDetectionDlibFro.face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
        if not rectangles:
            return list()
        else:
            return [FaceDetectionDlibFro.dlib_box_to_cv(rectangle) for rectangle in rectangles]
        return boxes


class FaceDetectionCvCas(FaceDetection):
    cascade_xml = os.path.normpath('%s/%s'%(os.path.dirname(__file__),"../model/haarcascade_frontalface_default.xml"))
    face_detector = cv2.CascadeClassifier(cascade_xml)

    def detection(self, frame):
        boxes = FaceDetectionCvCas.face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.15, 5)
        return boxes


class FaceDetectionFactory(object):
    """
    face detection 的工厂类
    """
    face_detection_construct = {"FR_FL": FaceDetectionFrFoc,
                                "CV_CAS": FaceDetectionCvCas,
                                "DLIB_FRO": FaceDetectionDlibFro}

    @staticmethod
    def get_detection(detection_method):
        """
        获取detection_method对应的detection实例

        :param detection_method: 人脸检测方法
        :return: 检测方法实例
        """
        return FaceDetectionFactory.face_detection_construct.get(detection_method)()
