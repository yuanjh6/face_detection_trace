import os
from collections import defaultdict
from typing import Tuple, Dict, List
import cv2


class Util(object):
    """
    工具类

    主要包括坐标系转换,画框,图片特定区域切割,以及文件夹遍历等工具类

    坐标系转换:由于调用了不同的工具包,不同包的坐标标识方法是不同的,为了保持内部变量含义统一性,程序内部均采用opencv坐标系
    """

    @staticmethod
    def cv_to_fl_box(rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
       坐标系转换

       将opencv坐标转为face_recognition的face_locations坐标

       :param rect: opencv坐标
       :return:face_recognition.face_locations中的坐标
        """
        x, y, w, h = rect
        left = x
        top = y
        right = w + left
        bottom = h + top
        return top, right, bottom, left

    @staticmethod
    def draw_boxes(frame, box: Tuple[int, int, int, int]):
        """
        矩形框绘制
        """
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    @staticmethod
    def cut_frame_box(frame, box: Tuple[int, int, int, int]):
        """
        从图片中截取出矩形区域

        :param frame: 图片,h,w,s格式的3维数组
        :param box: 矩形框
        :return: 矩形框内的图片,h,w,s格式的3维数组
        """
        return frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

    @staticmethod
    def get_dirs_files(dir: str) -> Dict[str, List[str]]:
        """
        获取dir下的所有文件列表

        :param dir: 目录路径
        :return:dict,key:person name,value:list,person face img
        """
        name_files_map = defaultdict(list)
        assert os.path.exists(dir) and os.path.isdir(dir), "dir is illegal"
        # for sub_dir in os.listdir(dir):
        #     if os.path.isdir(dir):
        #         for file_path in os.listdir(dir + sub_dir + "/"):
        #             if os.path.isfile(dir + sub_dir + "/" + file_path):
        #                 name_files_map[sub_dir].append(file_path)

        [name_files_map[sub_dir].append(dir + sub_dir + "/" + file_path) for sub_dir in os.listdir(dir) if
         os.path.isdir(dir) for file_path
         in os.listdir(dir + sub_dir + "/") if os.path.isfile(dir + sub_dir + "/" + file_path)]
        return name_files_map

    @staticmethod
    def get_file_path_split(filename: str) -> Tuple[str, str, str]:
        """
        将文件全路径拆分为文件目录路径,文件名,文件扩展名

        :param filename: 文件全路径
        :return: tuple,文件目录路径,文件名,文件扩展名
        """
        (filepath, tempfilename) = os.path.split(filename);
        (shotname, extension) = os.path.splitext(tempfilename);
        return filepath, shotname, extension


class LimitList(object):
    """
    长度受限制的list

    :cvar int maxsize: list最大长度
    """

    def __init__(self, maxsize=10):
        self.maxsize = maxsize
        self.__list = list()

    def append(self, item):
        """
        向list新增元素

        :param item: 新增元素
        :return: 是否添加成功,长度超过最大限度则返回false
        """
        if len(self.__list) >= self.maxsize:
            return False
        else:
            self.__list.append(item)
            return True

    def pop(self):
        """
        从list里弹出元素

        :return: 返回弹出的元素,如果list为空则返回None
        """
        if len(self.__list) == 0:
            return None
        return self.__list.pop()

    def __iter__(self):
        return self.__list.__iter__()


class FrameBox(object):
    """
    自定义对象,包含图片帧和头像位置信息

    :cvar list img: 图片,hws三维数组格式
    :cvar tuple box: 头像位置,长度为4的list
    """

    def __init__(self, img=None, box=None):
        assert (img is not None and box is not None), "bad param"
        self.img = img
        self.box = list(map(int, box))

    @property
    def name(self) -> str:
        """
        使用box的头像坐标生成图片文件名称

        :return: 图片文件名称
        """
        return "_".join(map(str, self.box)) + ".png"

    @staticmethod
    def parse_file(file_name):
        """
        从文件名和文件内容恢复出frame_box信息

        :param file_name: 文件名
        :return: tuple元祖,tuple[0]图片信息,hws三维信息,tuple[1]头像位置,长度为4的list
        """
        img = cv2.imread(file_name)
        file_path, file_name, file_ext = Util.get_file_path_split(file_name)
        box = file_name.split("_")
        return img, box
