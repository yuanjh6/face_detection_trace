import abc
import json
import os
from collections import defaultdict
from typing import Tuple, Dict, List

import dlib
import itertools
import logging
import threading
from datetime import datetime
from queue import Queue
import cv2
import face_recognition
import numpy as np

"""
人脸自动录入和识别,并生成事件形式的打卡记录

自动录入:未识别头像自动采集到到指定文件夹,如果是公司员工,修改文件夹名称将未识别头像变成公司内部员工
打卡记录:通过人脸检测,跟踪,识别(特征值提取),比对底库,等生成人员打卡记录

帮助:python face_detection_trace.py -h
optional arguments:
  -c CONFIG_PATH, --config_path :配置文件路径
  -fe FACE_ENCODING, --face_encoding :人脸特征值提取算法
  -pimg PERSON_IMAGE_DIR, --person_image_dir :人脸图片保存路径
  -vimg VIDEO_IMAGE_DIR, --video_image_dir :视频图片保存路径

使用示例:python face_detection_trace.py -c config.json 
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(handler)


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
    cascade_xml = "model/haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_xml)

    def detection(self, frame):
        boxes = FaceDetectionCvCas.face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.15, 5)
        return boxes


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
    shape = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    face_encoding = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")

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


class Person(object):
    """
    人员,可能是员工或者陌生人

    :cvar str ipc_name: 摄像头名称
    :cvar bool is_new: 是否是新人
    :cvar str person_name:人员姓名
    :cvar list frames_box_limit:如果是新人,建立list,保存新人头像
    :cvar List __encodings:frames_box_limit里头像对应的头像特征码信息
    """
    __unknow_max_id = 0
    face_encoding = None
    img_dir = ""

    def __init__(self, person_name, img_files, ipc_name, is_new=False, new_face_frame_max=10):
        self.ipc_name = ipc_name
        self.is_new = is_new
        self.person_name = person_name
        assert np.all([os.path.exists(img) for img in img_files]), "img file is error"
        # assert np.all([img.find("unknow") == -1 for img in self.imgs]), "img file can"t contain unknow"
        frames_box = [FrameBox(*FrameBox.parse_file(img_file)) for img_file in img_files]
        encodings = [Person.face_encoding.encoding_frame_box(frame_box) for frame_box in frames_box]
        self.__encodings = encodings

        if is_new:
            self.frames_box_limit = LimitList(new_face_frame_max)
            self.__encodings = []

        logger.info("add new persion %s" % (str([np.sum(encoding) for encoding in self.encodings_valid()])))

    def new_frame_box(self, frame_box):
        """
        向frames_box_limit中新增frame_box信息

        :param frame_box: FrameFox信息,包含图片和图片里头像位置信息
        """
        if self.is_new:
            self.frames_box_limit.append(frame_box) and self.__encodings.append(
                Person.face_encoding.encoding_frame_box(frame_box))

    def encodings_valid(self):
        """
        返回有效的encoding信息

        :return: encoding中有效的头像编码
        """
        return [x for x in self.__encodings if x is not None and len(x)]

    @staticmethod
    def new_unknow_person(ipc_name):
        """
        新增陌生人

        :param ipc_name:摄像头名称
        :return:陌生人person实例
        """
        return Person(Person.get_unknow_name(), list(), ipc_name=ipc_name, is_new=True)

    @staticmethod
    def get_unknow_name() -> str:
        """
        生成陌生人名字

        :return: 陌生人名字
        """
        Person.__unknow_max_id += 1
        return "unknow{0:03d}".format(Person.__unknow_max_id)

    def save(self):
        """
        保存frames_box_limit中的图片和头像位置信息
        """
        if self.is_new:
            dir_path = Person.img_dir + self.ipc_name + "/" + self.person_name + "/"
            os.path.exists(dir_path) or os.makedirs(dir_path)
            for frame_box in self.frames_box_limit:
                cv2.imwrite(dir_path + frame_box.name, frame_box.img)
            self.is_new = False

    @staticmethod
    def get_camera_person_files(cameras_dir):
        """
        加载某个摄像头下某人的所有头像图片信息

        :param cameras_dir: 图片目录
        :return: dict,key:camera_name,摄像头名称,value:dict02,dict02:key:person_name,姓名,value:img_list,此人对应头像列表
        """
        assert os.path.exists(cameras_dir) and os.path.isdir(cameras_dir), "dir is illegal"
        camera_person_dict = defaultdict(dict)
        for camera_dir in os.listdir(cameras_dir):
            if os.path.isdir(cameras_dir + camera_dir):
                camera_person_dict[camera_dir] = Util.get_dirs_files(cameras_dir + "/" + camera_dir + "/")
        return camera_person_dict


class Track(object):
    """
    跟踪器

    内部包含了opencv追踪器,或者说对opencv追踪器的二次封装

    :cvar str ipc_name: 摄像头名称
    :cvar int __id:跟踪器id
    :cvar object tracker:跟踪器,opencv追踪器实例
    :cvar list face_img:图片里的头像小图信息
    :cvar list img:图片
    :cvar list encoding:人脸头像对应特征码
    :cvar list __history:临近的各帧是否包含此人
    :cvar object match_person:跟踪器匹配的人
    :cvar callable event_call_back:追踪的人消失之后的回调函数,比如生成事件日志
    """
    __id = 0

    def __init__(self, ipc_name, tracker, img, box, encoding, persons, event_call_back, history=5):
        self.ipc_name = ipc_name
        self.__id = Track.__id = Track.__id + 1
        self.tracker = tracker
        self.face_img = img[(box[1]):(box[1] + box[3]), (box[0]):(box[0] + box[2])]
        self.img = img
        self.encoding = encoding
        self.__history = [False] * history
        self.__history_iter = itertools.cycle(range(history))
        self.match_person = None
        # 暂不用self.__history_have=bool
        self.__history[next(self.__history_iter)] = True

        self.__init_tracker(img, box)

        self.find_person(persons)
        self.event_call_back = event_call_back
        self.event_call_back(0, self.ipc_name, self.__id, self.face_img, box, self.match_person.person_name)

    def __init_tracker(self, img, box):
        """
        使用图片和图片里的头像位置,初始化内置的opencv追踪器

        :param img: 图片
        :param box: 头像位置
        """
        self.tracker.init(img, tuple(box))

    def update_img(self, img, box, encoding):
        """
        更新追踪器信息

        :param img:图片
        :param box:人脸位置
        :param encoding:人脸特征编码
        """
        self.face_img = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        self.match_person.new_frame_box(FrameBox(img, box))
        self.encoding = encoding
        self.img = img
        iter_num = next(self.__history_iter)
        if self.alive() and self.__history[iter_num] == 1 and sum(self.__history) == 1:
            self.event_call_back(1, self.ipc_name, self.id, self.img, box, self.match_person.person_name)
            self.__history[iter_num] = False

    def update(self, img):
        """
        更新追踪器图片

        :param img: 图片
        :return:
        """
        return self.tracker.update(img)

    def find_person(self, persons, tolerance=0.6):
        """
        从入参的persons中匹配track追踪器追踪的人(将track和person关联起来)

        :param persons:所有人员列表
        :param tolerance:人员匹配阈值,如果人员距离小于此阈值则认为是同一个人
        :return:
        """
        if self.encoding is None or len(self.encoding) == 0:
            return
        person_dist = [min(face_recognition.face_distance(person.encodings_valid(), self.encoding), default=1.0) for
                       person in
                       persons]
        logger.info("match_person self.encodings %s " % (str(np.sum(self.encoding))))
        if min(person_dist, default=1.0) < tolerance:
            min_dist_index = np.argmin(person_dist)
            self.match_person = persons[min_dist_index]
        else:
            self.match_person = Person.new_unknow_person(ipc_name=self.ipc_name)
            persons.append(self.match_person)

    def alive(self):
        """
        追踪器是否处于活跃活跃状态

        追踪器匹配的人最近几帧是否出现过

        :return:
        """
        return sum(self.__history) > 0

    @property
    def id(self):
        return self.__id


class CapDetectionTrack(threading.Thread):
    """
    检测追踪类

    集成人脸检测和追踪

    :cvar bool is_start:是否已经开启
    :cvar object face_detector:人脸检测器
    :cvar object face_encoding:人脸编码器
    :cvar object __last_frame:视频流的最新帧
    :cvar queue frame_queue:视频流的视频帧队列
    :cvar bool is_realtime:是否是实时模式
    :cvar list tracks:追踪器列表,一个视频会出现多个人,自然也有多个追踪器
    :cvar list ipc_info:视频源配置信息
    :cvar list persons:已保存的人员和对应的人脸信息
    """
    video_imgs = None

    def __init__(self, ipc_info, is_realtime, face_detector, face_encoding, detection_freq, persons, face_decector_lock,
                 face_encoding_lock):
        super(CapDetectionTrack, self).__init__()
        self.is_start = False

        self.face_detector = face_detector
        self.face_encoding = face_encoding
        self.face_detector_lock = face_decector_lock
        self.face_encoding_lock = face_encoding_lock

        self.__last_frame = None
        self.frame_queue = Queue(maxsize=60)
        self.detection_freq_iter = itertools.cycle(range(detection_freq))

        self.is_realtime = is_realtime

        self.tracks = list()
        self.ipc_info = ipc_info
        self.persons = persons

    @property
    def name(self):
        return self.ipc_info["name"]

    @property
    def path(self):
        return self.ipc_info["path"]

    @property
    def is_save_stranger(self):
        return bool(self.ipc_info.get("save_stranger", 0))

    def run(self):
        """
        启动视频解码,人脸检测和人脸特征码提取线程

        """
        self.is_start = True

        cv_cap = cv2.VideoCapture(self.path)
        size = (int(cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_file_path = "%s_output.avi" % self.path
        video_write = cv2.VideoWriter(
            out_file_path,
            cv2.VideoWriter_fourcc("M", "P", "4", "2"),  # 编码器
            cv_cap.get(cv2.CAP_PROP_FPS),
            size
        )

        cap_thread = threading.Thread(target=self.__start_capture, args=(cv_cap,))
        cap_thread.start()

        dec_thread = threading.Thread(target=self.__start_detection_trace, args=(video_write,))
        dec_thread.start()

        cap_thread.join()
        dec_thread.join()
        self.save_release_resouce()

    def __start_capture(self, cv_cap):
        """
        启动视频解码线程

        :param cv_cap: 视频解码器
        """
        if self.is_realtime:
            while self.is_start:
                ret, frame = cv_cap.read()
                if ret:
                    self.__last_frame = frame
                else:
                    self.__last_frame = None
                    self.is_start = False  # stop threading:_start_capture,_start_detection_trace
                    # todo enhance,some tail frame wo"t be run by threading _start_detection_trace

        else:
            while self.is_start:
                ret, frame = cv_cap.read()
                if ret:
                    self.frame_queue.put(frame)
                else:
                    self.frame_queue.put(None)
                    self.is_start = False
        cv_cap.release()

    def __get_last_frame(self):
        """
        返回视频流最新帧

        :return:
        """
        ret = self.__last_frame if self.is_realtime else self.frame_queue.get()
        self.__last_frame = None
        return ret

    def __start_detection_trace(self, video_write):
        """
        启动视频的人脸检测和追踪线程

        :param video_write: 处理后的视频写入文件
        """
        while self.is_start:
            last_frame = self.__get_last_frame()
            if last_frame is None:
                continue
            if next(self.detection_freq_iter) == 0:
                boxes = self.__face_dec(last_frame)
            else:
                boxes = self.__face_track(last_frame)
            [Util.draw_boxes(last_frame, list(box)) for box in boxes]
            video_write.write(last_frame)
            # cv2.imshow(self.name, last_frame)
            # cv2.waitKey(1)
        video_write.release()

    def __face_dec(self, img):
        """
        更新人脸检测图片帧信息

        :param img: 图片帧
        :return:
        """
        with self.face_detector_lock:
            boxes = self.face_detector.detection(img)
        self.__face_upgrade_track(img, boxes)
        return boxes

    @staticmethod
    def event_call_back(type, ipc_name, track_id, img=None, box=None, person_name=None):
        """
        追踪的人消失后的回调函数

        :param type: 事件类型
        :param ipc_name: 摄像头名称
        :param track_id: 追踪器id
        :param img: 图片帧
        :param box: 图片帧中的人脸位置
        :param person_name: 人名
        """
        # type=0 enter, 1 out
        cv2.imwrite("%snew_face_%s.png" % (CapDetectionTrack.video_imgs, track_id), img)
        logger.info("lost person" + ",".join((ipc_name, str(type),
                                              datetime.now().strftime("%Y%m%d%H%M%S"), str(track_id),
                                              "%snew_face_%s.png" % (CapDetectionTrack.video_imgs, track_id), str(box),
                                              person_name)))

    def __face_upgrade_track(self, img, boxes):
        """
        用图片帧更新追踪器

        图片中可能存在多张人脸
        如果人脸和已有track中的某人脸匹配,则用此frame更新已有track
        如果人脸无法和已有track中的人脸匹配,新建立track

        :param img: 图片帧
        :param boxes: 图片帧中的人脸位置信息
        """
        tracks_map = {track.id: track for track in self.tracks}
        track_ids, track_encodings = list(map(lambda x: x.id, self.tracks)), list(
            map(lambda x: x.encoding, self.tracks))
        boxes_imgs_encoding = list()
        if boxes is not None and len(boxes):
            with self.face_encoding_lock:
                boxes_imgs_encoding = [self.face_encoding.encoding(img, box) for box in boxes]
            boxes_encoding_filter = [boxes_img_encoding is not None and len(boxes_img_encoding) > 0 for
                                     boxes_img_encoding in boxes_imgs_encoding]
            boxes = np.array(boxes)[boxes_encoding_filter]
            boxes_imgs_encoding = np.array(boxes_imgs_encoding)[boxes_encoding_filter]
            # logger.info("###", np.array(boxes_imgs_encoding).shape)
            # boxes_imgs_encodings = [
            #     face_recognition.face_encodings(frame[(box[1]):(box[1] + box[3]), (box[0]):(box[0] + box[2])]) for box in boxes]
            # boxes_imgs_encoding = [boxes_imgs_encoding[0] for boxes_imgs_encoding in boxes_imgs_encodings if
            #                        boxes_imgs_encoding[:1]]
            # logger.info("####", np.array(boxes_imgs_encoding).shape)
            # [cv2.imwrite(str(np.sum(boxes_encoding)) + ".png", Util.cut_frame_box(frame, box))
            #  for boxes_encoding, box in zip(boxes_imgs_encoding, boxes)]

        box_track_ids = [
            np.array(track_ids, int)[face_recognition.compare_faces(track_encodings, unknown_face_encoding)] for
            unknown_face_encoding in boxes_imgs_encoding]

        new_trackers = []
        for box_track_id, boxes_img, boxes_img_encoding in zip(box_track_ids, boxes, boxes_imgs_encoding):
            if len(box_track_id) > 0:
                tracks_map[box_track_id[0]].update_img(img, boxes_img, boxes_img_encoding)
            else:
                new_trackers.append(Track(self.name, cv2.TrackerKCF_create(), img, boxes_img, boxes_img_encoding,
                                          self.persons, event_call_back=self.event_call_back))

        # keep alive tracks only
        if self.is_save_stranger:
            [tracker.match_person.save() for tracker in self.tracks if not tracker.alive()]
        self.tracks = [tracker for tracker in self.tracks if tracker.alive()]
        self.tracks.extend(new_trackers)

    def __face_track(self, img):
        """
        使用frame更新已有人脸追踪器track

        :param img: 图片帧
        :return:
        """
        boxes = [list(map(int, track.update(img)[1])) for track in self.tracks]
        return boxes

    def save_release_resouce(self):
        """
        资源保存和释放,保存追踪器里关联的陌生人的人脸图片
        """
        del self.detection_freq_iter
        if self.is_save_stranger:
            [tracker.match_person.save() for tracker in self.tracks]
        del self.tracks
        del self.frame_queue


class DetectionTracksCtl(object):
    """
    人脸检测,识别控制器

    :cvar object face_detector:人脸识别器
    :cvar object face_encoding:人脸编码器
    """

    def __init__(self, face_detector, face_encoding):
        self.face_detector = face_detector
        self.face_encoding = face_encoding

    def start_all(self, ipc_infos, camera_persons):
        """
        启动所有视频流的人脸检测线程

        :param ipc_infos: 视频流配置信息
        :param camera_persons: 各视频流对应对应人员信息
        """
        # ipc_infos:list.map.key=ipc_url/ipc_name,list.map.value="xx/xx.mp4"/test01
        threads = list()

        face_decector_lock = threading.Lock()
        face_encoding_lock = threading.Lock()
        for ipc_info in ipc_infos:
            ipc_name = ipc_info["name"]
            is_realtime = ipc_info["realtime"]
            detection_freq = ipc_info["detection_freq"]
            thread = CapDetectionTrack(ipc_info, is_realtime, face_detector, face_encoding, detection_freq,
                                       camera_persons[ipc_name], face_decector_lock, face_encoding_lock)
            threads.append(thread)
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        self.__after_all_stop()

    def __after_all_stop(self):
        print("__after_all_stop")
        # self.event_df["img_file"] = self.event_df["img_file"].apply(lambda x: "<img src="%s">" % x)
        # event_name_map = {0: "进", 1: "出"}
        # self.event_df["event_name"] = self.event_df["event_name"].map(event_name_map)
        # self.event_df.to_html("event.html")

    # todo 01 use diff change to decation detection ,02,use yield change param
    # ref Human-detection-and-Tracking
    @staticmethod
    def background_subtraction(previous_frame, frame_resized_grayscale, min_area):
        """
        通过视频帧变化比率,判断是否需要启动检测线程

        如果视频帧不发生变化,说明没有人出现,不需要进行人脸检测和追踪识别

        :param previous_frame: 上一帧图片信息
        :param frame_resized_grayscale:
        :param min_area: 最小变化区域阈值
        :return: 图片变化率
        """
        frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        temp = 0
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) > min_area:
                temp = 1
        return temp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", help="the path of the config file(json format config)",
                        default="config.json")
    parser.add_argument("-fe", "--face_encoding", help="face encoding method")
    parser.add_argument("-fd", "--face_detection", help="face detection method")
    parser.add_argument("-pimg", "--person_image_dir", help="person images save dir")
    parser.add_argument("-vimg", "--video_image_dir", help="video images save dir")
    args = parser.parse_args()
    config_path = args.config_path

    assert os.path.exists(config_path), "config file not exists"
    config_json = json.load(open(config_path, "r"))
    ipc_infos = config_json["ipcs"]

    face_encoding_config = args.face_encoding or config_json.get("face_encoding", "DLIB_REG")
    face_detection_config = args.face_detection or config_json.get("face_detection", "CV_CAS")
    person_image_dir_config = args.person_image_dir or config_json.get("person_image_dir", "data/person_img/")
    video_image_dir_config = args.video_image_dir or config_json.get("video_image_dir", "data/video_imgs/")

    os.path.exists(person_image_dir_config) or os.makedirs(person_image_dir_config)
    os.path.exists(video_image_dir_config) or os.makedirs(video_image_dir_config)

    CapDetectionTrack.video_imgs = video_image_dir_config

    face_encoding = FaceEncodingFactory.get_instance(face_encoding_config)
    Person.face_encoding = face_encoding
    Person.img_dir = person_image_dir_config

    camera_persons = defaultdict(list)

    # 加载摄像头和对应人员以及人员人脸头像信息
    camere_persons_files = Person.get_camera_person_files(person_image_dir_config)
    [camera_persons[camera_name].append(Person(person_name, person_files, camera_name))
     for camera_name, persons_map in camere_persons_files.items()
     for person_name, person_files in persons_map.items()]

    # camera_persons = {"test1": persons, "test6": persons}
    face_detector = FaceDetectionFactory.get_detection(face_detection_config)
    detection_track = DetectionTracksCtl(face_detector, face_encoding)
    detection_track.start_all(ipc_infos, camera_persons=camera_persons)
