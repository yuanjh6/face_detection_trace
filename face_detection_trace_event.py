import os
from collections import defaultdict

import dlib
import itertools
import logging
import threading
from datetime import datetime
from queue import Queue
import pandas as pd
import cv2
import face_recognition
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
VIDEO_IMG = 'video_img'
PERSON_IMG_DIR = 'data/person_img/'


class Util(object):
    @staticmethod
    def fl_to_cv_box(rect):  # 获得人脸矩形的坐标信息
        top, right, bottom, left = rect
        x = left
        y = top
        w = right - left
        h = bottom - top
        return x, y, w, h

    @staticmethod
    def cv_to_fl_box(rect):  # 获得人脸矩形的坐标信息
        x, y, w, h = rect
        left = x
        top = y
        right = w + left
        bottom = h + top
        return top, right, bottom, left

    @staticmethod
    def draw_boxes(frame, box):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    @staticmethod
    def cut_frame_box(frame, box):
        return frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

    @staticmethod
    def get_dirs_files(dir):
        name_files_map = defaultdict(list)
        assert os.path.exists(dir) and os.path.isdir(dir), 'dir is illegal'
        # for sub_dir in os.listdir(dir):
        #     if os.path.isdir(dir):
        #         for file_path in os.listdir(dir + sub_dir + '/'):
        #             if os.path.isfile(dir + sub_dir + '/' + file_path):
        #                 name_files_map[sub_dir].append(file_path)

        [name_files_map[sub_dir].append(dir + sub_dir + '/' + file_path) for sub_dir in os.listdir(dir) if
         os.path.isdir(dir) for file_path
         in os.listdir(dir + sub_dir + '/') if os.path.isfile(dir + sub_dir + '/' + file_path)]
        return name_files_map

    @staticmethod
    def get_file_path_split(filename):
        (filepath, tempfilename) = os.path.split(filename);
        (shotname, extension) = os.path.splitext(tempfilename);
        return filepath, shotname, extension


class FaceDetectionFrFoc(object):
    @staticmethod
    def detection(frame):
        boxes = face_recognition.face_locations(frame)
        boxes = [Util.fl_to_cv_box(box) for box in boxes]
        return boxes


class FaceDetectionDlibFro(object):
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

    @staticmethod
    def dlib_box_to_cv(rectangle):
        x, y, w, h = rectangle.left(), rectangle.top(), rectangle.right() - rectangle.left(), rectangle.bottom() - rectangle.top()
        return x, y, w, h

    def detection(self, frame):
        rectangles = self.face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
        if not rectangles:
            return list()
        else:
            return [FaceDetectionDlibFro.dlib_box_to_cv(rectangle) for rectangle in rectangles]
        return boxes


class FaceDetectionCvCas(object):
    def __init__(self, face_add):
        self.face_detector = cv2.CascadeClassifier(face_add)

    def detection(self, frame):
        boxes = self.face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.15, 5)
        return boxes


class FaceEncodingFrFe(object):
    @staticmethod
    def encoding_frame_box(frame_box):
        if frame_box.frame is not None and len(frame_box.frame) > 0 and frame_box.box is not None and len(
                frame_box.box) > 0:
            return FaceEncodingFrFe.encoding(frame_box.frame, frame_box.box)
        return FaceEncodingFrFe.encoding_img(frame_box.img)

    @staticmethod
    def encoding(frame, box):
        fl_box = Util.cv_to_fl_box(box)
        face_encodings = face_recognition.face_encodings(frame, [fl_box])
        return np.array(face_encodings[0]) if face_encodings else None

    @staticmethod
    def encoding_img(img):
        face_encodings = face_recognition.face_encodings(img)
        return np.array(face_encodings[0]) if face_encodings else None


class FaceEncodingDlibReg(object):
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        self.face_encoding = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")

    @staticmethod
    def cv_box_to_dlib(box):
        x, y, w, h = box
        rectangle = dlib.rectangle(x, y, x + w, y + h)
        return rectangle

    def encoding_frame_box(self, frame_box):
        if frame_box.frame is not None and len(frame_box.frame) > 0 and frame_box.box is not None and len(
                frame_box.box) > 0:
            return self.encoding(frame_box.frame, frame_box.box)
        return self.encoding_img(frame_box.img)

    def encoding(self, frame, box):
        rectangle = FaceEncodingDlibReg.cv_box_to_dlib(box)
        shape = self.shape(frame, rectangle)
        face_descriptor = self.face_encoding.compute_face_descriptor(frame, shape)
        return np.array(face_descriptor)

    def encoding_img(self, img):
        boxes = self.face_detector(img, 1)
        if not boxes:
            return None
        box = boxes[0]
        shape = self.shape(img, box)
        face_descriptor = self.face_encoding.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)


class FaceFactory(object):
    @staticmethod
    def get_encoding(name):
        if name == 'FR_FE':
            return FaceEncodingFrFe()
        elif name == 'DLIB_REG':
            return FaceEncodingDlibReg()
        else:
            return None

    @staticmethod
    def get_detection(name):
        if name == "FR_FL":
            return FaceDetectionFrFoc()
        elif name == "CV_CAS":
            return FaceDetectionCvCas(face_add="model/haarcascade_frontalface_default.xml")
        elif name == "DLIB_FRO":
            return FaceDetectionDlibFro()
        else:
            return None


class FrameBox(object):
    def __init__(self, frame=None, box=None):
        assert (frame is not None and box is not None), 'bad param'
        self.frame = frame
        self.box = list(map(int, box))

    @property
    def name(self):
        return '_'.join(map(str, self.box)) + '.png'

    @staticmethod
    def parse_file(file_name):
        frame = cv2.imread(file_name)
        file_path, file_name, file_ext = Util.get_file_path_split(file_name)
        box = file_name.split('_')
        return frame, box


class LimitList(object):
    def __init__(self, maxsize=10):
        self.maxsize = maxsize
        self.__list = list()

    def append(self, item):
        if len(self.__list) >= self.maxsize:
            return False
        else:
            self.__list.append(item)
            return True

    def pop(self):
        if len(self.__list) == 0:
            return None
        return self.__list.pop()

    def __iter__(self):
        return self.__list.__iter__()


class Person(object):
    __unknow_max_id = 0
    face_encoding = None
    img_dir = ''

    def __init__(self, person_name, img_files, ipc_name, is_new=False, new_face_frame_max=10):
        self.ipc_name = ipc_name
        self.is_new = is_new
        self.person_name = person_name
        assert np.all([os.path.exists(img) for img in img_files]), 'img file is error'
        # assert np.all([img.find('unknow') == -1 for img in self.imgs]), "img file can't contain unknow"
        frames_box = [FrameBox(*FrameBox.parse_file(img_file)) for img_file in img_files]
        encodings = [Person.face_encoding.encoding_frame_box(frame_box) for frame_box in frames_box]
        self.__encodings = encodings

        if is_new:
            self.frames_box_limit = LimitList(new_face_frame_max)
            self.__encodings = []

        logger.info('new persion %s' % (str([np.sum(encoding) for encoding in self.encodings_valid()])))

    def new_frame_box(self, frame_box):
        if self.is_new:
            self.frames_box_limit.append(frame_box) and self.__encodings.append(
                Person.face_encoding.encoding_frame_box(frame_box))

    def encodings_valid(self):
        return [x for x in self.__encodings if x is not None and len(x)]

    @staticmethod
    def new_unknow_person(ipc_name):
        return Person(Person.get_unknow_name(), list(), ipc_name=ipc_name, is_new=True)

    @staticmethod
    def get_unknow_name():
        Person.__unknow_max_id += 1
        return "unknow{0:03d}".format(Person.__unknow_max_id)

    def save(self):
        if self.is_new:
            dir_path = Person.img_dir + self.ipc_name + '/' + self.person_name + '/'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            for frame_box in self.frames_box_limit:
                cv2.imwrite(dir_path + frame_box.name, frame_box.frame)
            self.is_new = False

    @staticmethod
    def get_camera_person_files(cameras_dir):
        assert os.path.exists(cameras_dir) and os.path.isdir(cameras_dir), 'dir is illegal'
        camera_person_dict = defaultdict(dict)
        for camera_dir in os.listdir(cameras_dir):
            if os.path.isdir(cameras_dir + '/' + camera_dir):
                camera_person_dict[camera_dir] = Util.get_dirs_files(cameras_dir + '/' + camera_dir + '/')
        return camera_person_dict


# img_items,[(file_name,img_frame)]
# todo


class Track(object):
    __id = 0

    def __init__(self, ipc_name, tracker, frame, box, encoding, persons, event_call_back, history=5):
        self.ipc_name = ipc_name
        self.__id = Track.__id = Track.__id + 1
        self.tracker = tracker
        self.img = frame[(box[1]):(box[1] + box[3]), (box[0]):(box[0] + box[2])]
        self.frame = frame
        self.encoding = encoding
        self.__history = [False] * history
        self.__history_iter = itertools.cycle(range(history))
        self.match_person = None
        # 暂不用self.__history_have=bool
        self.__history[next(self.__history_iter)] = True

        self.__init_tracker(frame, box)

        self.find_person(persons)
        self.event_call_back = event_call_back
        self.event_call_back(0, self.ipc_name, self.__id, self.img, box, self.match_person.person_name)

    def __init_tracker(self, frame, box):
        self.tracker.init(frame, tuple(box))

    def update_img(self, frame, box, encoding):
        self.img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        self.match_person.new_frame_box(FrameBox(frame, box))
        self.encoding = encoding
        self.frame = frame
        iter_num = next(self.__history_iter)
        if self.alive() and self.__history[iter_num] == 1 and sum(self.__history) == 1:
            self.event_call_back(1, self.ipc_name, self.id, self.frame, box, self.match_person.person_name)
            self.__history[iter_num] = False

    def update(self, frame):
        return self.tracker.update(frame)

    def find_person(self, persons, tolerance=0.6):
        if self.encoding is None or len(self.encoding) == 0:
            return
        person_dist = [min(face_recognition.face_distance(person.encodings_valid(), self.encoding), default=1.0) for
                       person in
                       persons]
        logger.info('find_person self.encodings %s ' % (str(np.sum(self.encoding))))
        if min(person_dist, default=1.0) < tolerance:
            min_dist_index = np.argmin(person_dist)
            self.match_person = persons[min_dist_index]
        else:
            self.match_person = Person.new_unknow_person(ipc_name=self.ipc_name)
            persons.append(self.match_person)

    def alive(self):
        return sum(self.__history) > 0

    @property
    def id(self):
        return self.__id


class DetectionTrack(object):
    def __init__(self, face_detector, face_encoding, detecton_freq, camera_persons):
        self.__last_frame = None
        self.detecton_freq = detecton_freq
        self.detecton_freq_iter_map = dict()
        self.is_start_map = dict()
        self.is_realtime = False  # only false ,code only support
        self.tracks_map = dict()
        self.cap_thread_map = dict()
        self.det_thread = None
        self.face_detector = face_detector
        self.face_encoding = face_encoding
        self.frame_queue_map = dict()
        self.cv_map = dict()
        self.video_write_map = dict()
        self.ipc_infos = None
        self.persons_map = camera_persons
        self.event_df = pd.DataFrame(
            columns=['ipc_name', 'event_name', 'datetime', 'track_id', 'img_file', 'box', 'person_name'])

    def event_call_back(self, type, ipc_name, track_id, frame=None, box=None, person_name=None):
        # type=0 enter, 1 out
        cv2.imwrite('%s/new_face_%s.png' % (VIDEO_IMG, track_id), frame)
        self.event_df.loc[self.event_df.shape[0]] = [ipc_name, type,
                                                     datetime.now().strftime('%Y%m%d%H%M%S'), track_id,
                                                     '%s/new_face_%s.png' % (VIDEO_IMG, track_id), box,
                                                     person_name]

    def start_all(self, ipc_infos):
        # ipc_infos:list.map.key=ipc_url/ipc_name,list.map.value='xx/xx.mp4'/test01
        self.ipc_infos = ipc_infos
        for ipc_info in ipc_infos:
            self.is_start_map[ipc_info['name']] = True
            self.start_one(ipc_info['name'], ipc_info['path'])
        self.det_thread = threading.Thread(target=self.__start_detection_trace)
        self.det_thread.start()

    def start_one(self, ipc_name, ipc_path):
        self.detecton_freq_iter_map[ipc_name] = itertools.cycle(range(self.detecton_freq))
        self.tracks_map[ipc_name] = list()
        self.frame_queue_map[ipc_name] = Queue()
        self.cv_map[ipc_name] = cv2.VideoCapture(ipc_path)

        videoCapture = cv2.VideoCapture(ipc_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (
            int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        out_file_path = "%s_output.avi" % ipc_path
        videoWriter = cv2.VideoWriter(
            out_file_path,
            cv2.VideoWriter_fourcc('M', 'P', '4', '2'),  # 编码器
            fps,
            size
        )

        self.video_write_map[ipc_name] = videoWriter
        cap_thread = threading.Thread(target=self.__start_capture, args=(ipc_name,))
        self.cap_thread_map[ipc_name] = cap_thread
        cap_thread.start()

    def stop_one(self, ipc_name):
        self.is_start_map[ipc_name] = False
        self.cap_thread_map[ipc_name].join()
        del self.cap_thread_map[ipc_name]
        self.video_write_map[ipc_name].release()
        del self.video_write_map[ipc_name]

        del self.detecton_freq_iter_map[ipc_name]
        [tracker.match_person.save() for tracker in self.tracks_map[ipc_name]]
        del self.tracks_map[ipc_name]
        del self.frame_queue_map[ipc_name]
        self.cv_map[ipc_name].release()
        del self.cv_map[ipc_name]

    def __start_capture(self, ipc_name):
        cv_cap = self.cv_map[ipc_name]
        frame_queue = self.frame_queue_map.get(ipc_name)
        while self.is_start_map[ipc_name]:
            ret, frame = cv_cap.read()
            if ret:
                if self.is_realtime:
                    self.__last_frame = frame
                else:
                    frame_queue.put(frame)
            else:
                if self.is_realtime:
                    self.is_start_map[ipc_name] = False  # stop threading:_start_capture,_start_detection_trace
                    break
                    # todo enhance,some tail frame wo't be run by threading _start_detection_trace
                else:
                    frame_queue.put(None)
                    break

    def __get_last_frame(self, ipc_name):
        # if self.is_realtime:
        #     pass
        # else:
        #     pass
        ret = self.frame_queue_map[ipc_name].get()
        return ret

    def __start_detection_trace(self):
        ipc_name_iter = itertools.cycle([ipc_info['name'] for ipc_info in self.ipc_infos])
        while np.any(list(self.is_start_map.values())):
            ipc_name = next(ipc_name_iter)
            while self.is_start_map[ipc_name]:
                last_frame = self.__get_last_frame(ipc_name)
                if last_frame is None:
                    if self.is_realtime:
                        continue
                    else:
                        self.stop_one(ipc_name)
                        continue
                if next(self.detecton_freq_iter_map[ipc_name]) == 0:
                    boxes = self.__face_dec(last_frame, ipc_name)
                else:
                    boxes = self.__face_track(last_frame, ipc_name)
                [Util.draw_boxes(last_frame, list(box)) for box in boxes]
                self.video_write_map[ipc_name].write(last_frame)
                # cv2.imshow('xx', last_frame)
                # cv2.waitKey(1)
        self.after_all_stop()

    def after_all_stop(self):
        self.event_df['img_file'] = self.event_df['img_file'].apply(lambda x: "<img src='%s'>" % x)
        event_name_map = {0: '进', 1: '出'}
        self.event_df['event_name'] = self.event_df['event_name'].map(event_name_map)
        self.event_df.to_html('event.html')

    def __face_dec(self, frame, ipc_name):
        boxes = self.face_detector.detection(frame)
        self.__face_upgrade_track(frame, boxes, ipc_name)
        return boxes

    def __face_upgrade_track(self, frame, boxes, ipc_name):
        tracks_map = {track.id: track for track in self.tracks_map[ipc_name]}
        track_ids, track_encodings = list(map(lambda x: x.id, self.tracks_map[ipc_name])), list(
            map(lambda x: x.encoding, self.tracks_map[ipc_name]))
        boxes_imgs_encoding = list()
        if boxes is not None and len(boxes):
            boxes_imgs_encoding = [self.face_encoding.encoding(frame, box) for box in boxes]
            boxes_encoding_filter = [boxes_img_encoding is not None and len(boxes_img_encoding) > 0 for
                                     boxes_img_encoding in boxes_imgs_encoding]
            boxes = np.array(boxes)[boxes_encoding_filter]
            boxes_imgs_encoding = np.array(boxes_imgs_encoding)[boxes_encoding_filter]
            # logger.info('###', np.array(boxes_imgs_encoding).shape)
            # boxes_imgs_encodings = [
            #     face_recognition.face_encodings(frame[(box[1]):(box[1] + box[3]), (box[0]):(box[0] + box[2])]) for box in boxes]
            # boxes_imgs_encoding = [boxes_imgs_encoding[0] for boxes_imgs_encoding in boxes_imgs_encodings if
            #                        boxes_imgs_encoding[:1]]
            # logger.info('####', np.array(boxes_imgs_encoding).shape)
            [cv2.imwrite(str(np.sum(boxes_encoding)) + '.png', Util.cut_frame_box(frame, box))
             for boxes_encoding, box in zip(boxes_imgs_encoding, boxes)]

        box_track_ids = [
            np.array(track_ids, int)[face_recognition.compare_faces(track_encodings, unknown_face_encoding)] for
            unknown_face_encoding in boxes_imgs_encoding]

        new_trackers = []
        for box_track_id, boxes_img, boxes_img_encoding in zip(box_track_ids, boxes, boxes_imgs_encoding):
            if len(box_track_id) > 0:
                tracks_map[box_track_id[0]].update_img(frame, boxes_img, boxes_img_encoding)
            else:
                new_trackers.append(Track(ipc_name, cv2.TrackerKCF_create(), frame, boxes_img, boxes_img_encoding,
                                          self.persons_map[ipc_name], event_call_back=self.event_call_back))

        # keep alive tracks only
        [tracker.match_person.save() for tracker in self.tracks_map[ipc_name] if not tracker.alive()]
        self.tracks_map[ipc_name] = [tracker for tracker in self.tracks_map[ipc_name] if tracker.alive()]
        self.tracks_map[ipc_name].extend(new_trackers)

    def __face_track(self, frame, ipc_name):
        boxes = [list(map(int, track.update(frame)[1])) for track in self.tracks_map[ipc_name]]
        return boxes

    # todo 01 use diff change to decation detection ,02,use yield change param
    # ref Human-detection-and-Tracking
    @staticmethod
    def background_subtraction(previous_frame, frame_resized_grayscale, min_area):
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


camere_persons_files = Person.get_camera_person_files(PERSON_IMG_DIR)
if __name__ == '__main__':
    ipc_infos = [{'name': 'test1', 'path': 'video/1.mp4'}]
    face_encoding = FaceFactory.get_encoding("DLIB_REG")
    # persons = person_df.apply(lambda x: Person(face_encoding, x['imgs'].split(' ')), axis=1)
    Person.face_encoding = face_encoding
    Person.img_dir = PERSON_IMG_DIR

    camera_persons = defaultdict(list)
    [camera_persons[camera_name].append(Person(person_name, person_file, camera_name)) for camera_name, persons_map in
     camere_persons_files for person_name, person_files in persons_map.items() for person_file in person_files]

    # persons_map = {'test1': persons, 'test11': persons}
    face_detector = FaceFactory.get_detection('CV_CAS')
    detection_track = DetectionTrack(face_detector, face_encoding, detecton_freq=20, camera_persons=camera_persons)
    detection_track.start_all(ipc_infos)
