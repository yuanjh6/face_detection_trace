import copy
import itertools
import threading
from collections import deque
from datetime import datetime
from queue import Queue

import cv2
import face_recognition
import numpy as np
import logging

import copy
import itertools
import threading
import time
from collections import deque
from contextlib import suppress
from queue import Queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
VIDEO_IMG = 'video_img'


class DetectionTrackQueue(object):
    def __init__(self, min_hold_deteciton=1):
        self.wait_is_detection = None
        self.wait_boxes = None
        self.min_hold_deteciton = min_hold_deteciton
        self.hold_deteciton = 0
        self.deque = deque()
        self.track_queue = Queue(maxsize=20)
        self.lock01 = threading.Lock()
        self.is_end = False

    # thread 01
    # this modify left side of self.deque
    def put_frame(self, frame):
        # multi kaolv xia
        with self.lock01:
            self.deque.append((frame, [], []))  # frame,is_detection,boxes
        is_blow_hold_deteciton = self.hold_deteciton < self.min_hold_deteciton
        if is_blow_hold_deteciton:
            return

        left_frame, is_detection, left_boxes = self.deque.popleft()

        is_min_hold_deteciton = self.hold_deteciton == self.min_hold_deteciton
        is_detection_waiting = left_boxes is self.wait_boxes
        # print('left frame:%s is_min_hold_deteciton:%s(%s,%s) is_detectioned:%s is_detection_waiting:%s' % (
        #     left_frame, is_min_hold_deteciton, self.hold_deteciton, self.min_hold_deteciton, is_detectioned,
        #     is_detection_waiting))
        if is_min_hold_deteciton and (is_detection or is_detection_waiting):
            self.deque.appendleft((left_frame, is_detection, left_boxes))
        elif is_detection:
            self.hold_deteciton -= 1
            self.track_queue.put((left_frame, is_detection, left_boxes))
        else:
            self.track_queue.put((left_frame, is_detection, left_boxes))

    # thread 02
    # this modify right side of self.deque (low probability conflict with put_frame,so ignore)
    def get_detection_frame(self):
        # use defer like
        right_frame = None
        try:
            with self.lock01:
                right_frame, is_detection, right_boxes = self.deque.pop()
                self.deque.append((right_frame, is_detection, right_boxes))
        except:
            return True, None

        if right_frame is None:
            return False, None

        is_detection_waiting = self.wait_boxes is not None
        if is_detection or is_detection_waiting:
            return True, None

        self.wait_is_detection, self.wait_boxes = is_detection, right_boxes
        self.hold_deteciton += 1
        return True, copy.deepcopy(right_frame)

    # thread 02
    # no use lock,make sure this never run with get_detection_frame at the same time(wait_boxes synchro)
    def put_detection_boxes(self, boxes):
        if self.wait_boxes is None:
            raise Exception('no wait_boxes point')
        self.wait_is_detection.append(True)
        self.wait_boxes.extend(boxes)
        self.wait_boxes = None
        self.wait_is_detection = None

    # thread 03
    def get_tracker_frame(self):
        return self.track_queue.get()

    def end(self):
        self.is_end = True
        while len(self.deque):
            self.track_queue.put(self.deque.popleft())


#
#
# class TestDemo(object):
#     def __init__(self):
#         self.detection_queue = DetectionTrackQueue()
#
#     def start(self):
#         put_thread = threading.Thread(target=self.__put_frame)
#         detection_thread = threading.Thread(target=self.__get_decetion)
#         track_thread = threading.Thread(target=self.__get_track)
#
#         put_thread.start()
#         detection_thread.start()
#         track_thread.start()
#
#     def __put_frame(self):
#         itert = itertools.cycle(range(10000))
#         while True:
#             frame = next(itert)
#             self.detection_queue.put_frame(frame)
#             print('put frame:%s' % frame)
#             time.sleep(0.05)
#
#     def __get_decetion(self):
#         while True:
#             frame = self.detection_queue.get_detection_frame()
#             print('get detection frame:%s' % frame)
#             if frame is None:
#                 time.sleep(0.1)
#                 continue
#             time.sleep(0.50)
#             self.detection_queue.put_detection_boxes([1.0, 2.0, 3.0, 4.0])
#             print('put_detection_boxes frame:%s' % frame)
#
#     def __get_track(self):
#         while True:
#             frame, boxes = self.detection_queue.get_tracker()
#             print('get track frame:%s %s' % (frame, boxes))
#             time.sleep(0.01)
#
# #
# # testDemo = TestDemo()
# # testDemo.start()


class Track(object):
    __id = 0

    def __init__(self, tracker, frame, box, encoding, history=5):
        self.__id = Track.__id = Track.__id + 1
        self.tracker = tracker
        self.img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        self.frame = frame
        self.encoding = encoding
        self.__history = [False] * history
        self.__history_iter = itertools.cycle(range(history))
        # 暂不用self.__history_have=bool

        self.tracker.init(frame, tuple(box))
        self.__history[next(self.__history_iter)] = True
        Track.new_face_callback(self.__id, self.frame, box)

    def update_img(self, frame, box, encoding):
        self.img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        self.encoding = encoding
        self.frame = frame
        iter_num = next(self.__history_iter)
        if self.alive() and self.__history[iter_num] == 1 and sum(self.__history) == 1:
            Track.lost_face_callback(self.id, self.frame, box)
            self.__history[iter_num] = False

    def update(self, frame):
        return self.tracker.update(frame)

    def alive(self):
        return sum(self.__history) > 0

    @property
    def id(self):
        return self.__id

    @staticmethod
    def new_face_callback(track_id, frame=None, box=None):
        DetectionTrack.draw_boxes(frame, box)
        cv2.imwrite('%s/new_face_%s.png' % (VIDEO_IMG, track_id), frame)
        logger.info('new face datetime:%s track_id: %s img:%s box:%s' % (
            datetime.now().strftime('%Y%m%d%H%M%S'), track_id, '%s/new_face_%s.png' % (VIDEO_IMG, track_id), box))

    @staticmethod
    def lost_face_callback(track_id, frame=None, box=None):
        DetectionTrack.draw_boxes(frame, box)
        cv2.imwrite('%s/lost_face_%s.png' % (VIDEO_IMG, track_id), frame)
        logger.info('lost face datetime:%s track_id: %s img:%s box:%s' % (
            datetime.now().strftime('%Y%m%d%H%M%S'), track_id, '%s/lost_face_%s.png' % (VIDEO_IMG, track_id), box))


class DetectionTrack(object):
    def __init__(self, face_detector, detecton_freq):
        self.__last_frame = None
        self.detecton_freq_iter = itertools.cycle(range(detecton_freq))
        self.is_start = False
        self.is_realtime = False
        self.tracks = list()
        self.cap_thread = None
        self.det_thread = None
        self.face_detector = face_detector
        self.dt_frame_queue = DetectionTrackQueue(min_hold_deteciton=1)
        self.ipc_path = None

    def start(self, ipc_path=None):
        self.ipc_path = ipc_path
        if not self.is_start:
            self.is_start = True
            self.cap_thread = threading.Thread(target=self._start_capture, args=(ipc_path,))
            self.det_thread = threading.Thread(target=self._start_detection)
            self.trk_thread = threading.Thread(target=self._start_trace)
            self.cap_thread.start()
            self.det_thread.start()
            self.trk_thread.start()

    def _start_capture(self, ipc_path=None):
        cv_cap = cv2.VideoCapture(ipc_path)
        while self.is_start:
            ret, frame = cv_cap.read()
            if ret:
                if self.is_realtime:
                    self.__last_frame = frame
                else:
                    self.dt_frame_queue.put_frame(frame)
            else:
                if self.is_realtime:
                    self.is_start = False  # stop threading:_start_capture,_start_detection_trace
                    # todo enhance,some tail frame wo't be run by threading _start_detection_trace
                else:
                    self.dt_frame_queue.put_frame(None)
                break

    def _start_detection(self):
        while self.is_start:
            ret, last_frame = self.dt_frame_queue.get_detection_frame()
            if not ret:
                self.dt_frame_queue.end()
                break
            if last_frame is None:
                continue

            boxes = self.__face_dec(last_frame)
            self.dt_frame_queue.put_detection_boxes(boxes)

    def _start_trace(self):
        while self.is_start:
            frame, is_detection, boxes = self.dt_frame_queue.get_tracker_frame()
            if (boxes is None) or (frame is None):
                break
            is_detection_result = bool(is_detection)
            if not is_detection_result:
                boxes = self.__face_track(frame)
            else:
                tracks_map = {track.id: track for track in self.tracks}
                track_ids, track_encodings = list(map(lambda x: x.id, self.tracks)), list(
                    map(lambda x: x.encoding, self.tracks))
                boxes_imgs_encoding = list()
                if boxes is not None and list(boxes):
                    boxes_imgs_encoding = face_recognition.face_encodings(frame, known_face_locations=boxes)
                box_track_ids = [
                    np.array(track_ids, int)[face_recognition.compare_faces(track_encodings, unknown_face_encoding)] for
                    unknown_face_encoding in boxes_imgs_encoding]

                new_trackers = [
                    tracks_map[box_track_id[0]].update_img(frame, boxes_img, boxes_img_encoding)
                    if len(box_track_id) > 0
                    else Track(cv2.TrackerBoosting_create(), frame, boxes_img, boxes_img_encoding) for
                    box_track_id, boxes_img, boxes_img_encoding in
                    zip(box_track_ids, boxes, boxes_imgs_encoding)]
                self.tracks = list(filter(lambda x: x.alive(), self.tracks))  # keep alive tracks only
                self.tracks.extend(list(filter(lambda x: x is not None, new_trackers)))

            [DetectionTrack.draw_boxes(frame, list(box)) for box in boxes]

            cv2.imshow('xx', frame)
            cv2.waitKey(1)

    def stop(self):
        self.is_start = False

    def __face_dec(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.face_detector.detectMultiScale(gray, 1.15, 5)
        return boxes

    def __face_track(self, frame):
        boxes = [list(map(int, track.update(frame)[1])) for track in self.tracks]
        return boxes

    @staticmethod
    def draw_boxes(frame, box):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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


if __name__ == '__main__':
    ipc_path = 'video/1.mp4'  # sys.argv[1]
    faceadd = "model/haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(faceadd)
    # faces = face_detector.detectMultiScale(gray, 1.15, 5)
    detection_track = DetectionTrack(face_detector, detecton_freq=20)
    detection_track.start(ipc_path)
