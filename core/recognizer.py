# -*- coding: utf-8 -*-
# @Time    : 2018/05/08
# @Author  : WangRong

import json
import os
import requests
import numpy as np
# from core.youtuUtil import getFaceDataFromYoutu

import face_recognition

use_youtu = False
if use_youtu:
    FACE_POINTS = list(range(0, 89))
    JAW_POINTS = list(range(0, 20))
    LEFT_EYE_POINTS = list(range(21, 28))
    RIGHT_EYE_POINTS = list(range(29, 36))
    LEFT_BROW_POINTS = list(range(37, 44))
    RIGHT_BROW_POINTS = list(range(45, 52))
    MOUTH_POINTS = list(range(53, 74))
    NOSE_POINTS = list(range(75, 87))


    LEFT_FACE = list(range(21, 28)) + list(range(37, 44))
    RIGHT_FACE = list(range(29, 36)) + list(range(45, 52))

    JAW_END = 20
    FACE_START = 0
    FACE_END = 89

    OVERLAY_POINTS = [
        LEFT_FACE,
        RIGHT_FACE,
        JAW_POINTS,
    ]
else:
    FACE_POINTS = list(range(0, 72))
    JAW_POINTS = list(range(0, 17))
    LEFT_EYE_POINTS = list(range(36, 42))
    RIGHT_EYE_POINTS = list(range(42, 48))
    LEFT_BROW_POINTS = list(range(17, 22))
    RIGHT_BROW_POINTS = list(range(22, 27))
    MOUTH_POINTS = list(range(48, 72))
    NOSE_POINTS = list(range(27, 36))


    LEFT_FACE = list(range(36, 42)) + list(range(17, 22))
    RIGHT_FACE = list(range(42, 48)) + list(range(22, 27))

    JAW_END = 17
    FACE_START = 0
    FACE_END = 72

    OVERLAY_POINTS = [
        LEFT_FACE,
        RIGHT_FACE,
        JAW_POINTS,
    ]


def face_points(image):
    points = []
    points = landmarks_by_face__(image)
    faces = points
    if not faces:
        err = 404
        return [], [], [], err
    else:
        err = 0
    if(use_youtu):
        matrix_list = np.matrix(matrix_marks_youtu(str(points)))
    else:
        tmp = []
        for i in points[:-1]:
            tmp.append([i[0],i[1]])
        matrix_list = np.matrix(tmp)



    point_list = []
    # print(matrix_list)
    for p in matrix_list.tolist():
        # print(p)
        point_list.append((int(p[0]), int(p[1])))

    return matrix_list, point_list, faces, err


def landmarks_by_face__(image):
    if use_youtu:
        r = getFaceDataFromYoutu(image)
        if r["errorcode"] == 0:
            return r["face"][0]
    else:
        if isinstance(image,str):
            image = face_recognition.load_image_file(image)
        if not np.issubdtype(image.dtype,np.dtype('float')):
            image = image.astype(np.uint8)
        import time
        start = time.time()
        face_list = face_recognition.face_landmarks(image)
        if(len(face_list) == 0):
            return;
        res  = face_recognition.face_landmarks(image)[0]
        end = time.time() - start
        print("face Time:"+str(end))
        (top,right,bottom,left) = face_recognition.face_locations(image)[0]
        xywh = (left,top,right - left,bottom - top)
        points = []
        points+=res["chin"]
        points+=res["left_eyebrow"]
        points+=res["right_eyebrow"]
        points+=res["nose_bridge"]
        points+=res["nose_tip"]
        points+=res["left_eye"]
        points+=res["right_eye"]
        points+=res["top_lip"]
        points+=res["bottom_lip"]
        points.append(xywh)
        return points

def matrix_rectangle(left, top, width, height):
    pointer = [
        (left, top),#x0 y0
        (left + width / 2, top),
        (left + width - 1, top),
        (left + width - 1, top + height / 2),
        (left, top + height / 2),
        (left, top + height - 1),
        (left + width / 2, top + height - 1),
        (left + width - 1, top + height - 1)
    ]

    return pointer

def matrix_marks_youtu(ress):
    # print(ress)
    rTemp = ress.replace("u'", "\"")
    rTemp = rTemp.replace("'", "\"")
    res = json.loads(rTemp)
    pointer = [
        [int(res['face_shape']['face_profile'][0]['x']), int(res['face_shape']['face_profile'][0]['y'])],
        [int(res['face_shape']['face_profile'][1]['x']), int(res['face_shape']['face_profile'][1]['y'])],
        [int(res['face_shape']['face_profile'][2]['x']), int(res['face_shape']['face_profile'][2]['y'])],
        [int(res['face_shape']['face_profile'][3]['x']), int(res['face_shape']['face_profile'][3]['y'])],
        [res['face_shape']['face_profile'][4]['x'], res['face_shape']['face_profile'][4]['y']],
        [res['face_shape']['face_profile'][5]['x'], res['face_shape']['face_profile'][5]['y']],
        [res['face_shape']['face_profile'][6]['x'], res['face_shape']['face_profile'][6]['y']],
        [res['face_shape']['face_profile'][7]['x'], res['face_shape']['face_profile'][7]['y']],
        [res['face_shape']['face_profile'][8]['x'], res['face_shape']['face_profile'][8]['y']],
        [res['face_shape']['face_profile'][9]['x'], res['face_shape']['face_profile'][9]['y']],
        [res['face_shape']['face_profile'][10]['x'], res['face_shape']['face_profile'][10]['y']],
        [res['face_shape']['face_profile'][11]['x'], res['face_shape']['face_profile'][11]['y']],
        [res['face_shape']['face_profile'][12]['x'], res['face_shape']['face_profile'][12]['y']],
        [res['face_shape']['face_profile'][13]['x'], res['face_shape']['face_profile'][13]['y']],
        [res['face_shape']['face_profile'][14]['x'], res['face_shape']['face_profile'][14]['y']],
        [res['face_shape']['face_profile'][15]['x'], res['face_shape']['face_profile'][15]['y']],
        [res['face_shape']['face_profile'][16]['x'], res['face_shape']['face_profile'][16]['y']],
        [res['face_shape']['face_profile'][17]['x'], res['face_shape']['face_profile'][17]['y']],
        [res['face_shape']['face_profile'][18]['x'], res['face_shape']['face_profile'][18]['y']],
        [res['face_shape']['face_profile'][19]['x'], res['face_shape']['face_profile'][19]['y']],
        [res['face_shape']['face_profile'][20]['x'], res['face_shape']['face_profile'][20]['y']],

        [res['face_shape']['left_eye'][0]['x'], res['face_shape']['left_eye'][0]['y']],
        [res['face_shape']['left_eye'][1]['x'], res['face_shape']['left_eye'][1]['y']],
        [res['face_shape']['left_eye'][2]['x'], res['face_shape']['left_eye'][2]['y']],
        [res['face_shape']['left_eye'][3]['x'], res['face_shape']['left_eye'][3]['y']],
        [res['face_shape']['left_eye'][4]['x'], res['face_shape']['left_eye'][4]['y']],
        [res['face_shape']['left_eye'][5]['x'], res['face_shape']['left_eye'][5]['y']],
        [res['face_shape']['left_eye'][6]['x'], res['face_shape']['left_eye'][6]['y']],
        [res['face_shape']['left_eye'][7]['x'], res['face_shape']['left_eye'][7]['y']],

        [res['face_shape']['right_eye'][0]['x'], res['face_shape']['right_eye'][0]['y']],
        [res['face_shape']['right_eye'][1]['x'], res['face_shape']['right_eye'][1]['y']],
        [res['face_shape']['right_eye'][2]['x'], res['face_shape']['right_eye'][2]['y']],
        [res['face_shape']['right_eye'][3]['x'], res['face_shape']['right_eye'][3]['y']],
        [res['face_shape']['right_eye'][4]['x'], res['face_shape']['right_eye'][4]['y']],
        [res['face_shape']['right_eye'][5]['x'], res['face_shape']['right_eye'][5]['y']],
        [res['face_shape']['right_eye'][6]['x'], res['face_shape']['right_eye'][6]['y']],
        [res['face_shape']['right_eye'][7]['x'], res['face_shape']['right_eye'][7]['y']],

        [res['face_shape']['left_eyebrow'][0]['x'], res['face_shape']['left_eyebrow'][0]['y']],
        [res['face_shape']['left_eyebrow'][1]['x'], res['face_shape']['left_eyebrow'][1]['y']],
        [res['face_shape']['left_eyebrow'][2]['x'], res['face_shape']['left_eyebrow'][2]['y']],
        [res['face_shape']['left_eyebrow'][3]['x'], res['face_shape']['left_eyebrow'][3]['y']],
        [res['face_shape']['left_eyebrow'][4]['x'], res['face_shape']['left_eyebrow'][4]['y']],
        [res['face_shape']['left_eyebrow'][5]['x'], res['face_shape']['left_eyebrow'][5]['y']],
        [res['face_shape']['left_eyebrow'][6]['x'], res['face_shape']['left_eyebrow'][6]['y']],
        [res['face_shape']['left_eyebrow'][7]['x'], res['face_shape']['left_eyebrow'][7]['y']],

        [res['face_shape']['right_eyebrow'][0]['x'], res['face_shape']['right_eyebrow'][0]['y']],
        [res['face_shape']['right_eyebrow'][1]['x'], res['face_shape']['right_eyebrow'][1]['y']],
        [res['face_shape']['right_eyebrow'][2]['x'], res['face_shape']['right_eyebrow'][2]['y']],
        [res['face_shape']['right_eyebrow'][3]['x'], res['face_shape']['right_eyebrow'][3]['y']],
        [res['face_shape']['right_eyebrow'][4]['x'], res['face_shape']['right_eyebrow'][4]['y']],
        [res['face_shape']['right_eyebrow'][5]['x'], res['face_shape']['right_eyebrow'][5]['y']],
        [res['face_shape']['right_eyebrow'][6]['x'], res['face_shape']['right_eyebrow'][6]['y']],
        [res['face_shape']['right_eyebrow'][7]['x'], res['face_shape']['right_eyebrow'][7]['y']],

        [res['face_shape']['mouth'][0]['x'], res['face_shape']['mouth'][0]['y']],
        [res['face_shape']['mouth'][1]['x'], res['face_shape']['mouth'][1]['y']],
        [res['face_shape']['mouth'][2]['x'], res['face_shape']['mouth'][2]['y']],
        [res['face_shape']['mouth'][3]['x'], res['face_shape']['mouth'][3]['y']],
        [res['face_shape']['mouth'][4]['x'], res['face_shape']['mouth'][4]['y']],
        [res['face_shape']['mouth'][5]['x'], res['face_shape']['mouth'][5]['y']],
        [res['face_shape']['mouth'][6]['x'], res['face_shape']['mouth'][6]['y']],
        [res['face_shape']['mouth'][7]['x'], res['face_shape']['mouth'][7]['y']],
        [res['face_shape']['mouth'][8]['x'], res['face_shape']['mouth'][8]['y']],
        [res['face_shape']['mouth'][9]['x'], res['face_shape']['mouth'][9]['y']],
        [res['face_shape']['mouth'][10]['x'], res['face_shape']['mouth'][10]['y']],
        [res['face_shape']['mouth'][11]['x'], res['face_shape']['mouth'][11]['y']],
        [res['face_shape']['mouth'][12]['x'], res['face_shape']['mouth'][12]['y']],
        [res['face_shape']['mouth'][13]['x'], res['face_shape']['mouth'][13]['y']],
        [res['face_shape']['mouth'][14]['x'], res['face_shape']['mouth'][14]['y']],
        [res['face_shape']['mouth'][15]['x'], res['face_shape']['mouth'][15]['y']],
        [res['face_shape']['mouth'][16]['x'], res['face_shape']['mouth'][16]['y']],
        [res['face_shape']['mouth'][17]['x'], res['face_shape']['mouth'][17]['y']],
        [res['face_shape']['mouth'][18]['x'], res['face_shape']['mouth'][18]['y']],
        [res['face_shape']['mouth'][19]['x'], res['face_shape']['mouth'][19]['y']],
        [res['face_shape']['mouth'][20]['x'], res['face_shape']['mouth'][20]['y']],
        [res['face_shape']['mouth'][21]['x'], res['face_shape']['mouth'][21]['y']],

        [res['face_shape']['nose'][0]['x'], res['face_shape']['nose'][0]['y']],
        [res['face_shape']['nose'][1]['x'], res['face_shape']['nose'][1]['y']],
        [res['face_shape']['nose'][2]['x'], res['face_shape']['nose'][2]['y']],
        [res['face_shape']['nose'][3]['x'], res['face_shape']['nose'][3]['y']],
        [res['face_shape']['nose'][4]['x'], res['face_shape']['nose'][4]['y']],
        [res['face_shape']['nose'][5]['x'], res['face_shape']['nose'][5]['y']],
        [res['face_shape']['nose'][6]['x'], res['face_shape']['nose'][6]['y']],
        [res['face_shape']['nose'][7]['x'], res['face_shape']['nose'][7]['y']],
        [res['face_shape']['nose'][8]['x'], res['face_shape']['nose'][8]['y']],
        [res['face_shape']['nose'][9]['x'], res['face_shape']['nose'][9]['y']],
        [res['face_shape']['nose'][10]['x'], res['face_shape']['nose'][10]['y']],
        [res['face_shape']['nose'][11]['x'], res['face_shape']['nose'][11]['y']],
        [res['face_shape']['nose'][12]['x'], res['face_shape']['nose'][12]['y']],

        [res['face_shape']['pupil'][0]['x'], res['face_shape']['pupil'][0]['y']],
        [res['face_shape']['pupil'][1]['x'], res['face_shape']['pupil'][1]['y']],
    ]

    return pointer

if __name__ == "__main__":
    path = "images/model.jpg"
    face_points(path)

