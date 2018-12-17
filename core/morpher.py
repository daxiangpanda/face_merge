# -*- coding: utf-8 -*-
# @Time    : 2018/05/18
# @Author  : Tony Wang
import datetime
import json

import cv2
import numpy as np
import time
import core
import os
from PIL import Image, ImageDraw

def transformation_points(src_img, src_points, dst_img, dst_points):
    src_points = src_points.astype(np.float64)
    dst_points = dst_points.astype(np.float64)
    # print(src_points.shape)
    # print(dst_points)
    c1 = np.mean(src_points, axis=0)
    c2 = np.mean(dst_points, axis=0)

    src_points -= c1
    dst_points -= c2

    s1 = np.std(src_points)
    s2 = np.std(dst_points)

    src_points /= s1
    dst_points /= s2

    u, s, vt = np.linalg.svd(src_points.T * dst_points)
    r = (u * vt).T
    m = np.vstack([np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)), np.matrix([0., 0., 1.])])

    output = cv2.warpAffine(dst_img, m[:2],
                            (src_img.shape[1], src_img.shape[0]),
                            borderMode=cv2.BORDER_TRANSPARENT,
                            flags=cv2.WARP_INVERSE_MAP)

    return output


def tran_matrix(src_img, src_points, dst_img, dst_points):
    h = cv2.findHomography(dst_points, src_points)
    output = cv2.warpAffine(dst_img, h[0][:2], (src_img.shape[1], src_img.shape[0]),
                            borderMode=cv2.BORDER_TRANSPARENT,
                            flags=cv2.WARP_INVERSE_MAP)
    return output


def correct_color(img1, img2, landmark):
    blur_amount = 0.4 * np.linalg.norm(
        np.mean(landmark[core.LEFT_EYE_POINTS], axis=0)
        - np.mean(landmark[core.RIGHT_EYE_POINTS], axis=0)
    )
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

    return img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)


def tran_src(src_img, src_points, dst_points, face_area=None):
    # print(1111111)
    print(src_img.shape)
    jaw = core.JAW_END

    dst_list = dst_points \
               + core.matrix_rectangle(face_area[0], face_area[1], face_area[2], face_area[3]) \
               + core.matrix_rectangle(0, 0, src_img.shape[1], src_img.shape[0])

    src_list = src_points \
               + core.matrix_rectangle(face_area[0], face_area[1], face_area[2], face_area[3]) \
               + core.matrix_rectangle(0, 0, src_img.shape[1], src_img.shape[0])

    jaw_points = []

    for i in range(0, jaw):
        # print(i)
        jaw_points.append(dst_list[i])
        jaw_points.append(src_list[i])

    warp_jaw = cv2.convexHull(np.array(jaw_points), returnPoints=False)
    warp_jaw = warp_jaw.tolist()

    for i in range(0, len(warp_jaw)):
        warp_jaw[i] = warp_jaw[i][0]

    warp_jaw.sort()

    if len(warp_jaw) <= jaw:
        dst_list = dst_list[jaw - len(warp_jaw):]
        src_list = src_list[jaw - len(warp_jaw):]
        for i in range(0, len(warp_jaw)):
            dst_list[i] = jaw_points[int(warp_jaw[i])]
            src_list[i] = jaw_points[int(warp_jaw[i])]
    else:
        for i in range(0, jaw):
            if len(warp_jaw) > jaw and warp_jaw[i] == 2 * i and warp_jaw[i + 1] == 2 * i + 1:
                warp_jaw.remove(2 * i)

            dst_list[i] = jaw_points[int(warp_jaw[i])]

    dt = core.measure_triangle(src_img, dst_list,src_points,dst_points)

    res_img = np.zeros(src_img.shape, dtype=src_img.dtype)

    for i in range(0, len(dt)):
        t_src = []
        t_dst = []

        for j in range(0, 3):
            t_src.append(src_list[dt[i][j]])
            t_dst.append(dst_list[dt[i][j]])
        if(checkLine(t_src) or checkLine(t_dst)):
            # print("not checked")
            continue
        else:
            core.affine_triangle(src_img, res_img, t_src, t_dst)



    return res_img


def merge_img(src_img, dst_img, dst_matrix, dst_points, k_size=None, mat_multiple=None):
    face_mask = np.zeros(src_img.shape, dtype=src_img.dtype)

    for group in core.OVERLAY_POINTS:
        cv2.fillConvexPoly(face_mask, cv2.convexHull(dst_matrix[group]), (255, 255, 255))

    r = cv2.boundingRect(np.float32([dst_points[:core.FACE_END]]))

    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    if mat_multiple:
        mat = cv2.getRotationMatrix2D(center, 0, mat_multiple)
        face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))

    if k_size:
        face_mask = cv2.blur(face_mask, k_size, center)

    return cv2.seamlessClone(np.uint8(dst_img), src_img, face_mask, center, cv2.NORMAL_CLONE)

def drawLine(src_img,points):
    src_img = src_img.astype(np.uint8)
    im = Image.fromarray(src_img)
    draw = ImageDraw.Draw(im)
    # for i in points:
    draw.line(points,width = 5,fill = (255, 0, 0))
    return im

def morph_img(src_img, src_points, dst_img, dst_points, alpha=0.5):

    morph_points = []

    src_img = src_img.astype(np.float32)
    dst_img = dst_img.astype(np.float32)

    res_img = np.zeros(src_img.shape, src_img.dtype)
    # for i in src_points:
    #     print(i)

    # 这一步的目的是调整脸型，将原图关键点和目标图关键点之间取中间点，根据alpha值来取
    for i in range(0, len(src_points)):
        x = (1 - alpha) * src_points[i][0] + alpha * dst_points[i][0]
        y = (1 - alpha) * src_points[i][1] + alpha * dst_points[i][1]
        morph_points.append((x, y))

    dt = core.measure_triangle(src_img, morph_points,src_points,dst_points)

    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        t = []
        for j in range(0, 3):
            t1.append(src_points[dt[i][j]])
            t2.append(dst_points[dt[i][j]])
            t.append(morph_points[dt[i][j]])
        if(checkLine(t) or checkLine(t1) or checkLine(t2)):
            continue
        core.morph_triangle(src_img, dst_img, res_img, t1, t2, t, alpha,i)

    return res_img
def checkLine(t):
    if(len(t) != 3):
        return True
    if(t[0] == t[1] or t[1] == t[2] or t[0] == t[2]) :
        return True
    return False

def face_merge(
            src_img,
            dst_img,
            out_img,
            alpha=0.75,
            k_size=(10,5),
            mat_multiple=0.5
        ):

    src_matrix, src_points, src_faces,err = core.face_points(src_img)
    ##直接将第一次寻找目标人物读取的人脸数据作为参数传过来，减少查询人脸识别API次数
    dst_matrix, dst_points, dst_faces,err = core.face_points(dst_img)

    if not (isinstance(src_img,np.ndarray) and isinstance(dst_img,np.ndarray)):
        src_img = cv2.imread(src_img, cv2.IMREAD_COLOR)
        dst_img = cv2.imread(dst_img, cv2.IMREAD_COLOR)

    dst_img = transformation_points(src_img=src_img, src_points=src_matrix[core.FACE_POINTS],
                                        dst_img=dst_img, dst_points=dst_matrix[core.FACE_POINTS])
    # 转换
    trans_file = 'images/' + "trans"+ '.jpg'
    cv2.imwrite(trans_file, dst_img)

    _, dst_points, trans_faces, err = core.face_points(dst_img)

    dst_img = morph_img(src_img, src_points, dst_img, dst_points, alpha)

    # 融合
    # morph_file = 'images/' + "merge" + '.jpg'
    # cv2.imwrite(morph_file, dst_img)
    dst_matrix, dst_points, morph_faces,err = core.face_points(dst_img)

    if isinstance(src_faces,dict):
        src_img = tran_src(src_img, src_points, dst_points,
                           [int(src_faces['x']), int(src_faces['y']), int(src_faces['width']),
                            int(src_faces['height'])])
    else:
        src_img = tran_src(src_img, src_points, dst_points, [int(src_faces[-1][0]),int(src_faces[-1][1]),int(src_faces[-1][2]),int(src_faces[-1][3])])
    # cv2.imwrite('images/' + "tran_src" + '.jpg',src_img)
    dst_img = merge_img(src_img, dst_img, dst_matrix, dst_points, k_size, mat_multiple)

    # 删除掉临时生成的文件
    # os.remove(trans_file)
    # os.remove(morph_file)

    cv2.imwrite(out_img, dst_img)

    return err


def face_merge_ret(
            src_img,
            dst_img,
            out_img,
            alpha=0.75,
            k_size=(10,5),
            mat_multiple=0.5
        ):

    src_matrix, src_points, src_faces,err = core.face_points(src_img)

    if(err != 0 or len(src_points) == 0):
        return src_img
    ##直接将第一次寻找目标人物读取的人脸数据作为参数传过来，减少查询人脸识别API次数
    
    dst_matrix, dst_points, dst_faces,err = core.face_points(dst_img)
    
    if(err != 0 or len(dst_points) == 0):
        return src_img
    
    if not (isinstance(src_img,np.ndarray)):
        print("read")
        src_img = cv2.imread(src_img, cv2.IMREAD_COLOR)
    
    if not (isinstance(dst_img,np.ndarray)):
        dst_img = cv2.imread(dst_img, cv2.IMREAD_COLOR)

    dst_img = transformation_points(src_img=src_img, src_points=src_matrix[core.FACE_POINTS],
                                        dst_img=dst_img, dst_points=dst_matrix[core.FACE_POINTS])
    # 转换
    trans_file = 'images/' + "trans"+ '.jpg'
    cv2.imwrite(trans_file, dst_img)

    _, dst_points, trans_faces, err = core.face_points(dst_img)

    dst_img = morph_img(src_img, src_points, dst_img, dst_points, alpha)

    # 融合
    morph_file = 'images/' + "merge" + '.jpg'
    cv2.imwrite(morph_file, dst_img)
    dst_matrix, dst_points, morph_faces,err = core.face_points(dst_img)

    if isinstance(src_faces,dict):
        src_img = tran_src(src_img, src_points, dst_points,
                           [int(src_faces['x']), int(src_faces['y']), int(src_faces['width']),
                            int(src_faces['height'])])
    else:
        src_img = tran_src(src_img, src_points, dst_points, [int(src_faces[-1][0]),int(src_faces[-1][1]),int(src_faces[-1][2]),int(src_faces[-1][3])])
    cv2.imwrite('images/' + "tran_src" + '.jpg',src_img)
    dst_img = merge_img(src_img, dst_img, dst_matrix, dst_points, k_size, mat_multiple)
    cv2.imwrite(out_img,dst_img)
    return dst_img