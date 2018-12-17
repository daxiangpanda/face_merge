# 1 分解视频
# 2 依次处理
# 3 合成视频
# 4 添加背景音乐
import cv2
import core
import time
def fetch_frame(video_path):
    video = cv2.VideoCapture(video_path)
    success,image = video.read()
    count = 0
    frame_path_list = []
    while success:
        cv2.imwrite("image/frame%d.jpg" % count, image)     # save frame as JPEG file      \
        # try:
        start = time.time()
        changeFace("image/frame%d.jpg" % count,"images/model_zbll.jpg")
        print(time.time() - start)
        # except Exception as e:
        #     count += 1
        #     continue
        success,image = video.read()
        frame_path_list.append("image/frame%d.jpg" % count)
        count += 1

    return frame_path_list

def changeFace(src_img,dst_img):
    core.face_merge_ret(src_img,dst_img,src_img) 

fetch_frame("./gesture.MP4")