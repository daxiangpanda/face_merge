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
    return core.face_merge_ret(src_img,dst_img) 

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(1, 10.0)

while True:
    ret,frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 1)
        # a = out.write(frame)
        cv2.imshow("frame", changeFace(frame,"images/model_zbll.jpg"))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
