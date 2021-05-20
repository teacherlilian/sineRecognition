import cv2
import os,sys
import time
import uuid
IMAGES_PATH='Tensorflow/workspace/images/collectedimages'
labels =['hello','thanks','yes','no','i love you']
number_imgs=15;
for label in labels:
    os.makedirs('Tensorflow\\workspace\\images\\collectedimages\\'+label)#这里要用makedirs，创建各级目录，如果没有就完成创建
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        imagename=os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename,frame)
        cv2.imshow('frame',frame)
        time.sleep(2)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
