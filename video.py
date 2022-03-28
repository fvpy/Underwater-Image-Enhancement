import cv2
import numpy as np

def white_balance(img):
    wb = cv2.xphoto.createSimpleWB()
    return wb.balanceWhite(img)

def luminance(img):
    out_channels = []
    clahe=cv2.createCLAHE(clipLimit=1.0,tileGridSize=(8,8))
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y,cr,cb = cv2.split(imgYCC)
    Yclahe = clahe.apply(y)
    out_channels.append(Yclahe)
    out_channels.append(cr)
    out_channels.append(cb)
    img= cv2.merge(out_channels)
    return cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    
def path_two(img):
    out_channels = []
    clahe=cv2.createCLAHE(clipLimit=1.2,tileGridSize=(8,8))
    for channel in cv2.split(img):
        clahe_channel = clahe.apply(channel)
        out_channels.append(clahe_channel)
    img = luminance(cv2.merge(out_channels))
    return white_balance(img)
   
def path_one(img, p=1):
    out_channels = []
    limits = (
        img.shape[0] * img.shape[1] * p / 200.0,
        img.shape[0] * img.shape[1] * (1 - p / 200.0)
    )
    for channel in cv2.split(img):
        histogram = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_limit, high_limit = np.searchsorted(histogram, limits)
        lut = np.concatenate((
            np.zeros(low_limit),
            np.around(np.linspace(0, 255, high_limit - low_limit + 1)),
            255 * np.ones(255 - high_limit)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

if __name__ == '__main__':

    #input path
    cap=cv2.VideoCapture('input.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width=640
    frame_height=480

    #output path
    save = cv2.VideoWriter('output.mov',fourcc, 40.0, (frame_width,frame_height))

    while(cap.isOpened()):
        ret, frame=cap.read()
        if(ret== True):
            frame=cv2.resize(frame, (frame_width, frame_height))
            img_p1 = path_one(frame, 1)
            img_p2 = path_two(frame)
            out=cv2.addWeighted(img_p2, 0.35, img_p1, 0.65, 0, dst=None, dtype=None) 
            cv2.imshow("original",frame) 
            cv2.imshow("enhance", out)
            save.write(out)
            key=cv2.waitKey(1)
            if key==27:
                break
        else:
            break   
    cap.release()
    save.release()
    cv2.destroyAllWindows()
    

