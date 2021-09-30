import cv2
import imutils
import sys
import math
import numpy as np
import os
import argparse


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filename', required=True,
                    help='Filename of video')
    ap.add_argument('-i', '--image', required=False,
                    help='Filename of the calibration image')
    ap.add_argument('-p', '--preview', required=False,
                    help='Show a preview of the image',
                    action='store_true')               
    ap.add_argument('-c', '--calibration', required=False,
                    help='Show a preview of the calibration image',
                    action='store_true')  
    ap.add_argument('-s', '--stationary',
                    help='Stationary point hsv color bound', required=False,nargs="+", type=int)
    ap.add_argument('-m', '--moving', 
                    help='Moving point hsv color bound', required=False,nargs="+", type=int)  
    ap.add_argument('-r', '--crop', 
                    help = 'Crop x frames from beginning.', required=False, type=int, default=0)  
    args = vars(ap.parse_args())

    # if not xor(bool(args['image']), bool(args['webcam'])):
    #     ap.error("Please specify only one image source")

    # if not args['filter'].upper() in ['RGB', 'HSV']:
    #     ap.error("Please speciy a correct filter.")

    return args

# filename = 'erfan_1.mov'
# calibration_frame = 'erfan_1.jpg'
args = get_arguments()
filename = args['filename']
calibration_frame = None
if 'calibration' in args.keys():
    calibration_frame = args['image']

# print(filename, calibration_frame)
vid = cv2.VideoCapture(filename)
# rret, calibration = vid.read()


#Jae
# calibration_resize = imutils.resize(calibration, width=300)
# cv2.imwrite('lab3.jpg', calibration_resize)

def get_center(image, lower, upper, miny = False, minarea=False):
    image1 = cv2.GaussianBlur(image, (11, 11), 0)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image1, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        print('ISSUE')
        if miny or minarea:
            return -1, -1, -1
        else:
            return -1, -1
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # print(f'RECT {cv2.boundingRect(c)}')
    if miny:
        return center, cv2.contourArea(c), cv2.boundingRect(c)
    elif minarea:
        return center, cv2.contourArea(c), cv2.minAreaRect(c)
    else:
        return center, cv2.contourArea(c)

def clamp(v, min, max):
    if v < min:
        return min
    elif v > max:
        return max
    return v

def dist(l1, l2):
    return math.sqrt(sum(map(lambda x, y: (x-y)**2, l1, l2)))

def avg(l):
    return sum(l)/len(l) if len(l) > 0 else math.inf

def sign(i):
    return i//abs(i) if i != 0 else 0

#Constants
fps = 30
spf = 1/fps



#Andy
# sp_lower = (154, 144, 109)
# sp_upper = (172, 255, 255)

#Richard
# sp_lower = (163, 75, 0)
# sp_upper = (182, 255, 255)

#Jae
# sp_lower = (0, 196, 87)
# sp_upper = (255, 255, 255)

#Erfan
# sp_lower = (0, 172, 104)
# sp_upper = (9, 255, 255)


# print(args['stationary'])

# sp_lower = tuple(args['stationary'][:3])
# sp_upper = tuple(args['stationary'][3:])


#Andy
# mp_lower = (118, 45, 0)
# mp_upper = (141, 255, 255)

#Richard
# mp_lower = (108, 68, 19)
# mp_upper = (134, 255, 113)

#Jae
# mp_lower = (92, 19, 0)
# mp_upper = (168, 255, 213)

#Erfan
# sp_lower = (57, 102, 5)
# sp_upper = (151, 255, 255)

def get_trackbar_values(range_filter):
    values = []
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values

def callback(value):
    pass

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)
    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255
        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)




#Calibration


if calibration_frame:
    calibration = cv2.imread(calibration_frame)
else:
    temp, calibration = vid.read()
calibration = imutils.resize(calibration, width=600)
range_filter = 'HSV'
mp_lower = None
mp_upper = None
if args['moving'] is not None:
    mp_lower = tuple(args['moving'][:3])
    mp_upper = tuple(args['moving'][3:])
else:
    setup_trackbars(range_filter)
    while True:
        image = calibration.copy()
        frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)
        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        # if args['preview']:
        preview = cv2.bitwise_and(image, image, mask=thresh)
        cv2.imshow("Preview", preview)
        # else:
        #     cv2.imshow("Original", image)
        #     cv2.imshow("Thresh", thresh)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break
    mp_lower = (v1_min, v2_min, v3_min)
    mp_upper = (v1_max, v2_max, v3_max)



# calibration = cv2.imread(calibration_frame)

ogimage = calibration.copy()
# maskc = cv2.inRange(calibration, sp_lower, sp_upper)
# s_pos, s_area, s_miny = get_center(calibration, sp_lower, sp_upper, miny=True)
# s_pos = [300, 0]
cv2.namedWindow('image')
cv2.imshow('image', calibration)
# s_pos = [300, 0]
s_pos = [300, 0]
def find_pivot(event, x, y, flags, param):
    global s_pos
    if(event == cv2.EVENT_FLAG_LBUTTON):  
        s_pos = [x, y]
    elif event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(calibration,(x,y),5,(255,0,0),-1)
        cv2.imshow('image', calibration)

cv2.setMouseCallback('image',find_pivot)
while True:
    calibration = ogimage.copy()
    cv2.imshow('image', calibration)
    k1 = cv2.waitKey(2)
    if k1 == ord('q'):
        break
# cv2.waitKey(-1)
# print(s_pos)
# cv2.waitKey(-1)

# s_pos = [s_pos[0], s_miny[1]]
#maskm = cv2.inRange(calibration, mp_lower, mp_upper)
init_pos, m_area = get_center(calibration, mp_lower, mp_upper)
#print(init_pos)

real_wire_len = 1.3 #REMEMBER TO FUCKING CHANGE THIS/ ALSO ITS IN <INSERT UNIT HERE>
fake_wire_len = int(dist(s_pos, init_pos))
cfactor = real_wire_len/fake_wire_len
m_pos = [s_pos[0], int(s_pos[1] + fake_wire_len)]
init_dist = dist(init_pos, m_pos)



cv2.circle(calibration, s_pos, 5, (0, 0, 255), -1) #pin
cv2.circle(calibration, init_pos, 5, (0, 255, 0), -1) #object
cv2.line(calibration, s_pos, init_pos, (255, 0, 0), thickness=2) #string
cv2.circle(calibration, m_pos, 5, (0, 255, 255), -1)


if args['calibration']:
    cv2.imshow('calib', calibration)
    #cv2.waitKey(0)
    cont = cv2.waitKey(-1)
    if cont == ord('e'):
        sys.exit(0)

cv2.destroyAllWindows()

# theta = 2*math.asin((init_dist/2)/fake_wire_len)
theta = math.acos((init_pos[1] - s_pos[1])/dist(init_pos, s_pos))
init_amplitude = theta




xs = []
ys = []
xerrors = []
yerrors = []
q_factor = -1
counter = 0
oscillations = 0 #multiplied by 2
N = 2

cur_amplitude = 0
counter2 = 0
in_oscil = False


#print(calibration.shape)

# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# writer = cv2.VideoWriter('LFG.avi', fourcc, 10, (450, 600), isColor=True)

amplitude = init_amplitude
#print(amplitude/2/math.pi*360)
last_thetas = []
increasing = False
KEEP_N = 5
periods = []
last_time = 0
LAST_N = 10
crop = args['crop']
amplitudesx = []
amplitudes = []
increasing_time = 0
decreasing_time = 0
increasing_padding = 1.05
decreasing_padding = 1

amplitude2 = []

while(vid.isOpened()):
    oscil_thresh = clamp(amplitude*0.2, (10 - min(oscillations//15, 6))/360*2*math.pi, 15/360*2*math.pi)
    ret, frame = vid.read()
    if ret:
        if counter//spf < crop:
            counter += spf
            continue

        frame = imutils.resize(frame, width=600)
        center, area, minarea = get_center(frame, mp_lower, mp_upper, minarea=True)

        if center != -1:
            #print(minarea)
            #cv2.waitKey(-1)
            bbox = cv2.boxPoints(minarea)
            bbox = np.int0(bbox)
            ps = sorted(bbox, key=lambda x: x[1])
            # center = [(ps[0][0] + ps[1][0])/2, (ps[0][1] + ps[1][1])/2]
            # center = tuple(map(int, center))
            # print(center)
            fake_wire_len = int(dist(center, s_pos))
            m_pos = (s_pos[0], s_pos[1] + fake_wire_len)
            cv2.circle(frame, s_pos, 5, (0, 0, 255), -1) #pin
            cv2.circle(frame, m_pos, 5, (0, 255, 0), -1) #old object
            cv2.line(frame, s_pos, m_pos, (255, 0, 0), thickness=2) #string
            cv2.line(frame, s_pos, center, (0, 255, 255), thickness=2) #new object
            box = cv2.boxPoints(minarea)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), -1)
            cv2.circle(frame, center, 5, (255, 255, 255), -1) #new object
            cv2.line(frame, ps[0], ps[1], (0, 0, 0), thickness=2)
            cv2.ellipse(frame, s_pos, (int(fake_wire_len*0.5), int(fake_wire_len*0.5)),0,90-amplitude/2/math.pi*360,90+amplitude/2/math.pi*360,(255, 255, 0),-1)
            
            # xs.append(counter) 
            d_original = dist(center, m_pos)
            # theta = math.asin((center[0] - s_pos[0])/fake_wire_len)
            theta_not_abs = math.acos((center[1] - s_pos[1])/dist(center, s_pos))*sign(center[0] - s_pos[0])
            theta = abs(theta_not_abs)
            print(f'Theta: {theta}, Average Theta: {avg(last_thetas)}, Difference: {theta - avg(last_thetas)*increasing_padding}')
            # print(last_thetas)
            if theta < avg(last_thetas)*increasing_padding:
                if not increasing and counter//spf > crop:
                    last_thetas = []
                    if decreasing_time <= 3:
                        print('ISSUE')
                    else:
                        increasing_time = 0
                        decreasing_time = 0
                        print('HIT')
                        counter2 += 1

                        if counter2 >= 2:
                            oscillations += 1
                            counter2 = 0
                            print('AMPLITUDES:', cur_amplitude, amplitude)
                            amplitude = (cur_amplitude + theta)/2
                            amplitudesx.append(counter)
                            amplitudes.append(amplitude)
                            # cur_amplitude = 0
                            cur_period = counter - last_time
                            if abs(cur_period - avg(periods)) > avg(periods)*0.05 and oscillations > 11:
                                print('DISCREPANCY')
                                kk = sys.stdin.read(1)
                                if kk == 'q':
                                    break
                            periods.append(cur_period)
                            last_time = counter
                            periods = periods[::-1][:LAST_N][::-1]
                            if amplitude <= init_amplitude*math.e**(-math.pi/N) and q_factor == -1:
                                q_factor = oscillations*N
                                open(f'q_factor_{filename[:-4]}.txt', 'w').write(str(q_factor))
                                print('Q FACTOR:', q_factor)
                                # cv2.waitKey(-1)
                        else:
                            cur_amplitude = theta
                            # cur_amplitude = 0
                increasing_time += 1
                increasing = True
            else:
                decreasing_time += 1
                increasing = False 
            last_thetas.append(theta)
            last_thetas = last_thetas[::-1][:KEEP_N][::-1]
            #in_oscil = False

            print(f'Amplitude: {amplitude}, Theta: {theta}')
            print(f'Oscillations: {oscillations}, Counter: {counter2}, Q Factor: {q_factor}')
            print(f'Increasing: {increasing}, Period: {periods[-1] if len(periods) > 0 else None}, Avg Period: {avg(periods)},  Oscillations: {oscillations}, COUNTS: {increasing_time} {decreasing_time}')
            # print(chr(27) + "[2J")
            # amplitude = theta*fake_wire_len
            xs.append(counter)
            ys.append(theta_not_abs)
            xerrors.append(spf/2)
            yerrors.append(min(minarea[1])/fake_wire_len)
        if args['preview']:
            cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
            frame1 = imutils.resize(frame, width=400)
            cv2.imshow('frame', frame1)
            cv2.waitKey(1)
    else:
        break
    counter += spf 
# print(xs[:10])
print(q_factor)
cv2.destroyAllWindows()
#writer.release()
vid.release()
f = open(f'data_{filename[:-4]}.txt', 'w')
for x, y, xerror, yerror in zip(xs, ys, xerrors, yerrors):
    f.write(f'{x} {y} {xerror} {yerror}\n')

f.close()
f1 = open(f'amplitudes_{filename[:-4]}.txt', 'w')
for x, y in zip(amplitudesx, amplitudes):
    f1.write(f'{x} {y}\n')
f1.close()
print(mp_lower)
print(mp_upper)

