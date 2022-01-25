from platform import release
from cv2 import COLOR_BGR2GRAY
import cv2
import numpy as np
from PIL import Image
import dlib
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# creat list for landmarks
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

# create face detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# create VideoCapture object (input the video)
# 0 for web camera
# vid_in = cv2.VideoCapture(0)
# "---" for the video file
# vid_in = cv2.VideoCapture("baby_vid.mp4")

# capture the image in an infinite loop
# -> make it looks like a video
# while True:
# Get frame from video
# get success : ret = True / fail : ret = False
# ret, image_o = vid_in.read()

# resize the image
image_o_array = cv2.imread("./sample_image/005.png")
# print(image_o_array.shape)


image = cv2.resize(image_o_array, dsize=(
    512, 512), interpolation=cv2.INTER_AREA)
# 흑백으로 전환
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get faces (up-sampling=1)
# face_detector = detector(img_gray, 1)
face_detector = detector(image, 1)
# the number of face dtected
# print("The number of faces detected : {}".format(len(face_detector)))

# loop as the number of face
# one loop belong to one face
for face in face_detector:
    # face wrapped with rectangle
    # cv2.rectangle(image, (face.left(), face.top()),
    #               (face.right(), face.bottom()), (0, 0, 255), 1)

    # make prediction and transform to numpy array
    landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기

    # create list to contain landmarks
    landmark_list = []

    # append (x, y) in landmark_list
    # 얼굴 위 모든 landmark에 점 찍기
    for p in landmarks.parts():
        landmark_list.append([p.x, p.y])
        # cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)

left_eye_upper_side = landmark_list[36:40]
right_eye_upper_side = landmark_list[42:46]
# 왼쪽 눈 위에 점 찍기
# for left_eye_upper_point in left_eye_upper_side:
#     cv2.circle(image, left_eye_upper_point, 2, (255, 255, 255), -1)

# # 오른쪽 눈 위에 점 찍기
# for right_eye_upper_point in right_eye_upper_side:
#     cv2.circle(image, right_eye_upper_point, 2, (255, 255, 255), -1)

# 왼쪽 눈
left_eye_upper_side_x_pts = []
left_eye_upper_side_y_pts = []
for [x, y] in left_eye_upper_side:
    left_eye_upper_side_x_pts.append(x)
    # 사진에서 y좌표는 반대이므로,
    left_eye_upper_side_y_pts.append(512-y)

# 왼쪽 눈
right_eye_upper_side_x_pts = []
right_eye_upper_side_y_pts = []
for [x, y] in right_eye_upper_side:
    right_eye_upper_side_x_pts.append(x)
    # 사진에서 y좌표는 반대이므로,
    right_eye_upper_side_y_pts.append(512-y)

# 왼쪽 눈 interpolation 생성
left_xs = np.linspace(
    left_eye_upper_side_x_pts[0], left_eye_upper_side_x_pts[-1], 1000)
left_spl = UnivariateSpline(
    left_eye_upper_side_x_pts, left_eye_upper_side_y_pts)
# splined된 left_eye_position 리스트 생성
left_eye_position_splined = []
for i in range(1000):
    left_eye_position_splined.append([left_xs[i], float(left_spl(left_xs[i]))])
# plt.plot(left_xs, left_spl(left_xs), 'g', lw=3)
# 오른쪽 눈 interpolation 생성
right_xs = np.linspace(
    right_eye_upper_side_x_pts[0], right_eye_upper_side_x_pts[-1], 1000)
right_spl = UnivariateSpline(
    right_eye_upper_side_x_pts, right_eye_upper_side_y_pts)
# splined된 right_eye_position 리스트 생성
right_eye_position_splined = []
for i in range(1000):
    right_eye_position_splined.append(
        [right_xs[i], float(right_spl(right_xs[i]))])

# 그리기

# splined 된 left_eye_point cv2에 흰색 배경 그리기
for [x, y] in left_eye_position_splined:
    cv2.circle(image, [int(x), int(512-y-9)], 1, (255, 255, 255), 7)
# splined 된 left_eye_point cv2에 흰색 배경 그리기
for [x, y] in right_eye_position_splined:
    cv2.circle(image, [int(x), int(512-y-9)], 1, (255, 255, 255), 7)

# splined 된 left_eye_point cv2에 쌍커풀 라인 그리기
for [x, y] in left_eye_position_splined:
    cv2.circle(image, [int(x), int(512-y-6)], 1, (0, 0, 0), 1)
# splined 된 left_eye_point cv2에 쌍커풀 라인 그리기
for [x, y] in right_eye_position_splined:
    cv2.circle(image, [int(x), int(512-y-6)], 1, (0, 0, 0), 1)

cv2.imshow('result', image)

# wait for keyboard input
key = cv2.waitKey(0)

# if esc,
# if key == 27:
#     break

# vid_in.release()
