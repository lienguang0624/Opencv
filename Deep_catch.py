import numpy as np
import cv2 as cv
import binocular
import Binocular_correct
import monocular

# cv.namedWindow("left")
# cv.namedWindow("right")
# cv.namedWindow("depth")
# cv.moveWindow("left", 0, 0)
# cv.moveWindow("right", 600, 0)
# cv.createTrackbar("num", "depth", 0, 10, lambda x: None)
# cv.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)


def main(boardSize,pathl,pathr,board_distance,path_imagel,path_imager):
    path_imagel = cv.imread(path_imagel)
    path_imager = cv.imread(path_imager)
    # 双目图像校正
    leftMaps, rightMaps, Rl, Rr, Pl, Pr, Q = Binocular_correct.getMaps(boardSize, pathl, pathr,board_distance)
    imagelrmap = cv.remap(path_imagel, leftMaps[0], leftMaps[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    imagerrmap = cv.remap(path_imager, rightMaps[0], rightMaps[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv.cvtColor(imagelrmap, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(imagerrmap, cv.COLOR_BGR2GRAY)
    cv.imshow("grayL", imgL)
    cv.imshow("grayR", imgR)

    # 制作滑动测试窗口
    cv.namedWindow("depth")
    cv.createTrackbar("num", "depth", 0, 10, lambda x: None)
    cv.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)
    num = cv.getTrackbarPos("num", "depth")
    blockSize = cv.getTrackbarPos("blockSize", "depth")
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5


    cv.waitKey(-1)

if __name__=="__main__":
    main((6,9),'C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\Opencv\\left','C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\Opencv\\right',10,'C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\Opencv\\left\\left01.jpg','C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\Opencv\\right\\right01.jpg')