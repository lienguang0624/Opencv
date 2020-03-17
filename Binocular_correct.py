import cv2 as cv
import numpy as np
import binocular
import monocular
import os
'''
获得校正之后的两个相机的map
输入：
boardSize：棋盘格大小
leftPath：左侧相机照片文件夹
rightPath：右侧相机照片文件夹
输出：
leftMaps：左侧相机的map
rightMaps：右侧相机的map
'''
def getMaps(boardSize,leftPath,rightPath,board_distance):
    #获得内矩阵、畸变参数以及R, T, E, F
    cml, dcl, cmr, dcr, R, T, E, F, imageSize = \
        binocular.cv_stereoCal(boardSize, leftPath, rightPath,board_distance)
    # cml - 第一个摄像机的摄像机矩阵
    # dcl - 第一个摄像机的畸变向量
    # cmr - 第二个摄像机的摄像机矩阵
    # dcr - 第二个摄像机的畸变向量
    # imageSize - 图像大小
    # R - stereoCalibrate() 求得的R矩阵
    # T - stereoCalibrate() 求得的T矩阵
    # R1 - 输出矩阵，第一个摄像机的校正变换矩阵（旋转变换）
    # R2 - 输出矩阵，第二个摄像机的校正变换矩阵（旋转矩阵）
    # P1 - 输出矩阵，第一个摄像机在新坐标系下的投影矩阵
    # P2 - 输出矩阵，第二个摄像机在想坐标系下的投影矩阵
    # Q - 4 * 4的深度差异映射矩阵
    Rl, Rr, Pl, Pr, Q, validPixROI1, validPixROI2 = \
        cv.stereoRectify(cml, dcl, cmr, dcr, imageSize, R, T, 0, (0, 0))
    # camera_matrix——输入的3X3的摄像机内参数矩阵
    # dist_coeffs——输入的5X1的摄像机畸变系数矩阵
    # R——输入的第一和第二相机坐标系之间3X3的旋转矩阵
    # new_camera_matrix——输入的校正后的3X3摄像机矩阵
    # mapx——输出的X坐标重映射参数
    # mapy——输出的Y坐标重映射参数
    leftMaps = cv.initUndistortRectifyMap(
        cml, dcl, Rl, Pl, (imageSize[1], imageSize[0]), cv.CV_16SC2)
    print(leftMaps[1])
    rightMaps = cv.initUndistortRectifyMap(
        cmr, dcr, Rr, Pr, (imageSize[1], imageSize[0]), cv.CV_16SC2)
    return leftMaps,rightMaps, Rl, Rr, Pl, Pr, Q
'''
检测校正效果
输入：
boardSize：棋盘格大小
leftMaps：左侧相机的map
rightMaps：右侧相机的map
leftImagePath：左侧相机照片路径
rightImagePath：右侧相机照片路径
'''
def check(boardSize,leftMaps,rightMaps,leftImagePath,rightImagePath):
    imagel = cv.imread(leftImagePath)
    imager = cv.imread(rightImagePath)
    imageSize = imagel.shape
    cv.imshow('1', imagel)

    # 图片进行重投影
    imagelrmap = cv.remap(imagel, leftMaps[0], leftMaps[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    imagerrmap = cv.remap(imager, rightMaps[0], rightMaps[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    # 图片进行缩放
    imageShowl = cv.resize(imagelrmap, (imageSize[1], imageSize[0]), 0, 0, cv.INTER_AREA)
    imageShowr = cv.resize(imagerrmap, (imageSize[1], imageSize[0]), 0, 0, cv.INTER_AREA)

    # 两张图片水平堆叠
    imageShow = np.hstack((imageShowl, imageShowr))
    for i in range(0, imageShow.shape[0], 16):
        cv.line(imageShow, (0, i), (imageShow.shape[1], i), (0, 0, 255), 1, 8)
    cv.imshow('rectified', imageShow)
    cv.waitKey(-1)

def main(boardSize,pathl,pathr,board_distance):
    leftMaps, rightMaps, Rl, Rr, Pl, Pr, Q = getMaps(boardSize, pathl, pathr,board_distance)
    filelist1 = monocular.getFilelist(pathl)
    filelist2 = monocular.getFilelist(pathr)
    file1 = filelist1[int(len(filelist1) / 2)]
    file2 = filelist2[int(len(filelist2) / 2)]
    check(boardSize, leftMaps, rightMaps, file1, file2)
    if not os.path.exists('./result/'):
        os.mkdir('./result/')
    np.savetxt("./result/rectify.txt",(Rl, Rr, Pl, Pr, Q),'%s',',','\n')

if __name__=="__main__":
    main((6,9),'C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\Opencv\\left','C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\Opencv\\right',1)