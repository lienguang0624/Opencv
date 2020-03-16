#用到了单目标定中的函数
import monocular
import cv2 as cv
import numpy as np
import os

# 采用OpenCV的API进行立体视觉标定
def cv_stereoCal(boardSize,pathl,pathr,board_distance):
    filelistl = monocular.getFilelist(pathl)#得到待标定的图像序列
    filelistr = monocular.getFilelist(pathr)
    # 获得角点坐标（图像坐标，物理坐标，图像尺寸）
    imagePointsl, objectCornerls, imageSizel = monocular.findPoints(filelistl, boardSize,board_distance)
    imagePointsr, objectCornerrs, imageSizer = monocular.findPoints(filelistr, boardSize,board_distance)
    objectCorners=[]
    for i in range(len(filelistl)):
        objectCorners.append(objectCornerls)
        imagePointsl[i] = imagePointsl[i].reshape((-1, 2))
        imagePointsr[i] = imagePointsr[i].reshape((-1, 2))

    # 获得camera matrix
    cml, dcl, rvls, tvls = monocular.cv_calibrate(imagePointsl, objectCornerls, imageSizel)
    cmr, dcr, rvrs, tvrs = monocular.cv_calibrate(imagePointsr, objectCornerrs, imageSizer)
    # 进行stereoCal,获得R,T,E,F
    # objectCorners– 校正的图像点向量组.
    # imagePointsl–通过第一台相机观测到的图像上面的向量组
    # imagePointsr–通过第二台相机观测到的图像上面的向量组.
    # cml– 输入/输出第一个相机的内参数矩阵
    # dcl– 输入/输出第一个相机的畸变系数向量
    # cmr– 输入/输出第二个相机的内参数矩阵
    # dcr– 输入/输出第二个相机的畸变系数向量
    # R– 输出第一和第二相机坐标系之间的旋转矩阵。
    # T– 输出第一和第二相机坐标系之间的旋转矩阵平移向量
    # E–输出本征矩阵
    # F–输出基础矩阵
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F= cv.stereoCalibrate(objectCorners,
                           imagePointsl, imagePointsr,
                           cml, dcl, cmr, dcr,
                           imageSizel,
                           flags=cv.CALIB_FIX_INTRINSIC)
    #返回计算结果
    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, imageSizel

def main(boardSize, pathl, pathr,board_distance):
    cml, dcl, cmr, dcr, cvR, cvT, cvE, cvF, _ = cv_stereoCal(boardSize, pathl, pathr,board_distance)
    if not os.path.exists('./result/'):
        os.mkdir('./result/')
    np.savetxt("./result/stereCal.txt",(cml, dcl, cmr, dcr, cvR, cvT, cvE, cvF),'%s',',','\n')

if __name__=="__main__":
    main((6,9),'C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\Opencv\\left','C:\\Users\\lieng\\OneDrive\\Documents\\GitHub\\Opencv\\right',10)
