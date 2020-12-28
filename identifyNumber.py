import cv2
import numpy as np


def reshape(img,length,width):
    l,w = img.shape[0:2]
    while (l>length or w>width):
        l = int(l*2/3)
        w = int(w*2/3)
        if (l>length or w>width):
            img = cv2.resize(img,(l,w))
        else:
            img = cv2.resize(img,(length,width))
    return img


#计算图像面积
def getArea(img):
    area = 0
    for i in range(100):
        for j in range(100):
            if img[i,j] !=0:
                area = area + 1
    return area

#计算每个像素出现的概率
def getP(img):
    count = np.zeros(256, np.float)
    for i in range(100):
        for j in range(100):
            pixel = img[i, j]
            index = int(pixel)
            count[index] = count[index] + 1
    count = count / (100*100)
    return count




#特征1 圆度
def getRoundness(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#得到轮廓信息
    cnt = contours[0]                                                             #取第一条轮廓
    area = getArea(img)                                                           #计算面积
    perimeter = cv2.arcLength(cnt,False)                                          #计算周长
    Roundness = 4*3.14*area/(perimeter*perimeter)
    return Roundness


#特征2 偏心率
def getEccentricity(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#得到轮廓信息
    cnt = contours[0]                                                             #取第一条轮廓
    x, y, w, h = cv2.boundingRect(cnt)                                            #获得外接矩形,w:宽 h:长
    eccentricity = h/w
    return eccentricity

#特征3 归一化平滑度
def getR(img):
    (mean,stddv) = cv2.meanStdDev(img)      #均值&方差
    minMax = cv2.minMaxLoc(img)
    R = float(1 - 1/(1+stddv/minMax[1]/minMax[1]))
    return R

#特征4 一致性
def getU(img):
    p_i = getP(img)
    U = np.sum(p_i * p_i)
    return U

#特征5 平均熵
def getEntropy(img):
    p = getP(img)
    entropy = 0
    for i in range(255):
        if p[i] == 0:
            entropy += 0
        else:
            entropy -= p[i]*np.log2(p[i])
    return entropy


def couculate(path='test.jpg'):
    #prepare processing
    #读取图像->灰度化->->缩放100*100->局部阈值二值化(0,255)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    scaling = reshape(gray,100,100)
    dst = cv2.adaptiveThreshold(scaling, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)

    a = getRoundness(dst)
    b = getEccentricity(dst)
    c = getR(scaling)
    d = getU(scaling)
    e = getEntropy(scaling)

    A = [a,b,c,d,e]
    return A

def getChara(path = 'jpg\\test00.jpg'):
    A = np.array([0,0,0,0,0])
    B = np.zeros((10,5))
    for i in range(10):
        path = list(path)
        path[9] = str(i)
        path = "".join(path)
        for j in range(10):
          print(i,j)
          path = list(path)
          path[8] = str(j)
          path = "".join(path)
          print(path)
          A = A + np.array(couculate(path))
        B[i,:] = A/10
        A = [0,0,0,0,0]
    return B

def detective(img='test.jpg'):
    A = getChara()
    B = couculate(img)
    temp = 0
    C = []

    for i in range(10):
        for j in range(5):
            temp = B[j] - A[i][j]
            print(i, j, temp)
        C.append(abs(temp / 5))

    return (C.index(min(C)))

def main():
    result = detective('test.jpg')
    print(result)


if __name__ == '__main__':
    main()
