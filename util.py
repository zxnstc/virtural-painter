import mediapipe as mp
import numpy as np
import requests
import cv2

# 摄像头画面宽、高
HEIGHT = 1280
WIDTH = 720
HEADER_HEIGHT = 125


class HandDetector():
    # 手掌识别类 主要是调用mp.solutions.hands
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode  # 静态图像还是连续帧视频
        self.maxHands = maxHands  # 最多识别多少只手，手越多识别越慢
        self.modelComplex = modelComplexity  # 手部标志模型的复杂性 默认为1
        self.detectionCon = detectionCon  # 检测置信度阈值
        self.trackCon = trackCon  # 各帧之间跟踪置信度阈值

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # 五指指尖关键点序号

    def findHands(self, img, draw=True):
        """
        绘制识别结果
        Args:
            img: 要识别的图像
            draw: 是否绘制识别结果
        Returns:
            完成识别结果绘制后的图像
        """
        # opencv中图像默认是BGR格式，BGR转RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process()是手势识别最核心的方法，通过调用这个方法，将窗口对象作为参数，mediapipe就会将手势识别的信息存入到results对象中
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # multi_hand_landmarks: 被检测/跟踪的手的集合，
        # 其中每只手被表示为21个手部地标的列表，每个地标由x、y和z组成。
        # x和y分别由图像的宽度和高度归一化为[0.0,1.0]。
        # Z表示地标深度，以手腕深度为原点，值越小，地标离相机越近。 z的大小与x的大小大致相同
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    # handLms 关键点
                    # mpHands.HAND_CONNECTIONS 绘制手部关键点之间的连线
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        找出关键点坐标
        Args:
            img: 已完成识别和结果绘制的图像
            handNo: 手掌的编号(由于只有1只手，为0)
            draw: 是否绘制识别结果
        Returns:
            关键点的坐标集( [关键点序号, 该点x坐标, 该点y坐标]的list )
        """
        xList = []  # 关键点的x坐标
        yList = []  # 关键点的y坐标
        bbox = []
        within = True
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                # 获取手指关节点
                # id 为关键点序号
                # lm 为该关键点的xyz坐标
                h, w, c = img.shape
                # 将归一化的坐标转成真实像素坐标
                cx, cy = int(lm.x * w), int(lm.y * h)
                # 判断是否都在界内
                if cx > HEIGHT or cx < 0:
                    within = False
                elif cy > WIDTH or cy < HEADER_HEIGHT:
                    within = False
                # 记录坐标
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                # 相当于将multi_hand_landmarks里的坐标转存到lmList属性里
                self.lmList.append([id, cx, cy])
                # 将关键点画出
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if xList and yList:
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
        else:
            return [], [], False
        # 画矩形
        if draw:
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                          (0, 255, 0), 2)

        return self.lmList, bbox, within

    def fingersUp(self):
        fingers = []
        # Thumb
        # 大拇指的指尖和它的下一个关节比，若指尖的y坐标大，视为该指的指向是向上的
        if self.lmList[self.tipIds[0]][2] < self.lmList[self.tipIds[0] - 1][2]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            # 其余四指的指尖和它的下两个关节比，若指尖的y坐标大，视为该指的指向是向上的
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2] and \
                    self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


# 16进制颜色格式颜色转换为RGB格式
def Hex_to_RGB(hex, type):
    """
    16进制颜色编号转RGB/BGR
    Args:
        hex: 16进制颜色编号
        type: RGB/BGR
    Returns:
        转换后的颜色元组
    """
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    if type == 'RGB':
        return r, g, b
    elif type == 'BGR':
        return b, g, r


def getText(filePath):
    """
    识别手写文字
    Args:
        filePath: 手写问题图片路径
    Returns:
        识别出来的文字的字符串
    """
    password = '8907'
    url = "http://www.iinside.cn:7001/api_req"

    data = {
        'password': password,
        'reqmode': 'ocr_pp'
    }
    files = [('image_ocr_pp', ('ocr.jpg', open(filePath, 'rb'), 'application/octet-stream'))]
    headers = {}
    response = requests.post(url, headers=headers, data=data, files=files)
    text = response.text.split('data')[1].split('}')[0].split('"')[2]
    return text


def imgMerge(imgSource, imgTarget):
    """
    将imgSource图片合并至imgTarget
    Args:
        imgSource: 将该图片
        imgTarget: 合并到该图片
    Returns:
        合并后的图片
    """
    # 转灰度
    imgGray = cv2.cvtColor(imgSource, cv2.COLOR_BGR2GRAY)
    # 黑白二值反转化 大于50则变成0 其余为255
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    # 转成RGB
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # imgInv 是白底画板，画笔画黑线，擦除相同
    # 图像与
    imgTarget = cv2.bitwise_and(imgTarget, imgInv)  # 把画的黑线区域"与"成0，方便后面"或"操作
    # 图像或
    imgTarget = cv2.bitwise_or(imgTarget, imgSource)

    return imgTarget
