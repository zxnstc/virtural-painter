import cv2
import numpy as np
from util import *
from flask import Flask, render_template, request, Response, redirect, url_for, session
import os

"""
双指选择 单指绘画
初始时，指尖无圆点，无法进行绘画；
选择第一个 即彩笔时 默认蓝色 可选择颜色 单指绘画 双指暂停绘画
选择第二个 即形状时 默认矩形 可选择形状 第一次双指切单指时确定一个点 再切回双指时确定另一个点
    其中 选择多边形时
        初次双指切单指 不作为
        单指切双指 定位一个点
        双指切单指 画出之前的多边形(不闭合) 从最后一个点画线至当前单指指尖
        双指切三指 (食指、中指、小拇指向上指) 将之前的多边形闭合
选择第三个 即线条粗细时 选择线条粗细 为彩笔、形状、文字的粗细
选择第四个 即文本时 选择文本字体大小 第一次双指切单指时显示文字跟随食指指尖 第二次单指切双指时确定文字位置 开始手写识别
中间为可变工具栏 根据其他选项产生对应的选项
选择最后一个 即橡皮擦时 可选择橡皮擦粗细
"""
app = Flask(__name__)

@app.route('/')
def main():
    # 线条粗细
    brushThickness = 15
    # 橡皮擦粗细
    eraserThickness = 70
    # 字体大小
    text_size = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 画笔颜色集(BGR)
    colorList = ['#004aad', '#ff1616', '#008073', '#ff914d']
    # 当前画笔颜色在颜色集中的下标
    colorIndex = 0  # 初始化为蓝色
    # 橡皮擦颜色
    eraserColor = (0, 0, 0)  # 初始化为黑色
    # 画笔颜色 (画笔、形状、文本颜色)
    drawColor = Hex_to_RGB(colorList[colorIndex], 'BGR')  # 初始化为蓝色
    # 形状集
    shapeList = ["circle", "rectangle", "ellipse", "polygon"]
    # 当前形状在形状集中的下标
    shapeIndex = 1
    # 初始坐标 上一帧中指尖的坐标
    xp, yp = 0, 0
    # "": 空, "line": 线条模式, "shape": 形状模式, "text": 文本模式, "eraser": 橡皮擦模式
    mode = ""
    # "color": 颜色选择模式,"shape": 形状选择模式, "brush_thickness": 线条粗细选择模式, "eraser_thickness": 橡皮擦粗细选择模式
    # "text_size": 文字大小选择模式
    additional = ""
    current_shape = shapeList[shapeIndex]
    # 图形模式 第一次从两指状态变向单指时为图形一点，再从单指变为双指时为矩形另一点
    # 图形起始坐标(一个角的一点
    xr_s, yr_s = -1, -1
    # 文本放置状态
    text_picked = False
    text_putted = False
    # 文本输入状态
    text_enter = 0
    # 文本是否已被识别
    text_rec = False
    # 文本坐标
    x_w = -1
    y_w = -1
    # 文本内容
    text = ""
    # 单个手写字符识别结果的画布 (黑底白字)
    imgResult = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
    # 单个手写字符手写结果的画布 (白底黑字)
    # imgWriting = np.full((WIDTH, HEIGHT, 3), 255, dtype=np.uint8)
    imgWriting = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
    # 多边形点
    poly_list = []
    # 多边形绘画状态 0: 未选择多边形或刚完成一个多边形的绘画, 1: 上一个点已确定, 2:当前点在移动(待确定)
    poly_status = 0
    # 清楚状态
    clean = False

    def draw_shape(image, x_start, y_start, x_end, y_end):
        """
        画图形(多边形除外，即圆形、矩形、椭圆)
        Args:
            image: 绘画的画布
            x_start: 起始 x 坐标
            y_start: 起始 y 坐标
            x_end: 终止 x 坐标
            y_end: 终止 y 坐标
        Returns:
            完成图形绘画的画布
        """
        if current_shape == "rectangle":
            cv2.rectangle(image, (x_start, y_start + 25), (x_end, y_end - 25), drawColor, brushThickness)
        elif current_shape == "circle":
            cv2.circle(image, ((x_start + x_end) // 2, (y_start + y_end) // 2),
                       min((abs(x_end - x_start) // 2), (abs(y_end - y_start) // 2)), drawColor, brushThickness)
        elif current_shape == "ellipse":
            cv2.ellipse(image, ((x_start + x_end) // 2, (y_start + y_end) // 2),
                        ((abs(x_end - x_start) // 2), (abs(y_end - y_start) // 4)), 0, 0, 360, drawColor,
                        brushThickness)
        elif current_shape == "polygon":
            pass
        return image

    # 工具栏图片所在文件夹
    toolbar_folder = "ToolBar"
    toolbarFileList = os.listdir(toolbar_folder)
    # 按照数字顺序
    toolbarFileList.sort(key=lambda x: int(x.split(".")[0]))
    print(toolbarFileList)
    # list存储工具栏每个状态的图片
    toolbarList = []
    for imPath in toolbarFileList:
        image = cv2.imread(f'{toolbar_folder}/{imPath}')
        toolbarList.append(image)
    print(len(toolbarList))
    # 初始状态
    header = toolbarList[0]

    # 对于笔记本内置摄像头，参数为0
    cap = cv2.VideoCapture(0)

    cap.set(3, HEIGHT)  # 图像宽度
    cap.set(4, WIDTH)  # 图像高度
    # 创建手势检测实例
    detector = HandDetector(detectionCon=0.65, maxHands=1)

    # 初始化画面(黑色)
    imgCanvas = np.zeros((WIDTH, HEIGHT, 3), np.uint8)

    while True:
        # 读取摄像头画面帧
        success, img = cap.read()
        if not success:
            continue
        # 水平翻转 对于面向人身的摄像头来说方便操作
        img = cv2.flip(img, 1)

        # 绘制识别结果
        img = detector.findHands(img)
        # 获取关键点坐标
        lmList, _, within = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # print(lmList)
            if len(lmList[0]):  # 0号为手腕点
                # tip of index and middle fingers
                x1, y1 = lmList[8][1:]  # 食指指尖序号为 8
                x2, y2 = lmList[12][1:]  # 中指指尖序号为 12
                fingers = detector.fingersUp()
                # print(fingers)
                clean = False
                # 双指向上
                if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:  # 食指和中指向上指
                    xp, yp = 0, 0  # 这块实现模式切换后不会从上次绘画模式的最后一个点开始画
                    if mode == "shape" and current_shape != "polygon" and xr_s != -1:
                        # 形状位置确定 开始画图
                        imgCanvas = draw_shape(imgCanvas, xr_s, yr_s, x1, y1)
                        mode = ""
                        xr_s, yr_s = -1, -1
                    elif mode == "text":
                        if not text_picked and not text_putted:
                            # 单指切双指 此时(直接放置文本)进入文本输入状态
                            # 手写过程与绘画相同 写完之后三指向上 识别 并将结果写入文本框
                            text_putted = True
                            text_enter = 2
                            imgResult = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
                            if x_w == -1 and y_w == -1:
                                x_w, y_w = x1, y1
                            cv2.putText(imgResult, "Enter Here", (x_w, y_w), font, text_size, drawColor,
                                        brushThickness * 2 // 4)

                    elif poly_status > 0:
                        print(poly_status)
                        if len(poly_list) == 1:
                            # 只有一点 标记该点位置
                            cv2.circle(img, poly_list[-1], brushThickness // 2 + brushThickness % 2, drawColor,
                                       cv2.FILLED)
                        if poly_status == 2:
                            poly_status = 1
                            poly_list.append([x1, y1])
                            print(poly_list)
                            if len(poly_list) > 1:
                                # 先画不闭合的
                                cv2.polylines(imgCanvas, [np.array(poly_list, dtype=np.int32)], False, drawColor,
                                              brushThickness)

                    # 区域判断进行工具选择
                    # BGR
                    if y1 < 125:
                        if 50 < x1 < 170:
                            # 画笔模式 工具栏显示颜色
                            mode = "line"
                            header = toolbarList[colorIndex + 1]  # 默认蓝色
                            additional = "color"
                        elif 220 < x1 < 340:
                            # 形状模式 工具栏显示形状
                            mode = "shape"
                            header = toolbarList[5 + shapeIndex]  # 默认矩形
                            additional = "shape"
                        elif 390 < x1 < 510:
                            # 线条粗细选择模式 工具栏显示线条粗细
                            additional = "brush_thickness"
                            header = toolbarList[brushThickness // 10 + 9]  # 默认15
                        elif 560 < x1 < 680:
                            # 文本选择模式 工具栏显示文字大小
                            mode = "text"
                            additional = "text_size"
                            text_picked = True
                            text_putted = False
                            header = toolbarList[text_size + 12]  # 默认2
                        elif 680 < x1 < 1160:
                            # 可变工具栏部分
                            if additional == "color":
                                # 颜色选择
                                if 680 < x1 < 800:
                                    colorIndex = 0  # 蓝色
                                elif 800 < x1 < 920:
                                    colorIndex = 1  # 红色
                                elif 920 < x1 < 1040:
                                    colorIndex = 2  # 绿色
                                elif 1040 < x1 < 1160:
                                    colorIndex = 3  # 黄色
                                # 切换颜色
                                drawColor = Hex_to_RGB(colorList[colorIndex], 'BGR')
                                # 刷新顶部
                                header = toolbarList[colorIndex + 1]
                            elif additional == "shape":
                                # 形状选择
                                mode = "shape"
                                if 680 < x1 < 800:
                                    shapeIndex = 0  # 圆形
                                elif 800 < x1 < 920:
                                    shapeIndex = 1  # 矩形
                                elif 920 < x1 < 1040:
                                    shapeIndex = 2  # 椭圆形
                                elif 1040 < x1 < 1160:
                                    shapeIndex = 3  # 任意多边形
                                    poly_status = 1
                                # 切换形状
                                current_shape = shapeList[shapeIndex]
                                # 刷新顶部
                                header = toolbarList[5 + shapeIndex]
                            elif additional == "brush_thickness":
                                # 粗细选择
                                if 680 < x1 < 800:
                                    brushThickness = 5
                                elif 800 < x1 < 920:
                                    brushThickness = 15
                                elif 920 < x1 < 1040:
                                    brushThickness = 25
                                elif 1040 < x1 < 1160:
                                    brushThickness = 35
                                # 刷新顶部
                                header = toolbarList[brushThickness // 10 + 9]
                            elif additional == "text_size":
                                # 文字大小选择
                                text_picked = True
                                text_putted = False
                                text_enter = 0
                                if 680 < x1 < 800:
                                    text_size = 1
                                elif 800 < x1 < 920:
                                    text_size = 2
                                elif 920 < x1 < 1040:
                                    text_size = 3
                                elif 1040 < x1 < 1160:
                                    text_size = 4
                                # 刷新顶部
                                header = toolbarList[text_size + 12]
                            elif additional == "eraser_thickness":
                                # 橡皮擦粗细选择
                                if 680 < x1 < 800:
                                    eraserThickness = 10
                                elif 800 < x1 < 920:
                                    eraserThickness = 30
                                elif 920 < x1 < 1040:
                                    eraserThickness = 50
                                elif 1040 < x1 < 1160:
                                    eraserThickness = 70
                                # 刷新顶部
                                header = toolbarList[eraserThickness // 20 + 17]
                        elif 1160 < x1:
                            # 橡皮擦部分
                            mode = "eraser"
                            additional = "eraser_thickness"
                            # 刷新顶部
                            header = toolbarList[eraserThickness // 20 + 17]  # 默认70
                            # drawColor = (0, 0, 0)
                    # 食指指尖和中指指尖画个矩形
                    if mode != "":
                        if mode == "eraser":
                            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), eraserColor, cv2.FILLED)
                        else:
                            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

                # 单指向上
                elif fingers[1] and not fingers[2]:
                    # 食指指尖画个圆形
                    if mode != "":
                        if mode == "eraser":
                            cv2.circle(img, (x1, y1), eraserThickness // 2, eraserColor, cv2.FILLED)
                        else:
                            cv2.circle(img, (x1, y1), brushThickness, drawColor, cv2.FILLED)
                    if mode == "shape":
                        if xr_s == -1 and yr_s == -1:
                            xr_s, yr_s = x1, y1
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1  # 这块实现模式切换后重新从当前点开始画图
                    # (xp, yp)是上一帧的坐标 (x1, y1)是当前帧坐标
                    # cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)  # 这块没用 重复了而且不严谨
                    # 橡皮擦
                    if mode == "eraser":
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), eraserColor, eraserThickness)
                    elif mode == "line":
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    elif mode == "shape":
                        # img = draw_shape(img, xr_s, yr_s, x1, y1)
                        if current_shape == "rectangle":
                            cv2.rectangle(img, (xr_s, yr_s + 25), (x1, y1 - 25), drawColor, brushThickness)
                        elif current_shape == "circle":
                            cv2.circle(img, ((xr_s + x1) // 2, (yr_s + y1) // 2),
                                       min((abs(x1 - xr_s) // 2), (abs(y1 - yr_s) // 2)), drawColor,
                                       brushThickness)
                        elif current_shape == "ellipse":
                            cv2.ellipse(img, ((xr_s + x1) // 2, (yr_s + y1) // 2),
                                        ((abs(x1 - xr_s) // 2), (abs(y1 - yr_s) // 4)), 0, 0, 360, drawColor,
                                        brushThickness)
                        elif current_shape == "polygon":
                            if poly_status > 0:
                                # print(poly_status)
                                if poly_status == 1:
                                    poly_status = 2
                                if poly_list:
                                    # 不为空 说明有点 最后一个点画线至当前单指指尖
                                    cv2.line(img, poly_list[-1], (x1, y1), drawColor, brushThickness)
                    elif mode == "text":
                        # 文本单指
                        if not text_putted:
                            # 文本还未确定 随着食指指尖变换位置
                            text_picked = False
                            text_enter = 1
                            cv2.putText(img, "Enter Here", (x1, y1), font, text_size, drawColor,
                                        brushThickness * 2 // 4)
                        elif text_enter > 0:
                            text_rec = False
                            if text_enter == 2:
                                imgWriting = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
                                text_enter = 1
                            cv2.line(imgWriting, (xp, yp), (x1, y1), drawColor, brushThickness)
                    xp, yp = x1, y1
                elif fingers[1] and fingers[2] and fingers[4] and not fingers[3]:  # 食指、中指、小拇指向上指
                    if poly_status > 0:  # 可以不判断mode 因为poly_list为空
                        cv2.polylines(imgCanvas, [np.array(poly_list, dtype=np.int32)], True, drawColor, brushThickness)
                        poly_list = []
                        poly_status = 0
                        mode = ""
                    elif text_enter > 0 and not text_rec and mode == "text":
                        # 处理imgResult
                        cv2.imwrite("Temp\\1.jpg", imgWriting)
                        try:
                            text += getText("Temp\\1.jpg")
                            imgResult = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
                            cv2.putText(imgResult, text, (x_w, y_w), font, text_size, drawColor,
                                        brushThickness * 2 // 4)
                        except:  # 识别不出
                            imgResult = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
                            header = toolbarList[21]
                        imgWriting = np.zeros((WIDTH, HEIGHT, 3), np.uint8)  # 清楚所画文字
                        text_rec = True  # 已识别完成

                elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:  # 食指、中指、无名指向上指
                    if text_enter == 1 and mode == "text":
                        text_enter = 0
                        # 文字识别结果画布和Canvas画布结合
                        imgResult = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
                        text = ""
                        x_w, y_w = -1, -1
                        imgWriting = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
                        mode = ""
                # 五指均在画面内且均向上时清除画板
                elif within and sum(fingers) == 5:
                    print("clean")
                    # print(len(lmList))
                    # 重置设置
                    imgCanvas = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
                    imgResult = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
                    imgWriting = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
                    clean = True
                    mode = ""
                    text = ""
        if not clean:
            if text_enter == 2:
                # 画在img上 随时变化
                img = imgMerge(imgResult, img)
                imgResult = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
            else:
                # 画在imgCanvas 不变
                imgCanvas = imgMerge(imgResult, imgCanvas)
                imgResult = np.zeros((WIDTH, HEIGHT, 3), np.uint8)

            # imgCanvas是一块黑色背景的画板，所有的绘画操作实际上是在imgCanvas上画的，橡皮擦擦除的原理是黑色覆盖掉其他颜色
            # 最终要合并到img上
            img = imgMerge(imgCanvas, img)
            # 手写画布合并
            img = imgMerge(imgWriting, img)

        # 放置工具栏
        img[0:HEADER_HEIGHT, 0:HEIGHT] = header
        # cv2.imshow("Canvas", imgCanvas)
        # cv2.imshow("Inv", imgInv)
        cv2.imshow("Image", img)
        # 按esc退出
        if cv2.waitKey(5) & 0xFF == 27:
            cap.release()
            return render_template("thanks.html")
    cap.release()
