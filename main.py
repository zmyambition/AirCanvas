"""
文件名: main.py
描述: 基于手势识别的虚拟画布，支持多种画笔和图层操作。
"""

import cv2
import numpy as np
import time
import os
import math
from datetime import datetime
from modules.hand_detector import HandDetector

# 绘图与UI辅助工具

def create_color_header(width, height=100):
    """生成顶部调色盘和橡皮擦区域"""
    spectrum_width = width - 150
    hue_values = np.linspace(0, 179, spectrum_width, dtype=np.uint8)
    hsv_strip = np.zeros((1, spectrum_width, 3), np.uint8)
    hsv_strip[0, :, 0] = hue_values
    hsv_strip[0, :, 1] = 255
    hsv_strip[0, :, 2] = 255
    bgr_strip = cv2.cvtColor(hsv_strip, cv2.COLOR_HSV2BGR)
    header_img = cv2.resize(bgr_strip, (spectrum_width, height))

    # 右侧留出橡皮擦占位
    eraser_img = np.zeros((height, 150, 3), np.uint8)
    eraser_img[:] = (40, 40, 40)
    cv2.putText(eraser_img, "ERASER", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return np.hstack((header_img, eraser_img))


def draw_advanced_line(canvas, start_pt, end_pt, color, tool_type, scale=1.0):
    """处理不同笔触的绘制逻辑"""
    base_thick = {
        "Pencil": 2, "Pen": 8, "Brush": 25, "Eraser": 50, "Neon": 4
    }
    t = int(base_thick.get(tool_type, 5) * scale)
    t = max(1, t)

    if tool_type in ["Pencil", "Pen", "Eraser"]:
        cv2.line(canvas, start_pt, end_pt, color, t)
    elif tool_type == "Brush":
        # 笔刷带半透明效果
        alpha = 0.4
        overlay = canvas.copy()
        cv2.line(overlay, start_pt, end_pt, color, t)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    elif tool_type == "Neon":
        # 霓虹灯笔触：多层叠加形成光晕
        halo_outer = int(40 * scale)
        halo_inner = int(20 * scale)
        overlay = canvas.copy()
        cv2.line(overlay, start_pt, end_pt, color, halo_outer)
        cv2.addWeighted(overlay, 0.1, canvas, 0.9, 0, canvas)
        overlay = canvas.copy()
        cv2.line(overlay, start_pt, end_pt, color, halo_inner)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
        cv2.line(canvas, start_pt, end_pt, color, t)


def main():
    # --- 画布基础设置 ---
    CANVAS_WIDTH = 1280
    CANVAS_HEIGHT = 720

    current_tool = "Pen"
    draw_color = (255, 0, 0)

    is_dark_mode = False
    canvas_bg_color = (255, 255, 255)

    SMOOTHENING = 5  # 坐标平滑系数
    thickness_scale = 1.0

    # 限制绘画区域 (ROI)
    ROI_X1, ROI_Y1 = 250, 150
    ROI_X2, ROI_Y2 = 1100, 650

    folder_path = "output"
    if not os.path.exists(folder_path): os.makedirs(folder_path)

    # 手势触发计时器 (防止误触)
    clear_start_time = 0
    clear_duration = 1.5
    save_start_time = 0
    save_duration = 2.0
    undo_start_time = 0
    undo_duration = 1.5
    theme_start_time = 0
    theme_duration = 1.0

    # 撤销历史栈
    undo_stack = []
    undo_stack_video = []
    max_undo_steps = 10
    drawing_active = False

    # 屏幕反馈提示计时
    save_feedback_timer = 0
    undo_feedback_timer = 0
    theme_feedback_timer = 0

    # --- 初始化硬件和对象 ---
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(max_hands=1, detection_con=0.8)
    img_whiteboard = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), np.uint8) * 255
    canvas_video = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), np.uint8)  # 叠加在视频上的透明层

    header_h = 100
    ui_header = None
    xp, yp = 0, 0  # 上一帧绘图坐标
    plocX, plocY = 0, 0  # 平滑处理用的位置
    clocX, clocY = 0, 0

    window_name = "Air Canvas Controller"
    # 启用 WINDOW_KEEPRATIO 防止窗口拉伸导致坐标偏移
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Pure Canvas", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    print("系统已启动。控制台退出请按 'q'。")

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)  # 镜像处理
        hCam, wCam, _ = img.shape

        if ui_header is None:
            ui_header = create_color_header(wCam, header_h)

        # 渲染顶部 UI
        img[0:header_h, 0:wCam] = ui_header

        # 侧边工具栏按钮定义
        btn_w = 120
        start_y = 120
        btn_h = 70
        gap = 10
        btn_regions = {
            "Pen": (0, start_y, btn_w, start_y + btn_h),
            "Brush": (0, start_y + (btn_h + gap), btn_w, start_y + (btn_h + gap) + btn_h),
            "Pencil": (0, start_y + (btn_h + gap) * 2, btn_w, start_y + (btn_h + gap) * 2 + btn_h),
            "Neon": (0, start_y + (btn_h + gap) * 3, btn_w, start_y + (btn_h + gap) * 3 + btn_h)
        }

        # 绘制按钮状态
        for name, region in btn_regions.items():
            if current_tool == name:
                cv2.rectangle(img, (region[0], region[1]), (region[0] + 10, region[3]), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(img, (region[0] + 10, region[1]), (region[2], region[3]), (80, 80, 80), cv2.FILLED)
                txt_color = (0, 255, 0)
            else:
                cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (50, 50, 50), cv2.FILLED)
                txt_color = (200, 200, 200)
            cv2.putText(img, name, (region[0] + 25, region[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)

        # 粗细调节滑动条
        slider_x, slider_y = 30, 460
        slider_w, slider_h = 40, 150
        cv2.rectangle(img, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (50, 50, 50), cv2.FILLED)
        cv2.putText(img, "Size", (slider_x, slider_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        ratio = (thickness_scale - 0.5) / (1.5 - 0.5)
        knob_y = int((slider_y + slider_h) - (ratio * slider_h))
        cv2.circle(img, (slider_x + slider_w // 2, knob_y), 15, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, f"{thickness_scale:.1f}x", (slider_x + slider_w + 5, knob_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)

        # 手势逻辑处理
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1], lm_list[8][2]  # 食指
            x2, y2 = lm_list[12][1], lm_list[12][2]  # 中指
            x0, y0 = lm_list[4][1], lm_list[4][2]  # 拇指
            fingers = detector.fingers_up()

            # 坐标映射与平滑
            mapped_x = np.interp(x1, [ROI_X1, ROI_X2], [0, CANVAS_WIDTH])
            mapped_y = np.interp(y1, [ROI_Y1, ROI_Y2], [0, CANVAS_HEIGHT])
            clocX = plocX + (mapped_x - plocX) / SMOOTHENING
            clocY = plocY + (mapped_y - plocY) / SMOOTHENING

            # 手势功能判定 (长按逻辑)

            # 1. 切换画布颜色 (三指)
            is_three_fingers = (fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0)
            if is_three_fingers:
                drawing_active = False
                if theme_start_time == 0: theme_start_time = time.time()
                elapsed_theme = time.time() - theme_start_time
                cx, cy = lm_list[9][1], lm_list[9][2]
                cv2.ellipse(img, (cx, cy), (40, 40), 0, 0, int((elapsed_theme / theme_duration) * 360), (255, 0, 255),
                            8)
                if elapsed_theme > theme_duration:
                    is_dark_mode = not is_dark_mode
                    canvas_bg_color = (0, 0, 0) if is_dark_mode else (255, 255, 255)
                    img_whiteboard[:] = canvas_bg_color
                    canvas_video[:] = 0
                    undo_stack, undo_stack_video = [], []
                    if current_tool == "Eraser": draw_color = canvas_bg_color
                    theme_feedback_timer, theme_start_time = 20, 0
            else:
                theme_start_time = 0

            # 2. 清屏 (全掌)
            is_five_fingers = (fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1)
            if is_five_fingers and theme_start_time == 0:
                drawing_active = False
                if clear_start_time == 0: clear_start_time = time.time()
                elapsed = time.time() - clear_start_time
                cv2.ellipse(img, (wCam // 2, hCam // 2), (60, 60), 0, 0, int((elapsed / clear_duration) * 360),
                            (0, 0, 255), 10)
                if elapsed > clear_duration:
                    img_whiteboard[:] = canvas_bg_color
                    canvas_video[:] = 0
                    undo_stack, undo_stack_video = [], []
                    clear_start_time = 0
            else:
                clear_start_time = 0

            # 3. 撤销 (握拳)
            is_fist = (fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0)
            if is_fist and clear_start_time == 0 and theme_start_time == 0:
                drawing_active = False
                if undo_start_time == 0: undo_start_time = time.time()
                elapsed_undo = time.time() - undo_start_time
                cx, cy = lm_list[9][1], lm_list[9][2]
                cv2.ellipse(img, (cx, cy), (40, 40), 0, 0, int((elapsed_undo / undo_duration) * 360), (0, 255, 255), 8)
                if elapsed_undo > undo_duration:
                    if undo_stack:
                        img_whiteboard = undo_stack.pop()
                        canvas_video = undo_stack_video.pop()
                        undo_feedback_timer = 20
                    undo_start_time = 0
            else:
                undo_start_time = 0

            # 4. 保存 (捏合手势 + 剩余手指张开 “OK”)
            pinch = math.hypot(x1 - x0, y1 - y0)
            is_save_gesture = (pinch < 40) and (fingers[2] and fingers[3] and fingers[4])
            if is_save_gesture and not is_fist and clear_start_time == 0:
                drawing_active = False
                if save_start_time == 0: save_start_time = time.time()
                elapsed_save = time.time() - save_start_time
                cv2.ellipse(img, (x1, y1), (35, 35), 0, 0, int((elapsed_save / save_duration) * 360), (0, 255, 0), 5)
                if elapsed_save > save_duration:
                    cv2.imwrite(f"{folder_path}/Art_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", img_whiteboard)
                    save_feedback_timer, save_start_time = 30, 0
            else:
                save_start_time = 0

            # 模式切换：选择 vs 绘画

            # 无特殊功能触发时，处理正常交互
            if all(t == 0 for t in [clear_start_time, undo_start_time, save_start_time, theme_start_time]):

                # 选择模式 (食指中指并拢)
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                    drawing_active = False
                    xp, yp = 0, 0
                    cv2.circle(img, (x1, y1), 20, (255, 255, 255), 2)

                    # 检查是否点击 UI 区域
                    if y1 < header_h:
                        if x1 > (wCam - 150):  # 橡皮擦区
                            current_tool = "Eraser"
                            draw_color = canvas_bg_color
                        else:  # 颜色拾取
                            if current_tool == "Eraser": current_tool = "Pen"
                            px = min(x1, wCam - 151)
                            pc = ui_header[min(y1, header_h - 1), px]
                            draw_color = (int(pc[0]), int(pc[1]), int(pc[2]))
                            cv2.circle(img, (x1, y1), 25, draw_color, cv2.FILLED)

                    # 侧边栏按钮点击
                    elif x1 < btn_w and y1 < slider_y:
                        for name, region in btn_regions.items():
                            if region[1] < y1 < region[3]:
                                current_tool = name
                                if current_tool != "Eraser" and draw_color == canvas_bg_color:
                                    draw_color = (0, 0, 255) if not is_dark_mode else (0, 255, 255)

                    # 粗细滑动条调节
                    elif slider_x - 20 < x1 < slider_x + slider_w + 20 and slider_y < y1 < slider_y + slider_h:
                        slider_val = np.clip((slider_y + slider_h - y1) / slider_h, 0.0, 1.0)
                        thickness_scale = round(0.5 + slider_val, 1)
                        cv2.circle(img, (x1, y1), 20, (0, 255, 255), 2)

                # 绘画模式 (仅食指)
                elif fingers[1] == 1 and fingers[2] == 0:
                    if not drawing_active:
                        # 刚开始下笔，保存当前快照到撤销栈
                        undo_stack.append(img_whiteboard.copy())
                        undo_stack_video.append(canvas_video.copy())
                        if len(undo_stack) > max_undo_steps:
                            undo_stack.pop(0)
                            undo_stack_video.pop(0)
                        drawing_active = True

                    if ROI_X1 < x1 < ROI_X2 and ROI_Y1 < y1 < ROI_Y2:
                        cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                        if xp == 0 and yp == 0:
                            xp, yp = int(clocX), int(clocY)
                            vxp, vyp = x1, y1
                        else:
                            vxp, vyp = int(plocX_raw), int(plocY_raw)

                        # 在两个画布上同步绘制
                        active_color = canvas_bg_color if current_tool == "Eraser" else draw_color
                        draw_advanced_line(img_whiteboard, (xp, yp), (int(clocX), int(clocY)),
                                           active_color, current_tool, thickness_scale)

                        video_color = (0, 0, 0) if current_tool == "Eraser" else draw_color
                        draw_advanced_line(canvas_video, (vxp, vyp), (x1, y1),
                                           video_color, current_tool, thickness_scale)

                        xp, yp = int(clocX), int(clocY)
                    else:
                        xp, yp = 0, 0

            plocX, plocY = clocX, clocY
            plocX_raw, plocY_raw = x1, y1

        # 图像合成与最终显示
        # 将绘画层(canvas_video)叠加到相机层(img)
        img_gray = cv2.cvtColor(canvas_video, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY_INV)
        img = cv2.bitwise_and(img, img, mask=img_inv)
        img = cv2.add(img, canvas_video)

        # 渲染操作反馈文字
        def show_msg(text, color):
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 1.2, 3)
            cv2.putText(img, text, ((wCam - tw) // 2, (hCam + th) // 2), font, 1.2, color, 3)

        if save_feedback_timer > 0:
            show_msg("PICTURE SAVED", (0, 255, 0))
            save_feedback_timer -= 1
        if undo_feedback_timer > 0:
            show_msg("UNDO SUCCESS", (0, 255, 255))
            undo_feedback_timer -= 1
        if theme_feedback_timer > 0:
            show_msg("THEME SWITCHED", (255, 0, 255))
            theme_feedback_timer -= 1

        # 状态指示器
        cv2.rectangle(img, (wCam - 120, hCam - 100), (wCam, hCam), draw_color, cv2.FILLED)
        cv2.putText(img, current_tool, (wCam - 110, hCam - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        mode_txt = "Dark" if is_dark_mode else "Light"
        cv2.putText(img, f"{mode_txt} Mode", (wCam - 110, hCam - 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                    1)

        # 绘制 ROI 边界线
        cv2.rectangle(img, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 255), 2)
        cv2.putText(img, "Drawing Zone", (ROI_X1, ROI_Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        cv2.imshow(window_name, img)
        cv2.imshow("Pure Canvas", img_whiteboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()