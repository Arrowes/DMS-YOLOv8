import torch
import numpy as np
from collections import namedtuple
import math
import cv2

MarkingPoint = namedtuple('MarkingPoint', ['x',
                                           'y',
                                           'lenSepLine_x',
                                           'lenSepLine_y',
                                           'lenEntryLine_x',
                                           'lenEntryLine_y',
                                           'isOccupied'])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deploy_preprocess(output):
    ''' snpe 对sigmoid tanh 这些带exp指数的运算优化还没到位 '''
        # 放到模型外面（后）处理
        # Confidence point_x, point_y
   # output = np.array(output)  # 将output转换为NumPy数组
    output[:, :3] = sigmoid(output[:, :3])
    output[:, 3:7] = np.tanh(output[:, 3:7])
    output[:, 7] = sigmoid(output[:, 7])
    return output


def get_predicted_points(prediction):
    """从一个预测的特征图中获取标记点。"""
    # 确保传入的prediction是torch.Tensor类型
    prediction = torch.from_numpy(prediction)
    assert isinstance(prediction, torch.Tensor)
    # 存储预测的标记点
    predicted_points = []
    # 将prediction从GPU中分离并转换为NumPy数组
    prediction = prediction.detach().cpu().numpy()
    # 获取特征图的通道数(C)、高度(feature_H)和宽度(feature_W)
    C, feature_H, feature_W = prediction.shape[-3:]
    # 确保通道数为8（这里假设通道顺序为[置信度, obj_x, obj_y, lenSepLine_x, lenSepLine_y, lenEntryLine_x, lenEntryLine_y, isOccupied]）
    assert C == 8
    # 阈值，用于过滤预测的标记点
    thresh = 0.01
    # 遍历特征图的每个像素点
    for i in range(feature_H):
        for j in range(feature_W):
            # 如果该像素点的置信度大于等于阈值，将其视为有效的标记点
            if prediction[0, i, j] >= thresh:
                # 计算标记点在原图中的x坐标（obj_x）
                obj_x = (j + prediction[1, i, j]) / feature_W
                # 计算标记点在原图中的y坐标（obj_y）
                obj_y = (i + prediction[2, i, j]) / feature_H
                # 获取分隔线的长度在x方向上的预测值（lenSepLine_x）
                lenSepLine_x = prediction[3, i, j]
                # 获取分隔线的长度在y方向上的预测值（lenSepLine_y）
                lenSepLine_y = prediction[4, i, j]
                # 获取入口线的长度在x方向上的预测值（lenEntryLine_x）
                lenEntryLine_x = prediction[5, i, j]
                # 获取入口线的长度在y方向上的预测值（lenEntryLine_y）
                lenEntryLine_y = prediction[6, i, j]
                # 获取是否被占用的预测值（isOccupied）
                isOccupied = prediction[7, i, j]
                # 创建MarkingPoint对象，存储标记点的信息
                marking_point = MarkingPoint(obj_x, obj_y,
                                             lenSepLine_x, lenSepLine_y,
                                             lenEntryLine_x, lenEntryLine_y,
                                             isOccupied)
                # 将置信度和标记点信息添加到predicted_points列表中
                predicted_points.append((prediction[0, i, j], marking_point))

    # 对预测的标记点进行非极大值抑制，根据params中的规则来选择最佳的标记点
    return non_maximum_suppression(predicted_points)


def non_maximum_suppression(pred_points):
    """Perform non-maxmum suppression on marking points."""
    ''' 间隔车位可检测 '''
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            # 0是置信度，1是marking_point
            i_x = pred_points[i][1].x
            i_y = pred_points[i][1].y
            j_x = pred_points[j][1].x
            j_y = pred_points[j][1].y
            if abs(j_x - i_x) < 1 / 4 and abs(
                    j_y - i_y) < 1 / 12:
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True
    if any(suppressed):
        unsupres_pred_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                unsupres_pred_points.append(pred_points[i])
        return unsupres_pred_points
    return pred_points


    """
    画进入线目标点：逆时针旋转车位的进入线起始端点。
    AB-BC-CD-DA 这里指A点
       Parking Slot Example
            A.____________D
             |           |
           ==>           |
            B____________|C

     Entry_line: AB
     Separable_line: AD (regressive)
     Separable_line: BC (un-regressive, calc)
     Object_point: A (point_0)
      cos sin theta 依据的笛卡尔坐标四象限
               -y
                 |
             3  |  4
     -x -----|-----> +x (w)
             2  |  1
                ↓
               +y (h)
    """
def plot_slots(image, eval_results, img_name=None):
    pred_points =  eval_results['pred_points'] \
        if 'pred_points' in eval_results else eval_results
    height = image.shape[0]
    width = image.shape[1]
    for confidence, marking_point in pred_points:
        if confidence < 0.1:
            continue # 如果置信度太低，跳过此点的绘制
        x, y, lenSepLine_x, lenSepLine_y, \
        lenEntryLine_x, lenEntryLine_y, available = marking_point
        # p0->p1为进入线entry_line
        # p0->p3为分隔线separable_line
        # p1->p3也为分割线
        # 上述箭头"->"代表向量方向，p0->p1即p0为起点，p3为终点，p0指向p3
        p0_x = width * x
        p0_y = height * y
        p1_x = p0_x + width * lenEntryLine_x
        p1_y = p0_y + height * lenEntryLine_y
        length = 300

        H, W =[height,width]
        x_ratio, y_ratio = 1, H / W
        radian = math.atan2(marking_point.lenSepLine_y * y_ratio,
                            marking_point.lenSepLine_x * x_ratio)
        sep_cos = math.cos(radian)
        sep_sin = math.sin(radian)
        p3_x = int(p0_x + width * lenSepLine_x)
        p3_y = int(p0_y + height * lenSepLine_y)
        p2_x = int(p1_x +  width * lenSepLine_x)
        p2_y = int(p1_y + height * lenSepLine_y)

        p0_x, p0_y = round(p0_x), round(p0_y)   #四舍五入
        p1_x, p1_y = round(p1_x), round(p1_y)
        # 画进入线目标点：逆时针旋转车位的进入线起始端点。
        # AB-BC-CD-DA 这里指A点
        cv2.circle(image, (p0_x, p0_y), 5, (0, 0, 255), thickness=2)
        # 给目标点打上置信度，取值范围：0到1
        color = (255, 255, 255) if confidence > 0.7 else (100, 100, 255)
        if confidence < 0.3: color = (0, 0, 255)
        cv2.putText(image, f'{confidence:.3f}', # 不要四舍五入
                    (p0_x + 6, p0_y - 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, color)
        # 画上目标点坐标 (x, y)
        cv2.putText(image, f' ({p0_x},{p0_y})',
                    (p0_x, p0_y + 15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # 在图像左上角给出图像的分辨率大小 (W, H)
        H, W = [256, 128]
        cv2.putText(image, f'({W},{H})', (5, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # 画车位掩码图，分三步
        # 第一步：画未被占概率，画是否可用提示
        # avail = 'N' if isOccupied > 0.9 else 'Y'
        # cv2.putText(image, avail,
        #             ((p0_x + p1_x) // 2 + 10, (p0_y + p1_y) // 2 + 10),
        #             cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        area = np.array([[p0_x, p0_y], [p1_x, p1_y],
                        [p2_x, p2_y], [p3_x, p3_y]])
        if available > 0.7:
            color = (0, 255, 0)  # 置信度颜色
            cv2.putText(image,
                        f'{available:.3f}',  # 转换为"未被占用"置信度
                        ((p0_x + p2_x) // 2, (p0_y + p2_y) // 2 + 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color)
        # 第二步：判断，如果可用，才画掩码图，否则全零
        mask = np.zeros_like(image)
        if available > 0.7:
            cv2.fillPoly(mask, [area], [0, 64, 0])
        # 第三步：把掩码图和原图进行叠加： output= a*i1+b*i2+c
        image = cv2.addWeighted(image, 1., mask, 0.5, 0) # 数字是权重

        # 画车位进入线 AB
        cv2.arrowedLine(image, (p0_x, p0_y), (p1_x, p1_y), (0, 255, 0), 2, 8, 0, 0.2)
        # 画车位分割线 AD
        cv2.arrowedLine(image, (p0_x, p0_y), (p3_x, p3_y), (255, 0, 0), 2) # cv2.line
        # 画车位分割线 BC
        if p1_x >= 0 and p1_x <= width - 1 and p1_y >= 0 and p1_y <= height - 1:
            cv2.arrowedLine(image, (p1_x, p1_y), (p2_x, p2_y), (33, 164, 255), 2)
        print("Line---- AB:",(p0_x, p0_y), "->", (p1_x, p1_y), "AD:", (p0_x, p0_y), "->",  (p3_x, p3_y))
    return image