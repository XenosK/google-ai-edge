import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from pathlib import Path


class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化 YOLO 检测器
        
        Args:
            model_path: TFLite 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值（用于 NMS）
        """
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 获取输入输出详情
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 获取输入尺寸
        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        
        # 打印模型信息
        print(f"模型输入尺寸: {self.input_width}x{self.input_height}")
        print(f"模型输出详情:")
        for i, output_detail in enumerate(self.output_details):
            print(f"  输出 {i}: 形状 {output_detail['shape']}, 类型 {output_detail['dtype']}")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # COCO 类别名称（YOLO11 使用 COCO 数据集）
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理图像帧
        
        Args:
            frame: 输入帧 (BGR格式)
            
        Returns:
            预处理后的图像数组
        """
        # 转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        frame_resized = cv2.resize(frame_rgb, (self.input_width, self.input_height))
        
        # 归一化到 [0, 1]
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # 添加批次维度
        input_data = np.expand_dims(frame_normalized, axis=0)
        
        return input_data
    
    def postprocess(self, outputs: np.ndarray, original_shape: tuple) -> list:
        """
        后处理模型输出
        
        输出格式: [1, 84, 8400]
        - 84 = 4 (bbox: x_center, y_center, width, height) + 80 (class scores)
        - 8400 = 检测框数量
        - 坐标是归一化的 (0-1)，相对于输入尺寸 (640x640)
        
        Args:
            outputs: 模型输出 [1, 84, 8400]
            original_shape: 原始图像尺寸 (height, width)
            
        Returns:
            检测结果列表 [(x1, y1, x2, y2, conf, class_id), ...]
        """
        detections = []
        orig_h, orig_w = original_shape
        
        # 输出格式: [1, 84, 8400]
        # 转换为: [8400, 84] - 每个检测框有84个特征
        if len(outputs.shape) == 3:
            # 移除批次维度并转置: [84, 8400] -> [8400, 84]
            outputs = outputs[0].transpose(1, 0)
        elif len(outputs.shape) == 2:
            # 如果已经是 [84, 8400]，转置为 [8400, 84]
            if outputs.shape[0] == 84:
                outputs = outputs.transpose(1, 0)
        
        # 计算缩放比例
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height
        
        # 处理每个检测框
        for detection in outputs:
            # 格式: [x_center, y_center, width, height, class_scores...]
            # 坐标是归一化的 (0-1)，相对于输入尺寸 (640x640)
            x_center_norm, y_center_norm, width_norm, height_norm = detection[:4]
            class_scores = detection[4:]
            
            # 找到最高分的类别
            class_id = np.argmax(class_scores)
            conf = float(class_scores[class_id])
            
            if conf < self.conf_threshold:
                continue
            
            # 将归一化坐标转换为相对于640x640的像素坐标
            x_center = x_center_norm * self.input_width
            y_center = y_center_norm * self.input_height
            width = width_norm * self.input_width
            height = height_norm * self.input_height
            
            # 缩放到原始图像尺寸
            x_center_scaled = x_center * scale_x
            y_center_scaled = y_center * scale_y
            width_scaled = width * scale_x
            height_scaled = height * scale_y
            
            # 转换为左上角和右下角坐标
            x1 = int(x_center_scaled - width_scaled / 2)
            y1 = int(y_center_scaled - height_scaled / 2)
            x2 = int(x_center_scaled + width_scaled / 2)
            y2 = int(y_center_scaled + height_scaled / 2)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            # 确保边界框有效
            if x2 > x1 and y2 > y1:
                detections.append((x1, y1, x2, y2, conf, class_id))
        
        # 应用 NMS
        detections = self.nms(detections)
        
        return detections
    
    def nms(self, detections: list) -> list:
        """
        非极大值抑制
        
        Args:
            detections: 检测结果列表
            
        Returns:
            过滤后的检测结果
        """
        if len(detections) == 0:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        keep = []
        while detections:
            # 保留置信度最高的
            current = detections.pop(0)
            keep.append(current)
            
            # 移除与当前框 IoU 过高的框
            detections = [
                det for det in detections
                if self.calculate_iou(current, det) < self.iou_threshold
            ]
        
        return keep
    
    def calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """
        计算两个边界框的 IoU
        
        Args:
            box1: (x1, y1, x2, y2, conf, class_id)
            box2: (x1, y1, x2, y2, conf, class_id)
            
        Returns:
            IoU 值
        """
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def detect(self, frame: np.ndarray) -> list:
        """
        对单帧进行检测
        
        Args:
            frame: 输入帧
            
        Returns:
            检测结果列表
        """
        original_shape = frame.shape[:2]
        
        # 预处理
        input_data = self.preprocess(frame)
        
        # 设置输入
        input_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_index, input_data)
        
        # 运行推理
        self.interpreter.invoke()
        
        # 获取输出
        output_index = self.output_details[0]['index']
        outputs = self.interpreter.get_tensor(output_index)
        
        # 后处理
        detections = self.postprocess(outputs, original_shape)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        在帧上绘制检测结果
        
        Args:
            frame: 输入帧
            detections: 检测结果列表
            
        Returns:
            绘制后的帧
        """
        for x1, y1, x2, y2, conf, class_id in detections:
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class {class_id}"
            
            label = f"{class_name}: {conf:.2f}"
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制标签背景
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return frame


def process_video(input_path: str, output_path: str, detector: YOLODetector):
    """
    处理视频文件
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        detector: YOLO 检测器实例
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {input_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"处理视频: {input_path}")
    print(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 进行检测
        detections = detector.detect(frame)
        
        # 绘制检测结果
        frame_with_detections = detector.draw_detections(frame.copy(), detections)
        
        # 写入输出视频
        out.write(frame_with_detections)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"进度: {frame_count}/{total_frames} ({progress:.1f}%) - 检测到 {len(detections)} 个对象")
    
    cap.release()
    out.release()
    
    print(f"完成! 输出视频已保存到: {output_path}")


def main():
    """主函数"""
    # 配置路径
    media_dir = Path("media")
    output_dir = Path("output")
    model_path = "models/yolo11n_float16.tflite"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    # 初始化检测器
    print("正在加载 YOLO 模型...")
    detector = YOLODetector(model_path, conf_threshold=0.25, iou_threshold=0.45)
    print("模型加载完成!")
    
    # 获取所有视频文件
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = [
        f for f in media_dir.iterdir()
        if f.suffix.lower() in video_extensions and f.is_file()
    ]
    
    if not video_files:
        print(f"错误: 在 {media_dir} 目录下未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    for video_file in video_files:
        input_path = str(video_file)
        output_path = str(output_dir / f"{video_file.stem}_detected{video_file.suffix}")
        
        process_video(input_path, output_path, detector)
        print()
    
    print("所有视频处理完成!")


if __name__ == "__main__":
    main()
