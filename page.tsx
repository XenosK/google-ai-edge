'use client';

import Link from "next/link";
import { useEffect, useState, useRef } from "react";
import {runWithTfjsTensors} from '@litertjs/tfjs-interop';
import {loadAndCompile, loadLiteRt, getWebGpuDevice, CompiledModel} from '@litertjs/core';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import {WebGPUBackend} from '@tensorflow/tfjs-backend-webgpu';

// COCO 类别名称（YOLO11 使用 COCO 数据集）
const COCO_CLASSES = [
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
];

interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  classId: number;
}

export default function YOLOPage() {
  const [model, setModel] = useState<CompiledModel | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [originalImage, setOriginalImage] = useState<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const confThreshold = 0.25;
  const iouThreshold = 0.45;
  const inputWidth = 640;
  const inputHeight = 640;

  // 初始化模型
  useEffect(() => {
    async function init() {
      console.log('正在加载 YOLO 模型...');
      
      try {
      await loadLiteRt('/wasm/');
      console.log('LiteRT.js 初始化成功');
      } catch (error: any) {
        // 如果 LiteRT 已经加载，忽略错误
        if (error?.message?.includes('already loading') || error?.message?.includes('already loaded')) {
          console.log('LiteRT.js 已经加载，跳过初始化');
        } else {
          console.error('LiteRT.js 初始化失败:', error);
          throw error;
        }
      }
      
      const device = await getWebGpuDevice();
      if (!device) {
        console.error('未找到 WebGPU 设备');
        return;
      }
      
      const adapterInfo = (device as any).adapterInfo;
      tf.removeBackend('webgpu');
      tf.registerBackend('webgpu', () => new WebGPUBackend(device, adapterInfo));
      await tf.setBackend('webgpu');
      await tf.ready();
      console.log('TensorFlow.js WebGPU 后端初始化成功');
      
      const loadedModel = await loadAndCompile('/models/yolo11n_float32.tflite', { 
        accelerator: 'webgpu' 
      });
      console.log('模型加载完成!');
      
      setModel(loadedModel);
      setIsLoading(false);
    }
    
    init();
  }, []);

  // 计算 IoU
  function calculateIoU(box1: Detection, box2: Detection): number {
    const x1_i = Math.max(box1.x1, box2.x1);
    const y1_i = Math.max(box1.y1, box2.y1);
    const x2_i = Math.min(box1.x2, box2.x2);
    const y2_i = Math.min(box1.y2, box2.y2);

    if (x2_i < x1_i || y2_i < y1_i) return 0.0;

    const intersection = (x2_i - x1_i) * (y2_i - y1_i);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;

    return union === 0 ? 0.0 : intersection / union;
  }

  // NMS (非极大值抑制)
  function nms(detections: Detection[]): Detection[] {
    if (detections.length === 0) return [];

    const sorted = [...detections].sort((a, b) => b.conf - a.conf);
    const keep: Detection[] = [];

    while (sorted.length > 0) {
      const current = sorted.shift()!;
      keep.push(current);
      
      for (let i = sorted.length - 1; i >= 0; i--) {
        if (calculateIoU(current, sorted[i]) > iouThreshold) {
          sorted.splice(i, 1);
        }
      }
    }

    return keep;
  }

  // 后处理 - 参考 Python 代码的 postprocess 方法
  function postprocess(outputArray: any, originalWidth: number, originalHeight: number): Detection[] {
    let detections: Detection[] = [];
    let boxes: number[][];

    // 输出格式: [1, 84, 8400] 或 [84, 8400]
    // 需要转置为 [8400, 84] - 每个检测框有84个特征
    if (Array.isArray(outputArray)) {
      if (outputArray.length === 1 && Array.isArray(outputArray[0])) {
        // 格式: [1, 84, 8400] - 移除批次维度
        const data = outputArray[0];
        if (Array.isArray(data) && data.length === 84 && Array.isArray(data[0])) {
          // 转置: [84, 8400] -> [8400, 84]
          const numBoxes = data[0].length;
          boxes = [];
          for (let i = 0; i < numBoxes; i++) {
            boxes.push(data.map(row => row[i]));
          }
        } else {
          boxes = data as number[][];
        }
      } else if (outputArray.length === 84 && Array.isArray(outputArray[0])) {
        // 格式: [84, 8400] - 直接转置
        const numBoxes = outputArray[0].length;
        boxes = [];
        for (let i = 0; i < numBoxes; i++) {
          boxes.push(outputArray.map(row => row[i]));
        }
      } else {
        boxes = outputArray as number[][];
      }
    } else {
      console.error('意外的输出格式');
      return [];
    }

    // 计算缩放比例
    const scaleX = originalWidth / inputWidth;
    const scaleY = originalHeight / inputHeight;

    // 处理每个检测框
    for (const detection of boxes) {
      // 格式: [x_center_norm, y_center_norm, width_norm, height_norm, class_scores...]
      // 坐标是归一化的 (0-1)，相对于输入尺寸 (640x640)
      const xCenterNorm = detection[0];
      const yCenterNorm = detection[1];
      const widthNorm = detection[2];
      const heightNorm = detection[3];
      const classScores = detection.slice(4, 84);

      // 找到最高分的类别
      let maxScore = -1;
      let classId = -1;
      for (let i = 0; i < classScores.length; i++) {
        if (classScores[i] > maxScore) {
          maxScore = classScores[i];
          classId = i;
        }
      }

      if (maxScore < confThreshold) continue;

      // 将归一化坐标转换为相对于640x640的像素坐标
      const xCenter = xCenterNorm * inputWidth;
      const yCenter = yCenterNorm * inputHeight;
      const width = widthNorm * inputWidth;
      const height = heightNorm * inputHeight;

      // 缩放到原始图像尺寸
      const xCenterScaled = xCenter * scaleX;
      const yCenterScaled = yCenter * scaleY;
      const widthScaled = width * scaleX;
      const heightScaled = height * scaleY;

      // 转换为左上角和右下角坐标
      let x1 = Math.round(xCenterScaled - widthScaled / 2);
      let y1 = Math.round(yCenterScaled - heightScaled / 2);
      let x2 = Math.round(xCenterScaled + widthScaled / 2);
      let y2 = Math.round(yCenterScaled + heightScaled / 2);

      // 确保坐标在图像范围内
      x1 = Math.max(0, Math.min(x1, originalWidth));
      y1 = Math.max(0, Math.min(y1, originalHeight));
      x2 = Math.max(0, Math.min(x2, originalWidth));
      y2 = Math.max(0, Math.min(y2, originalHeight));

      // 确保边界框有效
      if (x2 > x1 && y2 > y1) {
        detections.push({ x1, y1, x2, y2, conf: maxScore, classId });
      }
    }

    // 应用 NMS
    const filteredDetections = nms(detections);
    console.log(`检测到 ${detections.length} 个对象，NMS 后剩余 ${filteredDetections.length} 个`);

    return filteredDetections;
  }

  // 预处理 - 参考 Python 代码的 preprocess 方法
  async function preprocess(image: HTMLImageElement): Promise<tf.Tensor> {
    // 先获取原始图像的三维数组（在 tf.tidy 之外，避免被自动清理）
    const originalTensor = tf.browser.fromPixels(image, 3);
    const originalArray = await originalTensor.array() as number[][][];
    originalTensor.dispose();
    
    // 打印原始图像的三维数组（前10个像素）
    console.log('原始图像三维数组形状:', originalArray.length, 'x', originalArray[0]?.length, 'x', originalArray[0]?.[0]?.length);
    console.log('原始图像前10个像素的RGB值:');
    const height = originalArray.length;
    const width = originalArray[0]?.length || 0;
    let pixelCount = 0;
    
    for (let h = 0; h < height && pixelCount < 10; h++) {
      for (let w = 0; w < width && pixelCount < 10; w++) {
        const rgb = originalArray[h]?.[w];
        if (rgb && Array.isArray(rgb) && rgb.length === 3) {
          console.log(`  像素 [${h}, ${w}]: R=${rgb[0]}, G=${rgb[1]}, B=${rgb[2]}`);
          pixelCount++;
        }
      }
    }
    
    console.log('原始图像信息:', {
      width: image.width,
      height: image.height,
      naturalWidth: image.naturalWidth,
      naturalHeight: image.naturalHeight,
      complete: image.complete
    });
    
    // 先调整大小（不归一化），用于打印
    const resizedTensor = tf.browser.fromPixels(image, 3)
      .resizeBilinear([inputHeight, inputWidth]);
    const resizedArray = await resizedTensor.array() as number[][][];
    resizedTensor.dispose();
    
    // 打印调整到640x640后的前10个像素（未归一化，值在0-255之间）
    console.log('调整到640x640后的图像形状:', resizedArray.length, 'x', resizedArray[0]?.length, 'x', resizedArray[0]?.[0]?.length);
    console.log('调整到640x640后的前10个像素RGB值（未归一化，0-255）:');
    const resizedHeight = resizedArray.length;
    const resizedWidth = resizedArray[0]?.length || 0;
    let resizedPixelCount = 0;
    
    for (let h = 0; h < resizedHeight && resizedPixelCount < 10; h++) {
      for (let w = 0; w < resizedWidth && resizedPixelCount < 10; w++) {
        const rgb = resizedArray[h]?.[w];
        if (rgb && Array.isArray(rgb) && rgb.length === 3) {
          console.log(`  像素 [${h}, ${w}]: R=${rgb[0]}, G=${rgb[1]}, B=${rgb[2]}`);
          resizedPixelCount++;
        }
      }
    }
    
    return tf.tidy(() => {
      // fromPixels 已经是 RGB 格式
      // 调整大小到模型输入尺寸 (640x640)
      // 归一化到 [0, 1]
      // 添加批次维度: [1, 640, 640, 3]
      const tensor = tf.browser.fromPixels(image, 3)
        .div(255)
        .resizeBilinear([inputHeight, inputWidth])
        .reshape([1, inputHeight, inputWidth, 3]);
      
      console.log('预处理后张量形状:', tensor.shape);
      return tensor;
    });
  }

  // 绘制检测结果 - 参考 Python 代码的 draw_detections 方法
  function drawDetections(image: HTMLImageElement, detections: Detection[]) {
    const canvas = canvasRef.current;
    if (!canvas) {
      console.warn('Canvas ref not available');
      return;
    }

    // 确保图片已加载
    if (!image.complete || image.naturalWidth === 0) {
      console.warn('Image not fully loaded');
      return;
    }

    // 设置 canvas 尺寸
    canvas.width = image.width;
    canvas.height = image.height;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.warn('Could not get canvas context');
      return;
    }

    // 清除 canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 绘制原图
    ctx.drawImage(image, 0, 0);

    // 绘制每个检测框
    for (const det of detections) {
      const color = '#00ff00'; // 绿色
      
      // 绘制边界框
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

      // 准备标签
      const className = COCO_CLASSES[det.classId] || `Class ${det.classId}`;
      const label = `${className}: ${det.conf.toFixed(2)}`;

      // 计算文本大小
      ctx.font = 'bold 16px Arial';
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const textHeight = 20;

      // 绘制标签背景
      ctx.fillStyle = color;
      ctx.fillRect(
        det.x1,
        Math.max(0, det.y1 - textHeight - 5),
        textWidth + 8,
        textHeight
      );

      // 绘制标签文本
      ctx.fillStyle = '#000000';
      ctx.fillText(label, det.x1 + 4, Math.max(textHeight, det.y1 - 5));
    }

    console.log('Canvas drawn with', detections.length, 'detections');
  }

  // 检测图像 - 参考 Python 代码的 detect 方法
  async function detect(image: HTMLImageElement) {
    const originalWidth = image.width;
    const originalHeight = image.height;

    console.log(`处理图片: ${originalWidth}x${originalHeight}`);

    // 预处理
    const inputData = await preprocess(image);
    
    // 打印张量信息
    console.log('inputData 形状:', inputData.shape);
    console.log('inputData 数据类型:', inputData.dtype);
    
    // 获取并打印图像数组数据（只打印一小部分，避免控制台卡顿）
    const inputArray = await inputData.array() as number[][][][];
    console.log('inputData 数组形状:', inputArray.length, inputArray[0]?.length, inputArray[0]?.[0]?.length, inputArray[0]?.[0]?.[0]?.length);
    console.log('inputData 数组示例（第一个像素的 RGB 值）:', inputArray[0]?.[0]?.[0]);
    console.log('inputData 数组示例（中心像素的 RGB 值）:', inputArray[0]?.[320]?.[320]);

    // 运行模型
    const outputs = await runWithTfjsTensors(model!, [inputData]);
    const outputTensor = outputs[0];
    const outputArray = await outputTensor.array();

    // 后处理
    const processedDetections = postprocess(outputArray, originalWidth, originalHeight);
    setDetections(processedDetections);

    // 打印检测结果
    console.log(`检测到 ${processedDetections.length} 个对象:`);
    processedDetections.forEach((det, i) => {
      const className = COCO_CLASSES[det.classId] || `Class ${det.classId}`;
      console.log(
        `  ${i + 1}. ${className}: 置信度 ${det.conf.toFixed(2)}, 位置 (${det.x1}, ${det.y1}) - (${det.x2}, ${det.y2})`
      );
    });

    // 清理
    inputData.dispose();
    outputTensor.dispose();
  }

  // 当检测结果或图片更新时，绘制到 canvas
  useEffect(() => {
    if (detections.length > 0 && originalImage && canvasRef.current) {
      // 使用 requestAnimationFrame 确保 DOM 已更新
      requestAnimationFrame(() => {
        drawDetections(originalImage, detections);
      });
    }
  }, [detections, originalImage]);

  // 处理图片上传
  function handleFileUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    // 清除之前的检测结果和 canvas
    setDetections([]);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const url = e.target?.result as string;
      setImageUrl(url);
      
      // 创建图片对象
      const img = new Image();
      img.onload = () => {
        setOriginalImage(img);
      };
      img.src = url;
    };
    reader.readAsDataURL(file);
  }

  // 运行检测
  async function runDetection() {
    if (!model || !originalImage) return;

    setIsProcessing(true);
    await detect(originalImage);
    setIsProcessing(false);
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black p-8">
      <div className="max-w-7xl mx-auto">
        {/* 导航 */}
        <div className="flex gap-4 mb-6 border-b border-zinc-300 dark:border-zinc-700 pb-4">
          <Link href="/" className="px-4 py-2 text-zinc-600 dark:text-zinc-400 hover:text-blue-600 dark:hover:text-blue-400">
            Full Version
          </Link>
          <Link href="/detect" className="px-4 py-2 text-zinc-600 dark:text-zinc-400 hover:text-blue-600 dark:hover:text-blue-400">
            Debug Version
          </Link>
          <Link href="/yolo" className="px-4 py-2 font-semibold text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400">
            YOLO Detection
          </Link>
        </div>

        <h1 className="text-4xl font-bold mb-2 text-black dark:text-zinc-50">
          YOLO 物体检测
        </h1>
        <p className="text-lg text-zinc-600 dark:text-zinc-400 mb-8">
          上传图片进行物体检测，使用 YOLO11n 模型
        </p>

        {/* 加载状态 */}
        {isLoading && (
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg mb-6">
            <p className="text-blue-800 dark:text-blue-200">
              正在加载模型和初始化 WebGPU...
            </p>
          </div>
        )}

        {/* 文件上传 */}
        {!isLoading && model && (
          <div className="mb-6">
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-zinc-300 dark:border-zinc-700 rounded-lg cursor-pointer bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg
                  className="w-10 h-10 mb-3 text-zinc-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
                <p className="mb-2 text-sm text-zinc-500 dark:text-zinc-400">
                  <span className="font-semibold">点击上传</span> 或拖拽图片
                </p>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  PNG, JPG, GIF 最大 10MB
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept="image/*"
                onChange={handleFileUpload}
              />
            </label>
          </div>
        )}

        {/* 图片显示和检测 */}
        {imageUrl && originalImage && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* 原图 */}
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-semibold text-black dark:text-zinc-50">
                原始图片
              </h2>
              <div className="relative w-full border-2 border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden">
                <img
                  src={imageUrl}
                  alt="Uploaded"
                  className="w-full h-auto"
                />
              </div>
              <button
                onClick={runDetection}
                disabled={isProcessing || !model}
                className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-zinc-400 disabled:cursor-not-allowed transition-colors"
              >
                {isProcessing ? '检测中...' : '运行检测'}
              </button>
            </div>

            {/* 检测结果 */}
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-semibold text-black dark:text-zinc-50">
                检测结果
              </h2>
              <div className="relative w-full border-2 border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-zinc-100 dark:bg-zinc-900 min-h-[200px] flex items-center justify-center">
                {detections.length > 0 ? (
                  <canvas ref={canvasRef} className="w-full h-auto" />
                ) : (
                  <p className="text-zinc-500 dark:text-zinc-400">
                    {isProcessing ? '正在处理...' : '点击"运行检测"查看结果'}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 检测结果列表 */}
        {detections.length > 0 && (
          <div className="mt-6">
            <h2 className="text-xl font-semibold mb-4 text-black dark:text-zinc-50">
              检测结果 ({detections.length} 个对象)
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {detections.map((det, idx) => {
                const className = COCO_CLASSES[det.classId] || `Class ${det.classId}`;
                return (
                  <div
                    key={idx}
                    className="p-4 bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800"
                  >
                    <div className="font-semibold text-lg text-black dark:text-zinc-50 mb-2">
                      {className}
                    </div>
                    <div className="text-sm text-zinc-600 dark:text-zinc-400 space-y-1">
                      <div>置信度: {(det.conf * 100).toFixed(1)}%</div>
                      <div>
                        位置: ({det.x1}, {det.y1}) - ({det.x2}, {det.y2})
                      </div>
                      <div>
                        尺寸: {det.x2 - det.x1} × {det.y2 - det.y1}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* 提示信息 */}
        {!imageUrl && !isLoading && model && (
          <div className="mt-8 p-6 bg-zinc-100 dark:bg-zinc-900 rounded-lg text-center">
            <p className="text-zinc-600 dark:text-zinc-400">
              模型加载成功！请上传图片开始检测。
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

