/* eslint-disable @typescript-eslint/no-var-requires */
import { GLView } from 'expo-gl';
import { Image } from 'expo-image';
import * as React from 'react';
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  useTensorflowModel
} from 'react-native-fast-tflite';
import ImagePicker from 'react-native-image-crop-picker';

// COCO 类别名称
const COCO_CLASS_NAMES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  classId: number;
  className: string;
}

/**
 * 使用 GLView 读取图片像素数据并转换为模型输入格式
 * 直接读取 640x640 尺寸的像素数据
 */
async function readImagePixels(
  imageUri: string,
  width: number,
  height: number
): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    console.log('[readImagePixels] 开始读取像素数据');
    console.log('[readImagePixels] 图片 URI:', imageUri);
    console.log('[readImagePixels] 原始图片尺寸:', width, 'x', height);

    const targetWidth = 640;
    const targetHeight = 640;

    GLView.createContextAsync()
      .then((gl) => {
        console.log('[readImagePixels] GL 上下文创建成功');
        try {
          // 创建源纹理（加载原始图片）
          const sourceTexture = gl.createTexture();
          if (!sourceTexture) {
            throw new Error('无法创建源纹理');
          }
          gl.bindTexture(gl.TEXTURE_2D, sourceTexture);

          // 处理 URI 格式
          let localUri = imageUri;
          if (!imageUri.startsWith('file://')) {
            localUri = imageUri.startsWith('/')
              ? `file://${imageUri}`
              : `file:///${imageUri}`;
          }

          // 加载图片到源纹理
          gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            { localUri } as any
          );

          // 设置源纹理参数
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

          // 创建目标纹理（640x640）
          const targetTexture = gl.createTexture();
          if (!targetTexture) {
            throw new Error('无法创建目标纹理');
          }
          gl.bindTexture(gl.TEXTURE_2D, targetTexture);
          gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            targetWidth,
            targetHeight,
            0,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            null
          );
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

          // 创建帧缓冲并绑定目标纹理
          const framebuffer = gl.createFramebuffer();
          if (!framebuffer) {
            throw new Error('无法创建帧缓冲');
          }
          gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
          gl.framebufferTexture2D(
            gl.FRAMEBUFFER,
            gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D,
            targetTexture,
            0
          );

          // 检查帧缓冲状态
          const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
          if (status !== gl.FRAMEBUFFER_COMPLETE) {
            throw new Error(`帧缓冲不完整: ${status}`);
          }

          // 设置视口为 640x640
          gl.viewport(0, 0, targetWidth, targetHeight);

          // 创建简单的着色器程序来渲染纹理
          const vertexShaderSource = `
            attribute vec2 a_position;
            attribute vec2 a_texCoord;
            varying vec2 v_texCoord;
            void main() {
              gl_Position = vec4(a_position, 0.0, 1.0);
              v_texCoord = a_texCoord;
            }
          `;

          const fragmentShaderSource = `
            precision mediump float;
            uniform sampler2D u_texture;
            varying vec2 v_texCoord;
            void main() {
              gl_FragColor = texture2D(u_texture, v_texCoord);
            }
          `;

          function createShader(gl: WebGLRenderingContext, type: number, source: string) {
            const shader = gl.createShader(type);
            if (!shader) throw new Error('无法创建着色器');
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
              const info = gl.getShaderInfoLog(shader);
              gl.deleteShader(shader);
              throw new Error(`着色器编译失败: ${info}`);
            }
            return shader;
          }

          function createProgram(gl: WebGLRenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader) {
            const program = gl.createProgram();
            if (!program) throw new Error('无法创建程序');
            gl.attachShader(program, vertexShader);
            gl.attachShader(program, fragmentShader);
            gl.linkProgram(program);
            if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
              const info = gl.getProgramInfoLog(program);
              gl.deleteProgram(program);
              throw new Error(`程序链接失败: ${info}`);
            }
            return program;
          }

          const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
          const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
          const program = createProgram(gl, vertexShader, fragmentShader);

          // 设置顶点数据（全屏四边形）
          const positionBuffer = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
          gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([
              -1, -1, 1, -1, -1, 1,
              -1, 1, 1, -1, 1, 1,
            ]),
            gl.STATIC_DRAW
          );

          const texCoordBuffer = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
          // 纹理坐标：OpenGL 的 (0,0) 在左下角，图片的 (0,0) 在左上角
          // 需要翻转 Y 坐标以正确映射
          gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([
              // 全屏四边形，纹理坐标翻转 Y 轴
              0, 1, 1, 1, 0, 0,  // 第一个三角形
              0, 0, 1, 1, 1, 0,  // 第二个三角形
            ]),
            gl.STATIC_DRAW
          );

          // 使用程序并设置属性
          gl.useProgram(program);

          const positionLocation = gl.getAttribLocation(program, 'a_position');
          gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
          gl.enableVertexAttribArray(positionLocation);
          gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

          const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');
          gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
          gl.enableVertexAttribArray(texCoordLocation);
          gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0);

          // 绑定源纹理
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
          const textureLocation = gl.getUniformLocation(program, 'u_texture');
          gl.uniform1i(textureLocation, 0);

          // 清除并绘制
          gl.clearColor(0, 0, 0, 0);
          gl.clear(gl.COLOR_BUFFER_BIT);
          gl.drawArrays(gl.TRIANGLES, 0, 6);

          // 读取 640x640 的像素数据 (RGBA)
          // 注意：gl.readPixels 从底部开始读取，第一行对应图片的最后一行
          // 需要翻转 Y 轴以匹配 OpenCV 的坐标系统（第一行对应第一行）
          const pixels = new Uint8Array(targetWidth * targetHeight * 4);
          gl.readPixels(0, 0, targetWidth, targetHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

          // 翻转 Y 轴：gl.readPixels 从底部读取，需要翻转以匹配 OpenCV（从顶部读取）
          const pixelsFlipped = new Uint8Array(targetWidth * targetHeight * 4);
          const bytesPerRow = targetWidth * 4;
          for (let y = 0; y < targetHeight; y++) {
            const srcRow = targetHeight - 1 - y; // 从底部到顶部
            const srcOffset = srcRow * bytesPerRow;
            const dstOffset = y * bytesPerRow;
            pixelsFlipped.set(pixels.subarray(srcOffset, srcOffset + bytesPerRow), dstOffset);
          }

          // 打印前几个像素的原始 RGBA 值用于调试（翻转后）
          console.log('[readImagePixels] 翻转后前3个像素的原始 RGBA 值:');
          for (let i = 0; i < 3; i++) {
            const idx = i * 4;
            console.log(`  像素 ${i}: R=${pixelsFlipped[idx]}, G=${pixelsFlipped[idx + 1]}, B=${pixelsFlipped[idx + 2]}, A=${pixelsFlipped[idx + 3]}`);
          }

          // RGBA 转换为 RGB (Uint8Array)
          // 注意：WebGL 读取的是 RGBA，OpenCV imread 默认是 BGR，然后转换为 RGB
          // 所以我们的顺序应该是正确的：R, G, B
          const rgbPixels = new Uint8Array(targetWidth * targetHeight * 3);
          for (let i = 0; i < pixelsFlipped.length; i += 4) {
            const rgbIdx = (i / 4) * 3;
            rgbPixels[rgbIdx] = pixelsFlipped[i];     // R
            rgbPixels[rgbIdx + 1] = pixelsFlipped[i + 1]; // G
            rgbPixels[rgbIdx + 2] = pixelsFlipped[i + 2]; // B
          }
          
          // 打印前10个像素的 RGB 值（对应 OpenCV 的 image_resized[0][i]）
          console.log('[readImagePixels] 第一行前10个像素的 RGB 值 (对应 OpenCV image_resized[0][i]):');
          for (let i = 0; i < 10; i++) {
            const idx = i * 3;
            console.log(`  [${i}]: [${rgbPixels[idx]}, ${rgbPixels[idx + 1]}, ${rgbPixels[idx + 2]}]`);
          }

          // RGB 归一化到 [0, 1] 的 Float32Array
          // 形状: [640, 640, 3] (扁平化为 [640*640*3])
          const normalized = new Float32Array(targetWidth * targetHeight * 3);
          for (let i = 0; i < rgbPixels.length; i++) {
            normalized[i] = rgbPixels[i] / 255.0;
          }
          
          // 打印归一化后的前10个像素值（对应 OpenCV 的 image_normalized[0][i]）
          console.log('[readImagePixels] 归一化后第一行前10个像素值 (对应 OpenCV image_normalized[0][i]):');
          for (let i = 0; i < 10; i++) {
            const idx = i * 3;
            console.log(`  [${i}]: [${normalized[idx].toFixed(6)}, ${normalized[idx + 1].toFixed(6)}, ${normalized[idx + 2].toFixed(6)}]`);
          }

          // 添加批次维度: [1, 640, 640, 3]
          // 对应 Python: input_data = np.expand_dims(image_normalized, axis=0)
          // 在 JavaScript 中，由于使用扁平数组，数据已经是 [1, 640, 640, 3] 的展开形式
          // 形状信息: [1, 640, 640, 3] = 1 * 640 * 640 * 3 = 1,228,800 个元素
          // 当前 normalized 数组长度: 640 * 640 * 3 = 1,228,800 (单个批次)
          console.log('[readImagePixels] 数据形状: [1, 640, 640, 3]');
          console.log('[readImagePixels] 数据长度:', normalized.length, '(对应 1 * 640 * 640 * 3)');

          // 清理资源
          gl.deleteTexture(sourceTexture);
          gl.deleteTexture(targetTexture);
          gl.deleteFramebuffer(framebuffer);
          gl.deleteBuffer(positionBuffer);
          gl.deleteBuffer(texCoordBuffer);
          gl.deleteShader(vertexShader);
          gl.deleteShader(fragmentShader);
          gl.deleteProgram(program);
          GLView.destroyContextAsync(gl);

          console.log('[readImagePixels] 像素数据读取完成，尺寸: 640x640');
          resolve(normalized);
        } catch (error) {
          console.error('[readImagePixels] 处理过程中出错:', error);
          GLView.destroyContextAsync(gl).catch(() => {});
          reject(error);
        }
      })
      .catch((error) => {
        console.error('[readImagePixels] 创建 GL 上下文失败:', error);
        reject(error);
      });
  });
}

/**
 * 计算两个边界框的 IoU
 */
function calculateIoU(
  box1: { x1: number; y1: number; x2: number; y2: number },
  box2: { x1: number; y1: number; x2: number; y2: number }
): number {
  const x1I = Math.max(box1.x1, box2.x1);
  const y1I = Math.max(box1.y1, box2.y1);
  const x2I = Math.min(box1.x2, box2.x2);
  const y2I = Math.min(box1.y2, box2.y2);

  if (x2I < x1I || y2I < y1I) {
    return 0.0;
  }

  const intersection = (x2I - x1I) * (y2I - y1I);
  const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  const union = area1 + area2 - intersection;

  if (union === 0) {
    return 0.0;
  }

  return intersection / union;
}

/**
 * 非极大值抑制 (NMS)
 */
function nms(
  detections: Array<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
    classId: number;
  }>,
  iouThreshold: number = 0.45
): Array<{
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  classId: number;
}> {
  if (detections.length === 0) {
    return [];
  }

  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const keep: typeof detections = [];

  while (sorted.length > 0) {
    const current = sorted.shift()!;
    keep.push(current);

    for (let i = sorted.length - 1; i >= 0; i--) {
      if (calculateIoU(current, sorted[i]) >= iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return keep;
}

/**
 * 后处理模型输出
 * 参考 React 项目的实现，处理输出形状 [1, 84, 8400] 或 [84, 8400]
 * react-native-fast-tflite 返回的是 TypedArray，需要转置为 [8400, 84] 格式
 */
function postprocess(
  outputs: Float32Array | number[],
  originalWidth: number,
  originalHeight: number,
  confThreshold: number = 0.25
): Detection[] {
  // 转换为数组格式
  const outputArray = Array.isArray(outputs) ? outputs : Array.from(outputs);
  
  // 检查输出形状并重塑
  // YOLO 输出格式可能是：
  // - [1, 84, 8400] 扁平化: 1 * 84 * 8400 = 705,600
  // - [84, 8400] 扁平化: 84 * 8400 = 705,600
  // - [8400, 84] 扁平化: 8400 * 84 = 705,600
  const totalElements = outputArray.length;
  const expectedElements1 = 1 * 84 * 8400; // [1, 84, 8400]
  const expectedElements2 = 84 * 8400; // [84, 8400]
  const expectedElements3 = 8400 * 84; // [8400, 84]
  
  let reshapedArray: number[];
  
  if (totalElements === expectedElements1 || totalElements === expectedElements2) {
    // 需要转置：从 [84, 8400] 转换为 [8400, 84]
    // 参考 Python: outputs = outputs[0].transpose(1, 0)  # [84, 8400] -> [8400, 84]
    reshapedArray = [];
    for (let i = 0; i < 8400; i++) {
      for (let j = 0; j < 84; j++) {
        // 转置索引: [84, 8400] 的 (j, i) -> [8400, 84] 的 (i, j)
        // 扁平化索引: j * 8400 + i
        reshapedArray.push(outputArray[j * 8400 + i]);
      }
    }
  } else if (totalElements === expectedElements3) {
    // 已经是 [8400, 84] 格式，直接使用
    reshapedArray = outputArray;
  } else {
    console.warn(`[postprocess] 意外的输出长度: ${totalElements}, 期望: ${expectedElements1}, ${expectedElements2}, 或 ${expectedElements3}`);
    // 尝试按 [8400, 84] 格式处理
    const numDetections = Math.floor(totalElements / 84);
    if (numDetections * 84 !== totalElements) {
      throw new Error(`输出长度 ${totalElements} 不能被 84 整除`);
    }
    reshapedArray = outputArray;
  }
  
  // 现在 reshapedArray 是 [8400, 84] 格式的扁平数组
  const numDetections = 8400;
  const numFeatures = 84;
  
  console.log('[postprocess] 原始输出长度:', totalElements);
  console.log('[postprocess] 重塑后长度:', reshapedArray.length);
  console.log('[postprocess] 检测框数量:', numDetections);
  console.log('[postprocess] 每个检测框特征数:', numFeatures);
  
  const detections: Array<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
    classId: number;
  }> = [];

  const inputWidth = 640;
  const inputHeight = 640;
  const scaleX = originalWidth / inputWidth;
  const scaleY = originalHeight / inputHeight;

  for (let i = 0; i < numDetections; i++) {
    const baseIdx = i * numFeatures;

    // 格式: [x_center, y_center, width, height, class_scores...]
    // 坐标是归一化的 (0-1)，相对于输入尺寸 (640x640)
    const xCenterNorm = reshapedArray[baseIdx];
    const yCenterNorm = reshapedArray[baseIdx + 1];
    const widthNorm = reshapedArray[baseIdx + 2];
    const heightNorm = reshapedArray[baseIdx + 3];

    // 找到最高分的类别
    let maxScore = 0;
    let classId = 0;
    for (let j = 0; j < 80; j++) {
      const score = reshapedArray[baseIdx + 4 + j];
      if (score > maxScore) {
        maxScore = score;
        classId = j;
      }
    }

    const confidence = maxScore;

    if (confidence < confThreshold) {
      continue;
    }

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
    const x1 = Math.max(0, Math.min(xCenterScaled - widthScaled / 2, originalWidth));
    const y1 = Math.max(0, Math.min(yCenterScaled - heightScaled / 2, originalHeight));
    const x2 = Math.max(0, Math.min(xCenterScaled + widthScaled / 2, originalWidth));
    const y2 = Math.max(0, Math.min(yCenterScaled + heightScaled / 2, originalHeight));

    // 确保边界框有效
    if (x2 > x1 && y2 > y1) {
      detections.push({
        x1,
        y1,
        x2,
        y2,
        confidence,
        classId,
      });
    }
  }

  // 应用 NMS
  const filteredDetections = nms(detections);

  // 转换为 Detection 格式
  return filteredDetections.map((det) => ({
    ...det,
    className:
      det.classId < COCO_CLASS_NAMES.length
        ? COCO_CLASS_NAMES[det.classId]
        : `Class ${det.classId}`,
  }));
}

export default function ExpoView2Screen(): React.ReactNode {
  const [selectedImage, setSelectedImage] = React.useState<string | null>(null);
  const [imageInfo, setImageInfo] = React.useState<{
    path: string;
    width: number;
    height: number;
  } | null>(null);
  const [processingImage, setProcessingImage] = React.useState(false);
  const [detections, setDetections] = React.useState<Detection[]>([]);
  const [imageScale, setImageScale] = React.useState({ 
    scaleX: 1, 
    scaleY: 1, 
    displayWidth: 0, 
    displayHeight: 0,
    imageDisplayWidth: 0,
    imageDisplayHeight: 0,
    offsetX: 0,
    offsetY: 0,
  });

  // 加载模型
  const model = useTensorflowModel(require('@/assets/models/yolo11n_float32.tflite'));
  const actualModel = model.state === 'loaded' ? model.model : undefined;

  React.useEffect(() => {
    if (actualModel) {
      console.log('Model loaded!');
      console.log('Inputs:', actualModel.inputs);
      console.log('Outputs:', actualModel.outputs);
    }
  }, [actualModel]);

  /**
   * 从相册选择图片
   */
  const pickImageFromGallery = async () => {
    try {
      const image = await ImagePicker.openPicker({
        cropping: false,
        includeBase64: false,
        compressImageQuality: 1,
      });

      if (image && image.path) {
        setSelectedImage(image.path);
        setImageInfo({
          path: image.path,
          width: image.width,
          height: image.height,
        });
        setDetections([]);
        console.log('Selected image:', {
          path: image.path,
          width: image.width,
          height: image.height,
        });
      }
    } catch (error: any) {
      if (error.message !== 'User cancelled image selection') {
        Alert.alert('错误', `选择图片失败: ${error.message}`);
      }
    }
  };

  /**
   * 从相机拍照
   */
  const takePhoto = async () => {
    try {
      const image = await ImagePicker.openCamera({
        cropping: false,
        includeBase64: false,
        compressImageQuality: 1,
      });

      if (image && image.path) {
        setSelectedImage(image.path);
        setImageInfo({
          path: image.path,
          width: image.width,
          height: image.height,
        });
        setDetections([]);
        console.log('Captured image:', {
          path: image.path,
          width: image.width,
          height: image.height,
        });
      }
    } catch (error: any) {
      if (error.message !== 'User cancelled image selection') {
        Alert.alert('错误', `拍照失败: ${error.message}`);
      }
    }
  };

  /**
   * 处理图片并进行检测
   */
  const processImage = async () => {
    if (!imageInfo) {
      Alert.alert('提示', '请先选择一张图片');
      return;
    }

    if (!actualModel) {
      Alert.alert('错误', '模型尚未加载完成');
      return;
    }

    setProcessingImage(true);
    setDetections([]);

    try {
      console.log('开始处理图片...');
      
      // 1. 读取像素数据
      console.log('读取像素数据...');
      const pixelData = await readImagePixels(
        imageInfo.path,
        imageInfo.width,
        imageInfo.height
      );

      // 2. 准备模型输入 (形状: [1, 640, 640, 3])
      // 对应 Python: input_data = np.expand_dims(image_normalized, axis=0)
      // pixelData 已经是 [1, 640, 640, 3] 的扁平数组形式
      const inputShape = [1, 640, 640, 3];
      console.log('运行模型推理...');
      console.log('输入数据形状:', inputShape);
      console.log('输入数据长度:', pixelData.length, '(对应 1 * 640 * 640 * 3)');
      
      // 3. 运行模型推理
      // model.run() 接受 TypedArray[]，每个元素对应一个输入张量
      // 对于形状 [1, 640, 640, 3]，传入扁平数组即可
      const outputs = await actualModel.run([pixelData]);
      console.log('模型推理完成');
      console.log('输出数量:', outputs.length);
      console.log('输出数据长度:', outputs.map((o: any) => o?.length || 'unknown'));
      
      // 打印 outputs 数组的前10个元素（如果 outputs 有多个输出）
      console.log('outputs 数组信息:');
      for (let i = 0; i < Math.min(outputs.length, 10); i++) {
        const output = outputs[i];
        console.log(`  outputs[${i}]: 长度=${output.length}, 类型=${output.constructor.name}`);
      }
      
      // 4. 后处理
      console.log('后处理检测结果...');
      // model.run() 返回 TypedArray[]，第一个输出是检测结果
      const outputData = outputs[0];
      console.log('输出数据长度:', outputData.length);
      console.log('输出数据类型:', outputData.constructor.name);
      
      // 打印第一个输出的前10个值
      console.log('outputs[0] 前10个值:');
      const maxPrint = Math.min(10, outputData.length);
      for (let i = 0; i < maxPrint; i++) {
        const value = outputData[i];
        if (typeof value === 'number') {
          console.log(`  [${i}]: ${value.toFixed(6)}`);
        } else {
          console.log(`  [${i}]: ${value}`);
        }
      }
      
      // 检查输出数据的形状
      // React 项目中使用 outputTensor.array() 可能返回多维数组
      // react-native-fast-tflite 返回的是 TypedArray，需要转换为数组格式
      let processedOutput: number[];
      
      if (outputData instanceof Float32Array || outputData instanceof Uint8Array || 
          outputData instanceof Int8Array || outputData instanceof Int16Array ||
          outputData instanceof Int32Array || outputData instanceof Uint16Array ||
          outputData instanceof Uint32Array) {
        // TypedArray，转换为普通数组
        processedOutput = Array.from(outputData);
      } else if (Array.isArray(outputData)) {
        // 已经是数组
        processedOutput = outputData as number[];
      } else {
        // 其他类型，尝试转换
        processedOutput = Array.from(outputData as ArrayLike<number>);
      }
      
      console.log('处理后输出类型: Array');
      console.log('处理后输出长度:', processedOutput.length);
      
      const detected = postprocess(
        processedOutput,
        imageInfo.width,
        imageInfo.height
      );

      console.log(`检测到 ${detected.length} 个目标`);
      setDetections(detected);

      Alert.alert('成功', `检测完成！\n检测到 ${detected.length} 个目标`);
    } catch (error: any) {
      console.error('处理失败:', error);
      Alert.alert('错误', `处理图片失败: ${error.message}`);
    } finally {
      setProcessingImage(false);
    }
  };

  // 重置图片缩放信息
  React.useEffect(() => {
    if (!imageInfo || !selectedImage) {
      setImageScale({ 
        scaleX: 1, 
        scaleY: 1, 
        displayWidth: 0, 
        displayHeight: 0,
        imageDisplayWidth: 0,
        imageDisplayHeight: 0,
        offsetX: 0,
        offsetY: 0,
      });
    } else {
      // 重置时也设置默认值
      setImageScale((prev) => ({
        ...prev,
        scaleX: 1,
        scaleY: 1,
        imageDisplayWidth: 0,
        imageDisplayHeight: 0,
        offsetX: 0,
        offsetY: 0,
      }));
    }
  }, [imageInfo, selectedImage]);

  return (
    <View style={styles.container}>
      {/* 标题 */}
      <View style={styles.header}>
        <Text style={styles.title}>YOLO 目标检测</Text>
      </View>

      {/* 按钮区域 */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.button, styles.galleryButton]}
          onPress={pickImageFromGallery}
          disabled={processingImage || model.state !== 'loaded'}>
          <Text style={styles.buttonText}>从相册选择</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.cameraButton]}
          onPress={takePhoto}
          disabled={processingImage || model.state !== 'loaded'}>
          <Text style={styles.buttonText}>拍照</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.processButton]}
          onPress={processImage}
          disabled={processingImage || !imageInfo || model.state !== 'loaded'}>
          <Text style={styles.buttonText}>开始检测</Text>
        </TouchableOpacity>
      </View>

      {/* 模型状态 */}
      {model.state === 'loading' && (
        <View style={styles.statusContainer}>
          <ActivityIndicator size="small" color="#007AFF" />
          <Text style={styles.statusText}>模型加载中...</Text>
        </View>
      )}

      {model.state === 'error' && (
        <View style={styles.statusContainer}>
          <Text style={styles.errorText}>模型加载失败: {model.error?.message}</Text>
        </View>
      )}

      {/* 图片和检测结果 */}
      {selectedImage && imageInfo && (
        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}>
          <View style={styles.imageContainer}>
            <View 
              style={styles.imageWrapper}
              onLayout={(e) => {
                if (!imageInfo) return;
                const { width: containerWidth, height: containerHeight } = e.nativeEvent.layout;
                setImageScale((prev) => {
                  // 如果容器尺寸已设置，计算图片实际显示尺寸和缩放比例
                  if (prev.imageDisplayWidth === 0 && imageInfo) {
                    const imageAspectRatio = imageInfo.width / imageInfo.height;
                    const containerAspectRatio = containerWidth / containerHeight;
                    
                    let actualDisplayWidth: number;
                    let actualDisplayHeight: number;
                    
                    if (imageAspectRatio > containerAspectRatio) {
                      actualDisplayWidth = containerWidth;
                      actualDisplayHeight = containerWidth / imageAspectRatio;
                    } else {
                      actualDisplayHeight = containerHeight;
                      actualDisplayWidth = containerHeight * imageAspectRatio;
                    }
                    
                    const scaleX = actualDisplayWidth / imageInfo.width;
                    const scaleY = actualDisplayHeight / imageInfo.height;
                    const offsetX = (containerWidth - actualDisplayWidth) / 2;
                    const offsetY = (containerHeight - actualDisplayHeight) / 2;
                    
                    return {
                      ...prev,
                      displayWidth: containerWidth,
                      displayHeight: containerHeight,
                      imageDisplayWidth: actualDisplayWidth,
                      imageDisplayHeight: actualDisplayHeight,
                      scaleX,
                      scaleY,
                      offsetX,
                      offsetY,
                    };
                  }
                  
                  return {
                    ...prev,
                    displayWidth: containerWidth,
                    displayHeight: containerHeight,
                  };
                });
              }}>
              <Image
                source={{ uri: selectedImage }}
                style={styles.image}
                contentFit="contain"
                onLoad={(e) => {
                  if (!imageInfo) return;
                  // 获取图片的实际显示尺寸
                  const { width: imageDisplayWidth, height: imageDisplayHeight } = e.source;
                  
                  // 使用当前容器尺寸（如果已设置）或计算默认值
                  const containerWidth = imageScale.displayWidth || Dimensions.get('window').width - 32;
                  const containerHeight = imageScale.displayHeight || containerWidth;
                  
                  // 计算实际显示尺寸（考虑 contentFit="contain"）
                  const imageAspectRatio = imageInfo.width / imageInfo.height;
                  const containerAspectRatio = containerWidth / containerHeight;
                  
                  let actualDisplayWidth: number;
                  let actualDisplayHeight: number;
                  
                  if (imageAspectRatio > containerAspectRatio) {
                    // 图片更宽，以宽度为准
                    actualDisplayWidth = containerWidth;
                    actualDisplayHeight = containerWidth / imageAspectRatio;
                  } else {
                    // 图片更高，以高度为准
                    actualDisplayHeight = containerHeight;
                    actualDisplayWidth = containerHeight * imageAspectRatio;
                  }
                  
                  // 计算缩放比例（基于原始图片尺寸）
                  const scaleX = actualDisplayWidth / imageInfo.width;
                  const scaleY = actualDisplayHeight / imageInfo.height;
                  
                  // 计算偏移量（图片在容器中居中）
                  const offsetX = (containerWidth - actualDisplayWidth) / 2;
                  const offsetY = (containerHeight - actualDisplayHeight) / 2;
                  
                  setImageScale({
                    scaleX,
                    scaleY,
                    displayWidth: containerWidth,
                    displayHeight: containerHeight,
                    imageDisplayWidth: actualDisplayWidth,
                    imageDisplayHeight: actualDisplayHeight,
                    offsetX,
                    offsetY,
                  });
                  
                  console.log('[Image onLoad] 图片信息:', {
                    原始尺寸: `${imageInfo.width}x${imageInfo.height}`,
                    容器尺寸: `${containerWidth}x${containerHeight}`,
                    实际显示尺寸: `${actualDisplayWidth.toFixed(2)}x${actualDisplayHeight.toFixed(2)}`,
                    缩放比例: `scaleX=${scaleX.toFixed(4)}, scaleY=${scaleY.toFixed(4)}`,
                    偏移量: `offsetX=${offsetX.toFixed(2)}, offsetY=${offsetY.toFixed(2)}`,
                  });
                }}
              />
              
              {/* 绘制检测框 */}
              {detections.length > 0 && imageScale.imageDisplayWidth > 0 && (
                <View style={StyleSheet.absoluteFill} pointerEvents="none">
                  {detections.map((det, index) => {
                    // 将原始图片坐标转换为显示坐标
                    // 检测结果的坐标是基于原始图片尺寸的
                    const x1 = det.x1 * imageScale.scaleX + (imageScale.offsetX || 0);
                    const y1 = det.y1 * imageScale.scaleY + (imageScale.offsetY || 0);
                    const x2 = det.x2 * imageScale.scaleX + (imageScale.offsetX || 0);
                    const y2 = det.y2 * imageScale.scaleY + (imageScale.offsetY || 0);
                    const width = x2 - x1;
                    const height = y2 - y1;

                    return (
                      <View key={index}>
                        <View
                          style={[
                            styles.boundingBox,
                            {
                              position: 'absolute',
                              left: x1,
                              top: y1,
                              width: width,
                              height: height,
                            },
                          ]}
                        />
                        <View
                          style={[
                            styles.labelContainer,
                            {
                              position: 'absolute',
                              left: x1,
                              top: Math.max(0, y1 - 20),
                            },
                          ]}>
                          <Text style={styles.labelText}>
                            {det.className} {(det.confidence * 100).toFixed(1)}%
                          </Text>
                        </View>
                      </View>
                    );
                  })}
                </View>
              )}
            </View>
          </View>

          {/* 检测结果列表 */}
          {detections.length > 0 && (
            <View style={styles.resultsContainer}>
              <Text style={styles.resultsTitle}>检测结果 ({detections.length})</Text>
              {detections.map((det, index) => (
                <View key={index} style={styles.resultItem}>
                  <Text style={styles.resultText}>
                    {index + 1}. {det.className} - {(det.confidence * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.resultCoords}>
                    位置: ({Math.round(det.x1)}, {Math.round(det.y1)}) - ({Math.round(det.x2)}, {Math.round(det.y2)})
                  </Text>
                </View>
              ))}
            </View>
          )}
        </ScrollView>
      )}

      {/* 处理中指示器 */}
      {processingImage && (
        <View style={styles.processingOverlay}>
          <ActivityIndicator size="large" color="white" />
          <Text style={styles.processingText}>处理图片中...</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    paddingTop: 60,
  },
  header: {
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  buttonContainer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    flexWrap: 'wrap',
  },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    minWidth: 100,
    alignItems: 'center',
    justifyContent: 'center',
  },
  galleryButton: {
    backgroundColor: '#007AFF',
  },
  cameraButton: {
    backgroundColor: '#34C759',
  },
  processButton: {
    backgroundColor: '#FF9500',
  },
  buttonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    gap: 8,
  },
  statusText: {
    fontSize: 14,
    color: '#666',
  },
  errorText: {
    fontSize: 14,
    color: '#ff3b30',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  imageContainer: {
    backgroundColor: '#000',
    borderRadius: 8,
    overflow: 'hidden',
    marginBottom: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageWrapper: {
    width: '100%',
    position: 'relative',
  },
  image: {
    width: '100%',
    aspectRatio: 1,
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00ff00',
    backgroundColor: 'transparent',
  },
  labelContainer: {
    position: 'absolute',
    backgroundColor: 'rgba(0, 255, 0, 0.8)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  labelText: {
    color: '#000',
    fontSize: 12,
    fontWeight: 'bold',
  },
  resultsContainer: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  resultsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  resultItem: {
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  resultText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '600',
  },
  resultCoords: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  processingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  processingText: {
    color: 'white',
    marginTop: 16,
    fontSize: 16,
  },
});

