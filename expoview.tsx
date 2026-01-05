import { GLView } from 'expo-gl';
import * as React from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import ImagePicker from 'react-native-image-crop-picker';

/**
 * 将扁平化的像素数组转换为三维数组
 * @param flatArray 扁平化的数组，格式为 [R, G, B, R, G, B, ...]
 * @param width 图片宽度
 * @param height 图片高度
 * @param channels 通道数，默认为 3 (RGB)
 * @returns 三维数组，格式为 [height][width][channels]
 */
function convertTo3DArray(
  flatArray: Float32Array | Uint8Array,
  width: number,
  height: number,
  channels: number = 3
): number[][][] {
  const result: number[][][] = [];
  
  for (let y = 0; y < height; y++) {
    const row: number[][] = [];
    for (let x = 0; x < width; x++) {
      const pixel: number[] = [];
      const flatIndex = (y * width + x) * channels;
      for (let c = 0; c < channels; c++) {
        pixel.push(flatArray[flatIndex + c]);
      }
      row.push(pixel);
    }
    result.push(row);
  }
  
  return result;
}

/**
 * 使用 GLView 读取图片像素数据
 * @param imageUri 图片 URI
 * @param width 图片宽度
 * @param height 图片高度
 * @returns 归一化后的 RGB 像素数据，包含扁平数组和三维数组
 */
async function readImagePixels(
  imageUri: string,
  width: number,
  height: number
): Promise<{
  flatArray: Float32Array;
  array3D: number[][][];
  width: number;
  height: number;
}> {
  return new Promise((resolve, reject) => {
    console.log('[readImagePixels] 开始读取像素数据');
    console.log('[readImagePixels] 图片 URI:', imageUri);
    console.log('[readImagePixels] 图片尺寸:', width, 'x', height);

    // 创建离屏 GL 上下文
    GLView.createContextAsync()
      .then((gl) => {
        console.log('[readImagePixels] GL 上下文创建成功');
        try {
          // 设置视口
          gl.viewport(0, 0, width, height);
          console.log('[readImagePixels] 视口设置完成:', width, 'x', height);

          // 创建纹理
          const texture = gl.createTexture();
          if (!texture) {
            throw new Error('无法创建纹理');
          }
          gl.bindTexture(gl.TEXTURE_2D, texture);
          console.log('[readImagePixels] 纹理创建并绑定成功');

          // 处理 URI 格式
          let localUri = imageUri;
          if (!imageUri.startsWith('file://')) {
            localUri = imageUri.startsWith('/')
              ? `file://${imageUri}`
              : `file:///${imageUri}`;
          }
          console.log('[readImagePixels] 处理后的 URI:', localUri);

          // 加载图片到纹理
          gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            { localUri } as any // Expo GLView 支持 localUri 格式
          );
          console.log('[readImagePixels] 图片已加载到纹理');

          // 设置纹理参数
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          console.log('[readImagePixels] 纹理参数设置完成');

          // 创建帧缓冲
          const framebuffer = gl.createFramebuffer();
          if (!framebuffer) {
            throw new Error('无法创建帧缓冲');
          }
          gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
          gl.framebufferTexture2D(
            gl.FRAMEBUFFER,
            gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D,
            texture,
            0
          );
          console.log('[readImagePixels] 帧缓冲创建并配置完成');

          // 检查帧缓冲状态
          const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
          if (status !== gl.FRAMEBUFFER_COMPLETE) {
            throw new Error(`帧缓冲不完整: ${status}`);
          }
          console.log('[readImagePixels] 帧缓冲状态检查通过');
          // const { resize } = useResizePlugin();
          // const resized = resize(framebuffer, {
          //   scale: {
          //     width: 320,
          //     height: 320,
          //   },
          //   pixelFormat: 'rgb',
          //   dataType: 'uint8',
          // })
          // console.log('[readImagePixels] 图片已调整大小:', resized)
          // 读取像素数据 (RGBA)
          const pixels = new Uint8Array(width * height * 4);
          console.log('[readImagePixels] 开始读取像素数据，数组大小:', pixels.length);
          gl.readPixels(0, 0, 640, 640, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          console.log('[readImagePixels] 像素数据读取完成');
          console.log('[readImagePixels] 原始像素数据长度:', pixels.length);
          console.log('[readImagePixels] 前10个像素值 (RGBA):', Array.from(pixels.slice(0, 40)));

          // 步骤 1: RGBA 转换为 RGB (Uint8Array，未归一化)
          const rgbPixels = new Uint8Array(width * height * 3);
          for (let i = 0; i < pixels.length; i += 4) {
            const rgbIdx = (i / 4) * 3;
            rgbPixels[rgbIdx] = pixels[i];     // R
            rgbPixels[rgbIdx + 1] = pixels[i + 1]; // G
            rgbPixels[rgbIdx + 2] = pixels[i + 2]; // B
            // 忽略 Alpha 通道 (pixels[i + 3])
          }
          console.log('[readImagePixels] RGBA 转 RGB 完成');
          console.log('[readImagePixels] RGB 数据长度:', rgbPixels.length);
          console.log('[readImagePixels] 前9个 RGB 值 (Uint8):', Array.from(rgbPixels.slice(0, 9)));
          console.log('[readImagePixels] 第一个像素 RGB:', `[${rgbPixels[0]}, ${rgbPixels[1]}, ${rgbPixels[2]}]`);
          console.log('[readImagePixels] 第二个像素 RGB:', `[${rgbPixels[3]}, ${rgbPixels[4]}, ${rgbPixels[5]}]`);
          console.log('[readImagePixels] 第三个像素 RGB:', `[${rgbPixels[6]}, ${rgbPixels[7]}, ${rgbPixels[8]}]`);

          // 步骤 2: 将 RGB 图片缩放到 640x640
          const targetWidth = 640;
          const targetHeight = 640;
          const resizedRgbPixels = new Uint8Array(targetWidth * targetHeight * 3);

          // 计算缩放比例
          const scaleX = width / targetWidth;
          const scaleY = height / targetHeight;

          // 使用双线性插值进行缩放
          for (let y = 0; y < targetHeight; y++) {
            for (let x = 0; x < targetWidth; x++) {
              // 计算源图像坐标
              const srcX = x * scaleX;
              const srcY = y * scaleY;
              
              // 获取四个最近邻像素的坐标
              const x1 = Math.floor(srcX);
              const y1 = Math.floor(srcY);
              const x2 = Math.min(x1 + 1, width - 1);
              const y2 = Math.min(y1 + 1, height - 1);
              
              // 计算插值权重
              const fx = srcX - x1;
              const fy = srcY - y1;
              
              // 获取四个像素的索引
              const idx11 = (y1 * width + x1) * 3; // 左上
              const idx21 = (y1 * width + x2) * 3; // 右上
              const idx12 = (y2 * width + x1) * 3; // 左下
              const idx22 = (y2 * width + x2) * 3; // 右下
              
              // 目标像素索引
              const dstIdx = (y * targetWidth + x) * 3;
              
              // 对每个通道进行双线性插值
              for (let c = 0; c < 3; c++) {
                const p11 = rgbPixels[idx11 + c];
                const p21 = rgbPixels[idx21 + c];
                const p12 = rgbPixels[idx12 + c];
                const p22 = rgbPixels[idx22 + c];
                
                // 双线性插值公式
                const value = 
                  p11 * (1 - fx) * (1 - fy) +
                  p21 * fx * (1 - fy) +
                  p12 * (1 - fx) * fy +
                  p22 * fx * fy;
                
                resizedRgbPixels[dstIdx + c] = Math.round(value);
              }
            }
          }

          console.log('[readImagePixels] RGB 缩放完成');
          console.log('[readImagePixels] 原始尺寸:', width, 'x', height);
          console.log('[readImagePixels] 缩放后尺寸:', targetWidth, 'x', targetHeight);
          console.log('[readImagePixels] 缩放后 RGB 数据长度:', resizedRgbPixels.length);
          console.log('[readImagePixels] 缩放后前9个 RGB 值:', Array.from(resizedRgbPixels.slice(0, 9)));
          console.log('[readImagePixels] 缩放后第一个像素 RGB:', `[${resizedRgbPixels[0]}, ${resizedRgbPixels[1]}, ${resizedRgbPixels[2]}]`);
          console.log('[readImagePixels] 缩放后第二个像素 RGB:', `[${resizedRgbPixels[3]}, ${resizedRgbPixels[4]}, ${resizedRgbPixels[5]}]`);

          // 步骤 3: RGB 归一化到 [0, 1] 的 Float32Array（使用缩放后的尺寸）
          const normalized = new Float32Array(targetWidth * targetHeight * 3);
          for (let i = 0; i < resizedRgbPixels.length; i++) {
            normalized[i] = resizedRgbPixels[i] / 255.0;
          }
          console.log('[readImagePixels] RGB 归一化完成');
          console.log('[readImagePixels] 归一化数据长度:', normalized.length);
          console.log('[readImagePixels] 前9个归一化值 (RGB):', Array.from(normalized.slice(0, 9)));
          console.log('[readImagePixels] 第一个像素归一化 RGB:', `[${normalized[0].toFixed(3)}, ${normalized[1].toFixed(3)}, ${normalized[2].toFixed(3)}]`);
          console.log('[readImagePixels] 第二个像素归一化 RGB:', `[${normalized[3].toFixed(3)}, ${normalized[4].toFixed(3)}, ${normalized[5].toFixed(3)}]`);
          
          // 使用循环计算统计信息，避免展开运算符导致栈溢出
          let min = Infinity;
          let max = -Infinity;
          let sum = 0;
          for (let i = 0; i < normalized.length; i++) {
            const value = normalized[i];
            if (value < min) min = value;
            if (value > max) max = value;
            sum += value;
          }
          const mean = sum / normalized.length;
          
          console.log('[readImagePixels] 数据统计:', {
            min: min,
            max: max,
            mean: mean,
          });

          // 转换为三维数组 [height][width][channels]（使用缩放后的尺寸）
          const array3D = convertTo3DArray(normalized, targetWidth, targetHeight, 3);
          console.log('[readImagePixels] 三维数组转换完成');
          console.log('[readImagePixels] 三维数组形状: [', targetHeight, '][', targetWidth, '][3]');
          console.log('[readImagePixels] 第一个像素 (array3D[0][0]):', array3D[0][0]);
          console.log('[readImagePixels] 第二个像素 (array3D[0][1]):', array3D[0][1]);

          // 清理资源
          gl.deleteTexture(texture);
          gl.deleteFramebuffer(framebuffer);
          GLView.destroyContextAsync(gl);
          console.log('[readImagePixels] 资源清理完成');

          resolve({
            flatArray: normalized,
            array3D: array3D,
            width: targetWidth,
            height: targetHeight,
          });
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

export default function ExpoViewScreen(): React.ReactNode {
  const [selectedImage, setSelectedImage] = React.useState<string | null>(null);
  const [imageInfo, setImageInfo] = React.useState<{
    path: string;
    width: number;
    height: number;
  } | null>(null);
  const [processingImage, setProcessingImage] = React.useState(false);
  const [pixelData, setPixelData] = React.useState<{
    length: number;
    sampleValues: number[];
    array3DShape: string;
    sample3D: number[][];
    statistics: {
      min: number;
      max: number;
      mean: number;
    };
  } | null>(null);
  const [logs, setLogs] = React.useState<string[]>([]);

  // 添加日志
  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const logMessage = `[${timestamp}] ${message}`;
    console.log(logMessage);
    setLogs((prev) => [...prev.slice(-49), logMessage]); // 保留最近50条日志
  };

  /**
   * 从相册选择图片
   */
  const pickImageFromGallery = async () => {
    try {
      addLog('开始从相册选择图片');
      const image = await ImagePicker.openPicker({
        cropping: false,
        includeBase64x: false,
        compressImageQuality: 1,
      });

      if (image && image.path) {
        addLog(`图片选择成功: ${image.path}`);
        setSelectedImage(image.path);
        setImageInfo({
          path: image.path,
          width: image.width,
          height: image.height,
        });
        addLog(`图片尺寸: ${image.width} x ${image.height}`);
      }
    } catch (error: any) {
      if (error.message !== 'User cancelled image selection') {
        addLog(`选择图片失败: ${error.message}`);
        Alert.alert('错误', `选择图片失败: ${error.message}`);
      } else {
        addLog('用户取消了图片选择');
      }
    }
  };

  /**
   * 从相机拍照
   */
  const takePhoto = async () => {
    try {
      addLog('开始拍照');
      const image = await ImagePicker.openCamera({
        cropping: false,
        includeBase64: false,
        compressImageQuality: 1,
      });

      if (image && image.path) {
        addLog(`拍照成功: ${image.path}`);
        setSelectedImage(image.path);
        setImageInfo({
          path: image.path,
          width: image.width,
          height: image.height,
        });
        addLog(`图片尺寸: ${image.width} x ${image.height}`);
      }
    } catch (error: any) {
      if (error.message !== 'User cancelled image selection') {
        addLog(`拍照失败: ${error.message}`);
        Alert.alert('错误', `拍照失败: ${error.message}`);
      } else {
        addLog('用户取消了拍照');
      }
    }
  };

  /**
   * 处理图片并读取像素
   */
  const processImage = async () => {
    if (!imageInfo) {
      Alert.alert('提示', '请先选择一张图片');
      return;
    }

    setProcessingImage(true);
    setPixelData(null);
    addLog('开始处理图片');

    try {
      // 读取像素数据
      addLog('开始使用 GLView 读取像素数据');
      addLog(`图片尺寸: ${imageInfo.width} x ${imageInfo.height}`);
      const pixelData = await readImagePixels(
        imageInfo.path,
        640,
        640
      );

      // 步骤 3: 计算统计信息（使用循环避免栈溢出）
      let min = Infinity;
      let max = -Infinity;
      let sum = 0;
      for (let i = 0; i < pixelData.flatArray.length; i++) {
        const value = pixelData.flatArray[i];
        if (value < min) min = value;
        if (value > max) max = value;
        sum += value;
      }
      const statistics = {
        min: min,
        max: max,
        mean: sum / pixelData.flatArray.length,
      };

      // 获取三维数组的示例数据
      const sample3D = pixelData.array3D[0][0]; // 第一个像素 [R, G, B]
      const sample3D2 = pixelData.array3D[0][1]; // 第二个像素 [R, G, B]
      const sample3DRow = pixelData.array3D[0].slice(0, 5); // 第一行的前5个像素

      setPixelData({
        length: pixelData.flatArray.length,
        sampleValues: Array.from(pixelData.flatArray.slice(0, 30)), // 前10个像素的RGB值
        array3DShape: `[${pixelData.height}][${pixelData.width}][3]`,
        sample3D: sample3DRow, // 第一行的前5个像素
        statistics,
      });

      addLog('像素数据读取完成');
      addLog(`扁平数组长度: ${pixelData.flatArray.length}`);
      addLog(`三维数组形状: [${pixelData.height}][${pixelData.width}][3]`);
      addLog(`第一个像素 (array3D[0][0]): [${sample3D.map(v => v.toFixed(3)).join(', ')}]`);
      addLog(`第二个像素 (array3D[0][1]): [${sample3D2.map(v => v.toFixed(3)).join(', ')}]`);
      addLog(`数据范围: [${statistics.min.toFixed(4)}, ${statistics.max.toFixed(4)}]`);
      addLog(`平均值: ${statistics.mean.toFixed(4)}`);

      Alert.alert('成功', `图片像素读取完成\n扁平数组长度: ${pixelData.flatArray.length}\n三维数组形状: [${pixelData.height}][${pixelData.width}][3]`);
    } catch (error: any) {
      addLog(`处理失败: ${error.message}`);
      console.error('Processing error:', error);
      Alert.alert('错误', `处理图片失败: ${error.message}`);
    } finally {
      setProcessingImage(false);
    }
  };

  /**
   * 清空日志
   */
  const clearLogs = () => {
    setLogs([]);
    addLog('日志已清空');
  };

  return (
    <View style={styles.container}>
      {/* 标题 */}
      <View style={styles.header}>
        <Text style={styles.title}>Expo View - 图片像素读取</Text>
      </View>

      {/* 按钮区域 */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.button, styles.galleryButton]}
          onPress={pickImageFromGallery}
          disabled={processingImage}>
          <Text style={styles.buttonText}>从相册选择</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.cameraButton]}
          onPress={takePhoto}
          disabled={processingImage}>
          <Text style={styles.buttonText}>拍照</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.processButton]}
          onPress={processImage}
          disabled={processingImage || !imageInfo}>
          <Text style={styles.buttonText}>读取像素</Text>
        </TouchableOpacity>
      </View>

      {/* 图片信息 */}
      {imageInfo && (
        <View style={styles.infoContainer}>
          <Text style={styles.infoTitle}>图片信息</Text>
          <Text style={styles.infoText}>路径: {imageInfo.path}</Text>
          <Text style={styles.infoText}>
            尺寸: {imageInfo.width} x {imageInfo.height}
          </Text>
        </View>
      )}

      {/* 像素数据信息 */}
      {pixelData && (
        <View style={styles.infoContainer}>
          <Text style={styles.infoTitle}>像素数据</Text>
          <Text style={styles.infoText}>扁平数组长度: {pixelData.length}</Text>
          <Text style={styles.infoText}>三维数组形状: {pixelData.array3DShape}</Text>
          <Text style={styles.infoText}>
            最小值: {pixelData.statistics.min.toFixed(4)}
          </Text>
          <Text style={styles.infoText}>
            最大值: {pixelData.statistics.max.toFixed(4)}
          </Text>
          <Text style={styles.infoText}>
            平均值: {pixelData.statistics.mean.toFixed(4)}
          </Text>
          <Text style={styles.infoText}>
            前10个像素值 (扁平数组 RGB):
          </Text>
          <Text style={styles.sampleText}>
            {pixelData.sampleValues.map((v, i) => 
              i % 3 === 0 ? `\n[${Math.floor(i/3)}] ` : ''
            ).join('')}
            {pixelData.sampleValues.map((v, i) => 
              `${v.toFixed(3)}${i % 3 === 2 ? ' ' : ', '}`
            ).join('')}
          </Text>
          <Text style={styles.infoText}>
            三维数组示例 (第一行前5个像素):
          </Text>
          <Text style={styles.sampleText}>
            {pixelData.sample3D.map((pixel, i) => 
              `[${i}]: [${pixel.map(v => v.toFixed(3)).join(', ')}]\n`
            ).join('')}
          </Text>
        </View>
      )}

      {/* 日志区域 */}
      <View style={styles.logContainer}>
        <View style={styles.logHeader}>
          <Text style={styles.logTitle}>日志</Text>
          <TouchableOpacity onPress={clearLogs} style={styles.clearButton}>
            <Text style={styles.clearButtonText}>清空</Text>
          </TouchableOpacity>
        </View>
        <ScrollView style={styles.logScrollView}>
          {logs.map((log, index) => (
            <Text key={index} style={styles.logText}>
              {log}
            </Text>
          ))}
        </ScrollView>
      </View>

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
  infoContainer: {
    backgroundColor: '#fff',
    margin: 16,
    padding: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 12,
    color: '#666',
    marginVertical: 2,
    fontFamily: 'monospace',
  },
  sampleText: {
    fontSize: 10,
    color: '#666',
    marginTop: 4,
    fontFamily: 'monospace',
  },
  logContainer: {
    flex: 1,
    backgroundColor: '#fff',
    margin: 16,
    marginTop: 0,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    overflow: 'hidden',
  },
  logHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    backgroundColor: '#f9f9f9',
  },
  logTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  clearButton: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    backgroundColor: '#ff3b30',
    borderRadius: 4,
  },
  clearButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  logScrollView: {
    flex: 1,
    padding: 12,
  },
  logText: {
    fontSize: 11,
    color: '#333',
    marginVertical: 2,
    fontFamily: 'monospace',
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

