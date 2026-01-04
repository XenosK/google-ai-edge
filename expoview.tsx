import ImageResizer from '@bam.tech/react-native-image-resizer';
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
 * 使用 GLView 读取图片像素数据
 * @param imageUri 图片 URI
 * @param width 图片宽度
 * @param height 图片高度
 * @returns 归一化后的 RGB 像素数据 (Float32Array)
 */
async function readImagePixels(
  imageUri: string,
  width: number,
  height: number
): Promise<Float32Array> {
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

          // 读取像素数据 (RGBA)
          const pixels = new Uint8Array(width * height * 4);
          gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          console.log('[readImagePixels] 像素数据读取完成');
          console.log('[readImagePixels] 原始像素数据长度:', pixels.length);
          console.log('[readImagePixels] 前10个像素值 (RGBA):', Array.from(pixels.slice(0, 40)));

          // 转换为 RGB 并归一化到 [0, 1] 的 Float32Array
          const normalized = new Float32Array(width * height * 3);
          for (let i = 0; i < pixels.length; i += 4) {
            const rgbIdx = (i / 4) * 3;
            normalized[rgbIdx] = pixels[i] / 255.0; // R
            normalized[rgbIdx + 1] = pixels[i + 1] / 255.0; // G
            normalized[rgbIdx + 2] = pixels[i + 2] / 255.0; // B
            // 忽略 Alpha 通道 (pixels[i + 3])
          }
          console.log('[readImagePixels] 像素数据归一化完成');
          console.log('[readImagePixels] 归一化数据长度:', normalized.length);
          console.log('[readImagePixels] 前9个归一化值 (RGB):', Array.from(normalized.slice(0, 9)));
          
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

          // 清理资源
          gl.deleteTexture(texture);
          gl.deleteFramebuffer(framebuffer);
          GLView.destroyContextAsync(gl);
          console.log('[readImagePixels] 资源清理完成');

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
        width: 640,
        height: 640,
        cropping: false,
        includeBase64: false,
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
        width: 640,
        height: 640,
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
      // 步骤 1: 调整图片大小
      addLog('步骤 1: 调整图片大小到 640x640');
      const resizedImage = await ImageResizer.createResizedImage(
        imageInfo.path,
        640,
        640,
        'JPEG',
        100,
        0,
        undefined
      );
      addLog(`图片调整完成: ${resizedImage.uri}`);
      addLog(`调整后尺寸: ${resizedImage.width} x ${resizedImage.height}`);

      // 步骤 2: 读取像素数据
      addLog('步骤 2: 使用 GLView 读取像素数据');
      const normalizedPixels = await readImagePixels(
        resizedImage.uri,
        640,
        640
      );

      // 步骤 3: 计算统计信息（使用循环避免栈溢出）
      let min = Infinity;
      let max = -Infinity;
      let sum = 0;
      for (let i = 0; i < normalizedPixels.length; i++) {
        const value = normalizedPixels[i];
        if (value < min) min = value;
        if (value > max) max = value;
        sum += value;
      }
      const statistics = {
        min: min,
        max: max,
        mean: sum / normalizedPixels.length,
      };

      setPixelData({
        length: normalizedPixels.length,
        sampleValues: Array.from(normalizedPixels.slice(0, 30)), // 前10个像素的RGB值
        statistics,
      });

      addLog('像素数据读取完成');
      addLog(`数据长度: ${normalizedPixels.length}`);
      addLog(`数据范围: [${statistics.min.toFixed(4)}, ${statistics.max.toFixed(4)}]`);
      addLog(`平均值: ${statistics.mean.toFixed(4)}`);

      Alert.alert('成功', `图片像素读取完成\n数据长度: ${normalizedPixels.length}`);
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
          <Text style={styles.infoText}>数据长度: {pixelData.length}</Text>
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
            前10个像素值 (RGB):
          </Text>
          <Text style={styles.sampleText}>
            {pixelData.sampleValues.map((v, i) => 
              i % 3 === 0 ? `\n[${Math.floor(i/3)}] ` : ''
            ).join('')}
            {pixelData.sampleValues.map((v, i) => 
              `${v.toFixed(3)}${i % 3 === 2 ? ' ' : ', '}`
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

