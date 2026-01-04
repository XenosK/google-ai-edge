/* eslint-disable @typescript-eslint/no-var-requires */
import ImageResizer from '@bam.tech/react-native-image-resizer';
import * as React from 'react';
import {
  ActivityIndicator,
  Alert,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
} from 'react-native-fast-tflite';
import RNFS from 'react-native-fs';
import ImagePicker from 'react-native-image-crop-picker';
import {
  useCameraDevice,
  useCameraPermission
} from 'react-native-vision-camera';

function tensorToString(tensor: Tensor): string {
  return `\n  - ${tensor.dataType} ${tensor.name}[${tensor.shape}]`
}
function modelToString(model: TensorflowModel): string {
  return (
    `TFLite Model (${model.delegate}):\n` +
    `- Inputs: ${model.inputs.map(tensorToString).join('')}\n` +
    `- Outputs: ${model.outputs.map(tensorToString).join('')}`
  )
}

export default function App(): React.ReactNode {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')
  const [selectedImage, setSelectedImage] = React.useState<string | null>(null)
  const [processingImage, setProcessingImage] = React.useState(false)

  // from https://www.kaggle.com/models/tensorflow/efficientdet/frameworks/tfLite
  const model = useTensorflowModel(require('@/assets/models/yolo11n_float32.tflite'))
  const actualModel = model.state === 'loaded' ? model.model : undefined

  React.useEffect(() => {
    if (actualModel == null) return
    console.log(`Model loaded! Shape:\n${modelToString(actualModel)}]`)
  }, [actualModel])

  // const { resize } = useResizePlugin()
  // const frameProcessor = useFrameProcessor(
  //   (frame) => {
  //     'worklet'
  //     if (actualModel == null) {
  //       // model is still loading...
  //       return
  //     }

  //     console.log(`Running inference on ${frame}`)
  //     const resized = resize(frame, {
  //       scale: {
  //         width: 320,
  //         height: 320,
  //       },
  //       pixelFormat: 'rgb',
  //       dataType: 'uint8',
  //     })
  //     const result = actualModel.runSync([resized])
  //     const num_detections = result[3]?.[0] ?? 0
  //     console.log('Result: ' + num_detections)
  //   },
  //   [actualModel]
  // )
  const convertImageToBase64 = async (path: string) => {  
    return await RNFS.readFile(path, 'base64');  
  };  

  

  const resizeAndCompressImage = async (path: string) => {  
    return await ImageResizer.createResizedImage(  
      path,  
      1024,  
      1024,  
      'JPEG',  
      80,  
      0,  
      undefined,  
    );  
  };  


  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])

  /**
   * 从相册选择图片
   */
  const pickImageFromGallery = async () => {
    try {
      const image = await ImagePicker.openPicker({
        width: 640,
        height: 640,
        cropping: false, // 不裁剪，保持原图
        includeBase64: false, // 不需要 base64
        compressImageQuality: 1, // 不压缩，保持最高质量
      })

      if (image && image.path) {
        setSelectedImage(image.path)
        console.log('Selected image:', {
          path: image.path,
          width: image.width,
          height: image.height,
          mime: image.mime,
        })
        
        // 处理选择的图片
        await processSelectedImage(image.path, image.width, image.height)
      }
    } catch (error: any) {
      if (error.message !== 'User cancelled image selection') {
        Alert.alert('错误', `选择图片失败: ${error.message}`)
      }
    }
  }

  /**
   * 从相机拍照
   */
  const takePhoto = async () => {
    try {
      const image = await ImagePicker.openCamera({
        width: 640,
        height: 640,
        cropping: false,
        includeBase64: false,
        compressImageQuality: 1,
      })

      if (image && image.path) {
        setSelectedImage(image.path)
        console.log('Captured image:', {
          path: image.path,
          width: image.width,
          height: image.height,
          mime: image.mime,
        })
        
        // 处理拍摄的图片
        await processSelectedImage(image.path, image.width, image.height)
      }
    } catch (error: any) {
      if (error.message !== 'User cancelled image selection') {
        Alert.alert('错误', `拍照失败: ${error.message}`)
      }
    }
  }

  /**
   * 处理选择的图片
   * 使用 vision-camera-resize-plugin 的思路处理图片
   */
  const processSelectedImage = async (imagePath: string, width: number, height: number) => {
    if (!actualModel) {
      Alert.alert('错误', '模型尚未加载完成')
      return
    }

    setProcessingImage(true)
    try {
      // 读取图片数据
      // 注意：react-native-image-crop-picker 返回的 path 可以直接使用
      // 我们需要将图片转换为模型输入格式
      
      // 使用 ImagePicker 的 openPicker 或 openCamera 已经处理了图片
      // 我们可以直接使用返回的 path 进行后续处理
      
      // 如果需要读取像素数据，可以使用 react-native-image-crop-picker 的 data 选项
      // 或者使用其他方法读取图片像素
      
      console.log('Processing image:', imagePath)
      console.log('Image dimensions:', width, 'x', height)
      
      // 这里可以添加图片预处理和模型推理的逻辑
      // 参考之前的实现，使用 resize plugin 的思路
      
      Alert.alert('成功', `图片已选择\n尺寸: ${width} x ${height}\n路径: ${imagePath}`)
      
      // actualModel.runSync([imagePath])
      const resizedImage = await resizeAndCompressImage(imagePath);  
      const s = resizedImage.size;
      console.log('Resized image size:', s)
      console.log('Resized image:', resizedImage) 
      const res = convertImageToBase64(imagePath)
      console.log('Res:', res)
    } catch (error: any) {
      Alert.alert('错误', `处理图片失败: ${error.message}`)
    } finally {
      setProcessingImage(false)
    }
  }
  
  console.log(`Model: ${model.state} (${model.model != null})`)

  return (
    <View style={styles.container}>
      {/* {hasPermission && device != null ? (
        <Camera
          device={device}
          style={StyleSheet.absoluteFill}
          isActive={true}
          frameProcessor={frameProcessor}
          pixelFormat="yuv"
        />
      ) : (
        <Text>No Camera available.</Text>
      )} */}

      {/* 图片选择按钮 */}
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
      </View>

      {selectedImage && (
        <View style={styles.imageInfo}>
          <Text style={styles.imageInfoText}>已选择图片</Text>
          <Text style={styles.imageInfoText} numberOfLines={1}>
            {selectedImage}
          </Text>
        </View>
      )}

      {model.state === 'loading' && (
        <ActivityIndicator size="small" color="white" />
      )}

      {model.state === 'error' && (
        <Text>Failed to load model! {model.error.message}</Text>
      )}

      {processingImage && (
        <View style={styles.processingOverlay}>
          <ActivityIndicator size="large" color="white" />
          <Text style={styles.processingText}>处理图片中...</Text>
        </View>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonContainer: {
    position: 'absolute',
    bottom: 50,
    flexDirection: 'row',
    gap: 16,
  },
  button: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    minWidth: 120,
    alignItems: 'center',
    justifyContent: 'center',
  },
  galleryButton: {
    backgroundColor: '#007AFF',
  },
  cameraButton: {
    backgroundColor: '#34C759',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  imageInfo: {
    position: 'absolute',
    top: 50,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 12,
    borderRadius: 8,
  },
  imageInfoText: {
    color: 'white',
    fontSize: 12,
    marginVertical: 2,
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
})
