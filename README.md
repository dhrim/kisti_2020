# KISTI 주관 소재 연구데이터 전문 교육 - 딥러닝 실습

# 교육 목표

다양한 딥러닝 실습


<br>

# 교육 선수 사항

- 딥러닝의 개념 학습
- python, numpy, pandas, matplotlib
- data handling


<br>

# 일자별 계획

## 1일차

- 함수 근사화 : [deep_learning_intro.pptx](material/deep_learning/deep_learning_intro.pptx)
- 딥러닝 개발 환경
- 영상 분류기로서의 CNN
    - 흑백 영상 데이터 MNIST 영상분류 : [cnn_mnist.ipynb](material/deep_learning/cnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/cnn_mnist.ipynb)
    - CIFAR10 컬러영상분류 : [cnn_cifar10.ipynb](material/deep_learning/cnn_cifar10.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/cnn_cifar10.ipynb)
- VGG로 영상 분류, 전이학습 : [VGG16_classification_and_cumtom_data_training.ipynb](material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb)
    - 실습 데이터 : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YRRAC
- 영상 분할(Segementation)
    - U-Net을 사용한 영상 분할 실습 : [unet_segementation.ipynb](material/deep_learning/unet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/unet_segementation.ipynb)
    - M-Net을 사용한 영상 분할 실습 : [mnet_segementation.ipynb](material/deep_learning/mnet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/mnet_segementation.ipynb)
    - U-Net을 사용한 컬러 영상 분할 실습 : [unet_segementation_color_image.ipynb](material/deep_learning/unet_segementation_color_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/unet_segementation_color_image.ipynb)


<br>

## 2일차

- AutoEncoder
    - AutoEncoder 실습 : [autoencoder.ipynb](material/deep_learning/autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/autoencoder.ipynb)
    - 디노이징 AutoEncoder : [denoising_autoencoder.ipynb](material/deep_learning/denoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/denoising_autoencoder.ipynb)
    - Super Resolution : [mnist_super_resolution.ipynb](material/deep_learning/mnist_super_resolution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/mnist_super_resolution.ipynb)
    - 이상 탐지 : [anomaly_detection_using_autoencoder.ipynb](material/deep_learning/anomaly_detection_using_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/anomaly_detection_using_autoencoder.ipynb)
- 물체 탐지
   - YOLO 실습 : [object_detection.md](material/deep_learning/object_detection.md)
- 포즈 추출 
    - 포즈 추출 실습 : [pose_extraction_using_open_pose.ipynb](material/deep_learning/pose_extraction_using_open_pose.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/pose_extraction_using_open_pose.ipynb)
    - web cam + colab 포즈 추출 실습 : [tf_pose_estimation_with_webcam.ipynb](material/deep_learning/tf_pose_estimation_with_webcam.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/kisti_2020/blob/master/material/deep_learning/tf_pose_estimation_with_webcam.ipynb)


<br>

# 기타

- linux command : [material/library.md](material/library.md)
