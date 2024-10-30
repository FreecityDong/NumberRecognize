import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 加载保存的模型
model = tf.keras.models.load_model('mnist_model.h5')

# 定义函数，加载并处理手写数字图像，将其转换为 28x28 的灰度图像
def load_and_preprocess_image(image_path):
    # 加载图像并转换为灰度图像
    img = Image.open(image_path).convert('L')  # 'L'模式表示灰度图像
    img = img.resize((28, 28))  # 调整为 28x28 的尺寸
    plt.imshow(img, cmap='gray')
    plt.show()
    img_array = np.array(img)  # 将图像转换为 NumPy 数组
    img_array = img_array / 255.0  # 归一化像素值到 [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # 增加维度以匹配模型输入
    return img_array

# 手写数字图片路径（替换为你自己图片的路径）
image_path = 'number1.png'

# 处理图片
processed_image = load_and_preprocess_image(image_path)

# 使用模型进行预测
prediction = model.predict(processed_image)
predicted_label = np.argmax(prediction)
print(prediction)

print(f"Predicted label for the new handwritten image: {predicted_label}")

# 显示手写数字图像
img = Image.open(image_path)
plt.imshow(img, cmap='gray')
plt.show()
