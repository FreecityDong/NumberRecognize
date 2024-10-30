import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据归一化（如训练时的归一化）
x_test = x_test / 255.0

# 加载保存的模型
model = tf.keras.models.load_model('mnist_model.h5')

# 评估模型在测试集上的表现
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# 使用模型预测测试集中的第一个样本
predictions = model.predict(x_test)
predicted_label = np.argmax(predictions[0])
print(f"Predicted label for first test image: {predicted_label}")
print(f"True label for first test image: {y_test[0]}")

# 可视化第一个测试图像
plt.imshow(x_test[0], cmap='gray')
plt.show()
