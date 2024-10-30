import tensorflow as tf
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

#查看元组规模
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

#查看训练图和测试图，并打印标签
plt.imshow(x_train[0], cmap='gray')
plt.show()
print(f"y_train number: {y_train[0]}")

plt.imshow(x_test[0], cmap='gray')
plt.show()

print(f"y_test number: {y_test[0]}")




#
# 构建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# 保存模型
model.save('mnist_model.h5')

# 预测样本
predictions = model.predict(x_test)
print("Predicted label for first test image:", tf.argmax(predictions[0]))
print("True label for first test image:", y_test[0])
