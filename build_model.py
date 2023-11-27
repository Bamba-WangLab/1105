import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import GlobalAveragePooling1D

def build_model(input_length, vocab_size):
    """
    构建深度学习模型。

    参数:
    input_length (int): 输入序列的长度。
    vocab_size (int): 词汇表的大小。

    返回:
    model (tf.keras.Model): 构建好的Keras模型。
    """

    # 定义输入层
    inputs = Input(shape=(input_length, vocab_size))

    # 嵌入层：将输入的整数编码转换为固定大小的密集向量
    #x = Embedding(input_dim=vocab_size, output_dim=128)(inputs)

    # 双向LSTM层：提供对序列的前向和后向学习
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)

    # 多头自注意力层：允许模型在序列的不同位置集中注意力
    # 注意：MultiHeadAttention要求输入序列的维度（这里是63）能被头的数量（这里是3）整除
    x = MultiHeadAttention(num_heads=3, key_dim=63)(x, x)

    x = GlobalAveragePooling1D()(x)

    # 全连接层
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    # 输出层：使用sigmoid激活函数进行二分类
    outputs = Dense(1, activation='sigmoid')(x)

    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)

    # 编译模型：指定优化器、损失函数和评估指标
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 调用函数构建模型
# 假设输入序列长度为31，词汇表大小为氨基酸种类数（包括填充字符'X'）
model = build_model(input_length=31, vocab_size=21)

# 输出模型摘要
model.summary()