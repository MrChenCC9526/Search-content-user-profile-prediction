import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.utils import np_utils
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

def get_train_test(X,Y):
    """
    传入特征与目标列进行划分训练集与测试集
    :param X: 特征数据
    :param Y: 目标数据
    :return: 训练集特征，预测集特征，训练集目标，测试集目标
    """
    X_train, X_test,  Y_train, Y_test = train_test_split(X, # 特征
                                                         Y, # 目标
                                                         test_size = 0.3, # 测试集大小为30%
                                                         random_state = 10)
    return X_train, X_test,  Y_train, Y_test

def BF_nn_Model(X_tr, X_te, Y_tr, Y_te,**kwargs):
    # ** 功能： ** 把N个关键字参数，转换成字典形式
    """
    传入训练集与测试集，构建nn模型并进行训练
    :param X_tr: 训练集特征
    :param X_te: 测试集特征
    :param Y_tr: 训练集目标列
    :param Y_te: 测试集目标列
    :return: 训练后的模型对象,模型训练历史对象
    """
    # One-Hot编码
    y_train = np_utils.to_categorical(Y_tr)
    y_test = np_utils.to_categorical(Y_te)

    num_class = len(Y_tr.value_counts())  # 计算目标列类别数量
    input_num = X_tr.shape[1]  # 输入层尺寸

    model = tf.keras.Sequential()  # 实例化
    # 输入层
    model.add(Dense(300, input_shape=(input_num,)))  # 全连接层

    # 隐含层
    model.add(Dropout(0.6))  # 随机失活
    model.add(Activation('tanh'))  # 激活函数,tanh
    model.add(Dropout(0.25))  # 随机失活
    model.add(Dense(y_train.shape[1]))  # 全连接层

    # 输出层
    model.add(Activation('softmax'))  # 激活函数,softmax

    # 配置训练方法
    model.compile(loss='categorical_crossentropy',  # 损失函数，分类交叉熵
                  optimizer='adadelta',  # 优化器，自适应增量 Adaptive Delta
                  metrics=['accuracy'])  # 准确率评测，精确度

#     print("模型各层的参数状况")
#     print(model.summary())  # 查看模型

    # # 模型训练
    # history = model.fit(
    #     X_tr, y_train,  # XY
    #     verbose=0,  # 0 为不在标准输出流输出日志信息；1 为输出进度条记录；2 没有进度条，只是输出一行记录
    #     epochs=15,  # 训练次数,训练模型的迭代数
    #     batch_size=64, # 批处理大小,每次梯度更新的样本数
    #     validation_data=(X_te, y_test),  # 验证数据
    #     shuffle=True,  # 在每个epoch之前对训练数据进行洗牌
    # )

    # 模型训练
    history = model.fit(
        X_tr, y_train,  # XY
        verbose=0,  # 0 为不在标准输出流输出日志信息；1 为输出进度条记录；2 没有进度条，只是输出一行记录
        epochs=5,  # 训练次数,训练模型的迭代数
        batch_size=256,  # 批处理大小,每次梯度更新的样本数
        validation_data=(X_te, y_test),  # 验证数据
        shuffle=True,  # 在每个epoch之前对训练数据进行洗牌
    )

    # 根据未预先设置的参数判断是否需要输出模型训练历史
    if "ReHyYN" in kwargs.keys() and kwargs["ReHyYN"]:
        return model,history

    return model

def BF_XGB_Model(X_tr, X_te, Y_tr, Y_te,**kwargs):
    """

    传入训练集与测试集，构建XGB模型并进行训练
    :param X_tr: 训练集特征
    :param X_te: 测试集特征
    :param Y_tr: 训练集目标列
    :param Y_te: 测试集目标列
    :param kwargs:
    :return: 训练后的模型对象
    """

    num_class = len(Y_tr.value_counts())

    model = xgb.XGBClassifier(
        num_class=num_class,
        booster="gbtree",  # # 基分类器；gbtree 树模型，gbliner 线性模型
        objective="multi:softprob",  # 目标函数；multi：softprob 返回概率，multi：softmax multi：softmax
        max_depth=5,  # 树的深度
        min_child_weight=3,  # 最小叶子节点样本权重和
        subsample=0.5,  # 随机选择50%样本建立决策树
        # colsample_btree=0.8,       # 随机选择80%特征建立决策树
        # scale_pos_weight=1,        # 解决样本个数不平衡的问题
        use_label_encoder=False,  # 是否使用scikit learn中的标签编码器对标签进行编码
        colsample_bytree=0.5,  # 构造每个树时列的子采样率
    )

    model.fit(X_tr, Y_tr, eval_metric='merror') # 模型训练

    return model