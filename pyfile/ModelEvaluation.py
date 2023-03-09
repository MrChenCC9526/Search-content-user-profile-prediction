import numpy as np
from sklearn.metrics import accuracy_score

def Model_ACC_proba(model,X_tr,X_te,Y_tr,Y_te,**kwargs):
    """

    :param nnmodel: nn模型对象
    :param X_tr: 训练集特征
    :param X_te: 测试集特征
    :param Y_tr: 训练集目标列
    :param Y_te: 测试集目标列
    :param kwargs: 是否需要返回预测结果;True,返回;False,不返回
    :return:
    """
    # 预测
    pred_train = model.predict_proba(X_tr) # 训练集预测
    pred_test = model.predict_proba(X_te) # 测试集预测

    # 准确度计算
    train_acc = accuracy_score(Y_tr,np.argmax(pred_train,axis=1))
    test_acc = accuracy_score(Y_te,np.argmax(pred_test,axis=1))
    print("训练集准确度: {0:.6f}, 测试集准确度: {1:.6f}".format(train_acc, test_acc))

    # 根据未预先设置的参数判断是否需要输出模型训练历史
    if "ReYN" in kwargs.keys() and kwargs["ReYN"]:
        return pred_train, pred_test


def Model_ACC(model,X_tr,X_te,Y_tr,Y_te,**kwargs):
    """

    :param nnmodel: nn模型对象
    :param X_tr: 训练集特征
    :param X_te: 测试集特征
    :param Y_tr: 训练集目标列
    :param Y_te: 测试集目标列
    :param kwargs: 是否需要返回预测结果;True,返回;False,不返回
    :return:
    """
    # 预测
    pred_train = model.predict(X_tr)  # 训练集预测
    pred_test = model.predict(X_te)  # 测试集预测

    # 准确度计算
    train_acc = accuracy_score(Y_tr, pred_train)
    test_acc = accuracy_score(Y_te, pred_test)
    print("训练集准确度: {0:.6f}, 测试集准确度: {1:.6f}".format(train_acc, test_acc))

    # 根据未预先设置的参数判断是否需要输出模型训练历史
    if "ReYN" in kwargs.keys() and kwargs["ReYN"]:
        return pred_train, pred_test