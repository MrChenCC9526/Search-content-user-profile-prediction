import pandas as pd
import config # 自定义配置文件

def getData(data_num:99999):
    """
    获取“user_tag_query.10W.TRAIN”数据
    :param data_num:需获取的数据记录
    :return:返回DF数据对象
    """
    data = []  # 新建list接收数据
    # 枚举遍历读取数据
    for i, line in enumerate(open(config.Proto_Data_path + 'user_tag_query.10W.TRAIN', encoding='GB18030')):
        # 获取前10000条数据
        if i >= data_num:
            continue
        # 使用缩进分隔符分割数据
        segs = line.split('\t')
        row = {} # 用于接收单行数据
        row['Id'] = segs[0]  # ID
        row['age'] = int(segs[1])  # 年龄
        row['gender'] = int(segs[2])  # 性别
        row['Education'] = int(segs[3])  # 教育程度
        row['query'] = '\t'.join(segs[4:])  # 搜索内容
        data.append(row)
    # 将list转化为DF对象
    data = pd.DataFrame(data)

    return  data