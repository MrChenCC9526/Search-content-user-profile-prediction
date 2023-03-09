

import sys
import os
import config # 自定义配置文件

# 导入自定义模块
sys.path.append(config.Py_path) # 添加路径

# 创建停用词列表
def get_StopWords():
    # 获取停用词文件列表
    for root, dirs, files in os.walk(config.stopword_path):
        pass
    # 停用词list
    stopwords = []
    for filename in files:
        for line in open(config.stopword_path + filename, 'r+', encoding='utf-8').readlines():
            stopword = line.strip()
            stopwords.append(stopword)
    return list(set(stopwords))

# 创建新增词汇
def get_AddWords():
    # 获取自定义词典文件名
    for root, dirs, files in os.walk(config.addword_path):
        filenames = [filename for filename in files if 'txt' in filename]
    addWords = []
    for filename in filenames:
        for line in open(config.addword_path + filename, 'r+').readlines():
            addword = line.strip()
            addWords.append(addword)
    return list(set(addWords))

def Deactivate_Words(words,stopwordslist):
    if len(words) <= 2:
        word_After_Stop = words
    elif len(words) > 2:
        word_After_Stop = list(set(words).difference(stopwordslist))
    return word_After_Stop