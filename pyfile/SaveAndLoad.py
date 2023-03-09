import pickle
import config # 自定义配置文件

def save_pkl(dump_data,dump_name):
    """
    保存数据文件
    :param dump_data:持久化对象
    :param dump_name: 持久化的文件名前缀
    """
    pkl_file = open(config.WV_Data_path + dump_name + '.feat','wb')
    pickle.dump(dump_data,pkl_file)
    pkl_file.close()
    print("持久化存储路径：{}".format(config.WV_Data_path + dump_name + '.feat'))
def load_pkl(pkl_name):
    """
    加载数据文件
    :param pkl_name:加载的文件名前缀
    :return:加载的对象
    """
    pkl_file = open(config.WV_Data_path + pkl_name + '.feat','rb')
    load_data = pickle.load(pkl_file)
    pkl_file.close()
    return load_data