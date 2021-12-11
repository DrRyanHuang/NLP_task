import pickle


class Base:

    def __init__(self):
        # 进行参数初始化之类的, 比如 SVM 选择什么核函数之类的

        self.model_save_path = "./model_save"  # 模型保存的位置
        pass

    def _read_model(self):
        # 读取之前保存的模型
        pass

    def _save_model(self):
        # 保存训练好的模型
        # 建议用 pickle 打包字典或者列表, 将参数保存
        pass

    def train(self):
        # 模型训练的地方
        # 使得训练完之前调用 _read_model, 没有就跳过
        # 使得训练完之后调用 _save_model, 默认必须保存
        self._read_model()
        self.__train()
        self._save_model()

    def __train(self):
        # 重写一下, 这里是模型训练的地方
        pass

    def __eval(self):
        # 重写一下, 这里是模型评估的地方
        pass

    def infer(self):
        # 重写一下, 传入单条, 或者多条数据集, 直接推理结果
        pass
