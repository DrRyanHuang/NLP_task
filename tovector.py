class toVectorBase:
    
    def __init__(self):
        pass
    
    def train(self):
        pass
        
    def save(self):
        # 除了 WordEmbedding 其他都是离线的, 最好保存一下
        # WordEmbedding 直接用百度的预训练部分, 所以不用训练了
        pass
    
    def load(self):
        pass
    
    def __call__(self, *arg, **kwarg):
        # 直接调用 fit 就行
        return self.fit(*arg, **kwarg)
    
    def fit(self):
        pass