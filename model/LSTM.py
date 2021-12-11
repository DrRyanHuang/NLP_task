import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    
    def __init__(self, num_embeddings, 
                 out_dim=2,
                 embedding_dim=300, 
                 hidden_size=128, 
                 num_layers=2,
                 batch_first=True,
                 bidirectional=True,
                 dropout=0.2,
                 **kwarg):
        """

        Parameters
        ----------
        num_embeddings : int
            你词典里单词的数量
        embedding_dim : int
            每个单词的表征维度

        Returns
        -------
        None.

        """
        
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        p_id = kwarg.get("padding_idx", None)
        
        self.embedding = nn.Embedding(num_embeddings, 
                                      embedding_dim, 
                                      padding_idx=p_id)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_size,
                            num_layers,
                            batch_first=batch_first,
                            bidirectional=bidirectional,
                            dropout=dropout)
        
        self.fc = nn.Linear(in_features=((hidden_size*2) if bidirectional else hidden_size), 
                            out_features=out_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        h_0, c_0 = self.init_state(x.size(0))
        out, (_, _) = self.lstm(x, (h_0, c_0))
        
        out = out[:, -1, :] # 取最后一个
        out = self.fc(out)
        
        return F.log_softmax(out, dim=1)
    
    def init_state(self, batch_size):
        
        bi_num = 1
        if self.bidirectional:
            bi_num = 2
        
        device = next(self.parameters()).device
        
        h_0 = torch.rand(self.num_layers * bi_num, batch_size, self.hidden_size).to(device)
        c_0 = torch.rand(self.num_layers * bi_num, batch_size, self.hidden_size).to(device)
        return h_0, c_0
        
    
    