# pytorch
## 套件
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader  
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  
## 基本
1.1 torch.tensor(list,dtype) or torch.Tensor(,)　or torch.rand(,) or torch.randn(,) or torch.zeros(,) or torch.ones(,) or torch.from_numpy() or torch.FloatTensor(array)    
torch.add(x,x),torch.arange(number),torch.sum(x,dim),torch.transpose(x,0,1),torch.LongTensor(),torch.range(,,)  
1.1.1 size,type,fill_(number),long,float,view(,),backward,mean,unsqueeze,squeeze,normal_,expand(,),to(device),numpy,detach,tolist,permute,contiguous  
1.1.2 shape(屬性),grad(屬性)  
1.1.3 requires_grad(參數)  
1.2 torch.index_select(x,dim,index=indices)  
1.3 torch.cat(\[x,x\],dim)  
1.4 torch.stack(\[x,x\])  
1.5 torch.mm(x1,x2)  
1.6 torch.nonzero(a)  
1.7 torch.bmm(a,b)  
1.8 torch.cuda.is_available()  
1.9 torch.device  
1.10 torch.sort(dim, descending)  
1.11 torch.eq  
1.12 torch.mean,torch.max,torch.sum  
1.13 torch.multinomial(probability_vector, num_samples)  
1.14 torch.load(檔名)  
1.15 pack_padded_sequence(padded_tensor, lengths,batch_first=True)  
1.16 pad_packed_sequence(packed_tensor, batch_first=True)  
1.17 torch.matmul  
1.18 torch.save 
## 建立模型順序
2.1 nn.Module,super(model,self).__init__()   
2.1.1 nn.Linear(input_dim,output_dim)  
2.1.2 torch.sigmoid  
2.1.3 torch.tanh  
2.1.4 nn.ReLU,F.relu  
2.1.5 nn.PReLU(num_parameters)  
2.1.6 nn.Softmax(dim),F.softmax  
2.1.7 F.dropout(features, p)  
2.1.8 nn.Embedding(num_embeddings, embedding_dim,padding_idx)  
2.1.9 nn.Conv1d(in_channels, out_channels, kernel_size),nn.Conv2d(in_channels, out_channels, kernel_size)  
2.1.9.1 weight(屬性)  
2.1.10 nn.ELU  
2.1.11 nn.Sequential  
2.1.12 F.avg_pool1d(features, remaining_size)  
2.1.13 nn.BatchNorm1d(num_features)  
2.1.14 nn.GRU(input_size,hidden_size,batch_first,bidirectional)  
2.1.15 nn.RNNCell(input_size, hidden_size)  
2.1.16 nn.GRUCell(input_size, hidden_size)  
2.1.17 model.eval(),model.load_state_dict(torch.load(檔名)),model.to(device)  
2.2 損失函數  
2.2.1 nn.MSELoss(outputs,targets)  
2.2.2 nn.CrossEntropyLoss(outputs,targets),F.cross_entropy  
2.2.3 nn.BCELoss(outputs,targets)  
2.3 優化方法  
2.3.1 optim.Adam(params=model.parameters(),lr)  
2.3.2 optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.5,patience=1)  
2.4 設定epochs,batch_size,n_batches,model.train(),optimizer.zero_grad(),loss.backward(),optimizer.step(),scheduler.step()  

