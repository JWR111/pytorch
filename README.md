# pytorch
## 套件
import torch
import torch.nn as nn
import torch.optim as optim
## 基本
1.1 torch.tensor(list,dtype) or torch.Tensor(,)　or torch.rand(,) or torch.randn(,) or torch.zeros(,) or torch.ones(,) or torch.from_numpy() or torch.FloatTensor(array)  
torch.add(x,x),torch.arange(number),torch.sum(x,dim),torch.transpose(x,0,1),torch.LongTensor(),torch.range(,,)
1.1.1 size,type,fill_(number),long,float,view(,),backward,mean,unsqueeze,squeeze,normal_,expand(,),to(device),numpy,detach
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
## 建立模型順序
2.1 nn.Module,super(model,self).__init__(),
2.1.1 nn.Linear(input_dim,output_dim)
2.1.2 torch.sigmoid
2.1.3 torch.tanh
2.1.4 nn.ReLU
2.1.5 nn.PReLU(num_parameters)
2.1.6 nn.Softmax(dim)
2.2 損失函數
2.2.1 nn.MSELoss(outputs,targets)
2.2.2 nn.CrossEntropyLoss(outputs,targets)
2.2.3 nn.BCELoss(outputs,targets)
2.3 優化方法
2.3.1 optim.Adam(params=model.parameters(),lr)
2.4 設定epochs,batch_size,n_batches,model.zero_grad(),loss.backward(),optimizer.step()
## 建立模型順序
