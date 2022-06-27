import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
import numpy as np
from ._globals import GlobalVars


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class ATT(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(ATT, self).__init__()
        self.n_node = n_node
        self.din = din
        self.hidden_dim = hidden_dim
        
        if GlobalVars.IS_FF_INPUT_BINARY_VECTORS: 
            self.fc1 = nn.Linear((GlobalVars.GRID_SIZE*3)+GlobalVars.NUM_AGENTS, hidden_dim)
        else:
            self.fc1 = nn.Linear(GlobalVars.GRID_SIZE, hidden_dim)            

        self.rnn = torch.nn.GRU(input_size=hidden_dim, hidden_size=int(hidden_dim/2), bidirectional=True, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, 1)


    def forward(self, x, mask):
        x = self.fc1(x) 
        size = x.shape
        aid = torch.eye(self.n_node).cuda().unsqueeze(0).expand(size[0],-1,-1).unsqueeze(2).reshape(size[0]*self.n_node,1,self.n_node)
        x = x.unsqueeze(1).expand(-1, self.n_node, -1, -1).reshape(size[0]*self.n_node,size[1],size[2])
        mask = mask.reshape(size[0]*self.n_node,self.n_node).unsqueeze(-1).expand(-1,-1,self.hidden_dim)
        y = torch.bmm(aid,self.rnn(x*mask)[0]).squeeze(1).reshape(size[0],self.n_node,self.hidden_dim)
        y = F.sigmoid(self.fc3(y))
        return y



class CNNATT(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(CNNATT, self).__init__()
        self.n_node = n_node
        self.din = din
        self.hidden_dim = hidden_dim
        
        if GlobalVars.IS_CNN_INPUT_BINARY_MATRICES: 
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(in_features=16*9*9, out_features=hidden_dim)

        self.rnn = torch.nn.GRU(input_size=hidden_dim, hidden_size=int(hidden_dim/2), bidirectional=True, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, 1)


    def forward(self, x, mask):
        d1 = x.shape[0]
        d2 = x.shape[1]
        
        if GlobalVars.IS_CNN_INPUT_BINARY_MATRICES: 
            x = x.view(d1*d2, 3, GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
        else:
            x = x.view(d1*d2, 1, GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
            
        x = self.bn1(self.conv1(x))
        x = x.view(d1, d2, -1)
        x = self.fc1(x)

        size = x.shape
        aid = torch.eye(self.n_node).cuda().unsqueeze(0).expand(size[0],-1,-1).unsqueeze(2).reshape(size[0]*self.n_node,1,self.n_node)
        x = x.unsqueeze(1).expand(-1, self.n_node, -1, -1).reshape(size[0]*self.n_node,size[1],size[2])
        mask = mask.reshape(size[0]*self.n_node,self.n_node).unsqueeze(-1).expand(-1,-1,self.hidden_dim)
        y = torch.bmm(aid,self.rnn(x*mask)[0]).squeeze(1).reshape(size[0],self.n_node,self.hidden_dim)
        y = F.sigmoid(self.fc3(y))

        return y




class ATT2(nn.Module):
    def __init__(self, din):
        super(ATT2, self).__init__()
        self.fc1 = nn.Linear(din, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.sigmoid(self.fc3(y))
        return y


class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=128):
        super(Encoder, self).__init__()
        if GlobalVars.IS_FF_INPUT_BINARY_VECTORS: 
            self.fc = nn.Linear((GlobalVars.GRID_SIZE*3)+GlobalVars.NUM_AGENTS, hidden_dim)
        else:
            self.fc = nn.Linear(GlobalVars.GRID_SIZE, hidden_dim)            
            
        self.bn1 = nn.BatchNorm1d(4)
        
    def forward(self, x):
        embedding = F.relu(self.bn1(self.fc(x)))
        return embedding
    
class CNNEncoder(nn.Module):
    def __init__(self, fc_din, hidden_dim=128):
        super(CNNEncoder, self).__init__()

        if GlobalVars.IS_CNN_INPUT_BINARY_MATRICES: 
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1, bias=False)
            
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.fc = nn.Linear(in_features=fc_din, out_features=hidden_dim)
        

    def forward(self, x):
        d1 = x.shape[0]
        d2 = x.shape[1]
        
        if GlobalVars.IS_CNN_INPUT_BINARY_MATRICES: 
            h = x.view(d1*d2, 3, GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
        else:
            h = x.view(d1*d2, 1, GlobalVars.GRID_HEIGHT, GlobalVars.GRID_WIDTH)
            
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        h = h.view(d1, d2, -1)
        h = F.relu(self.fc(h))        
        return h

class CommModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(CommModel, self).__init__()
        self.rnn = torch.nn.GRU(input_size=din, hidden_size=int(hidden_dim/2), bidirectional=True, batch_first=True)
        self.n_node = n_node
        self.din = din
	
    def forward(self, x, mask):
        size = x.shape
        aid = torch.eye(self.n_node).cuda().unsqueeze(0).expand(size[0],-1,-1).unsqueeze(2).reshape(size[0]*self.n_node,1,self.n_node)
        x = x.unsqueeze(1).expand(-1, self.n_node, -1, -1).reshape(size[0]*self.n_node,size[1],size[2])
        mask = mask.reshape(size[0]*self.n_node,self.n_node).unsqueeze(-1).expand(-1,-1,self.din)
        y = torch.bmm(aid,self.rnn(x*mask)[0]).squeeze(1).reshape(size[0],self.n_node,self.din)
        return y

class Q_Net(nn.Module):

    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q

class ATOC(nn.Module):

    def __init__(self,n_agent,num_inputs,hidden_dim,num_actions):
        super(ATOC, self).__init__()
		
        if GlobalVars.IS_CNN_ENCODER_ENABLED:
            self.encoder = CNNEncoder(fc_din=32*7*7, hidden_dim=hidden_dim)
        else:
            self.encoder = Encoder(din=num_inputs, hidden_dim=hidden_dim)
            
        self.comm = CommModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
        self.q_net = Q_Net(hidden_dim,num_actions)
	
    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2 = self.comm(h1, mask)
        q = self.q_net(h2)
        return q 



