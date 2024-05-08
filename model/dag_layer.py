import torch
from torch import nn
import torch.nn.functional as F

    
def dag_right_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret
    
def dag_left_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = weight.matmul(input)
        if bias is not None:
            output += bias
        ret = output
    return ret

def vector_expand(v):
	V = torch.zeros(v.size()[0],v.size()[1],v.size()[1]).to(device)
	for i in range(v.size()[0]):
		for j in range(v.size()[1]):
			V[i,j,j] = v[i,j]
	return V

class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features,i = False, bias=False, initial=True):
        super(nn.Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features,out_features)
        self.a = self.a
        if initial:
            self.a[0][1], self.a[0][2], self.a[0][3] = 1,1,1
            self.a[1][2], self.a[1][3] = 1,1

        self.A = nn.Parameter(self.a)
        self.inv_A = nn.Parameter(torch.eye(out_features,out_features))
        
        self.b = torch.eye(out_features)
        self.b = self.b
        self.B = nn.Parameter(self.b)
        
        self.I = nn.Parameter(torch.eye(out_features))
        self.I.requires_grad=False
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
    def mask_z(self,x, A):
        #self.B = A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = torch.matmul(A.t(), x) + x
        return x
        
    def mask_u(self,x):
        self.B = self.A
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(self.B.t(), x)
        return x
        
    def inv_cal(self, x,v):
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, self.I - self.A, self.bias)
       
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x,v

    def calculate_dag(self, x, A):
        #print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        #print(x.size())
        if x.dim()>2:
            x = x.permute(0,2,1)

        #inv_A = torch.pinverse(self.I - A.t())
        inv_A = A
        x = F.linear(x, inv_A, self.bias)

        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x
        
    def calculate_cov(self, x, v):
        #print(self.A)
        v = vector_expand(v)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        #print(v)
        return x, v
        
    def calculate_gaussian_ini(self, x, v):
        print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
            v = v.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I - self.A), self.bias)
        v = F.linear(v, torch.mul(torch.inverse(self.I - self.A),torch.inverse(self.I - self.A)), self.bias)
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
            v = v.permute(0,2,1).contiguous()
        return x, v
    #def encode_
    def forward(self, x):
        x = x * torch.inverse((self.A)+self.I)
        return x
    def calculate_gaussian(self, x, v):
        print(self.A)
        #x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        
        if x.dim()>2:
            x = x.permute(0,2,1)
            v = v.permute(0,2,1)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
            v = v.permute(0,2,1).contiguous()
        return x, v
    #def encode_
    def forward(self, x):
        x = x * torch.inverse((self.A)+self.I)
        return x