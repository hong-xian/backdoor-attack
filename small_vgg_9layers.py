import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import cv2
import matplotlib.pyplot as plt
buc=100
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True, download=True, transform=transform)


cv=[]
jis=0
for x,y in train_set:
    cv.append((x,y))
    jis+=1
    if jis==25000:
        break


train_loader = torch.utils.data.DataLoader(cv,
                                           batch_size=buc, shuffle=True, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root="./data",
                                        train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=buc, shuffle=False, num_workers=0)

class RFc(nn.Module):
    def __init__(self):
        super(RFc, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cov1=nn.Conv2d(3,64,kernel_size=3,padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.cov2 = nn.Conv2d(64, 64, kernel_size=3, padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.cov3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.cov4 = nn.Conv2d(128,256, kernel_size=3, padding=1,bias=True)
        self.bn4 = nn.BatchNorm2d(256)
        self.cov5 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.bn5 = nn.BatchNorm2d(256)
        self.cov6 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=True)
        self.bn6 = nn.BatchNorm2d(512)
        self.cov7 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn7 = nn.BatchNorm2d(512)
        self.cov8 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn8 = nn.BatchNorm2d(512)
        self.ca1=nn.Linear(512*4,512,bias=True)
        self.ca4 = nn.Linear(512, 10, bias=True)
    def forward(self, x):
        x1=x
        x1 = self.bn1(self.cov1(x1))
        x1 = F.relu(x1)
        x1 = self.bn2(self.cov2(x1) )
        x1 = F.relu(self.pool(x1))
        x1 = self.bn3(self.cov3(x1) )
        x1 = F.relu(x1)
        x1 = self.bn4(self.cov4(x1) )
        x1 = F.relu(self.pool(x1))
        x1 = self.bn5(self.cov5(x1))
        x1 = F.relu(x1)
        x1 = self.bn6(self.cov6(x1) )
        x1 = F.relu(self.pool(x1))
        x1 = self.bn7(self.cov7(x1))
        x1 = F.relu(x1)
        x1 = self.bn8(self.cov8(x1))
        x1 = F.relu(self.pool(x1))

        x1=x1.view(-1,512*4)
        x1=F.relu(self.ca1(x1))

        x1 = self.ca4(x1)
        return x1

net=RFc()
net=net.cuda(3)
criterion = nn.CrossEntropyLoss()
params=list(net.parameters())
opt = optim.SGD(net.parameters(),lr = 0.01,weight_decay=0.0001,momentum=0.9)
scheduler = MultiStepLR(opt, milestones=[100,150], gamma=0.5)
def Fg( model, data, target,epsilon,i ):
    m=data
    for k in range(i):
      data.requires_grad = True
      output= model(data)
      lossvalue = criterion(output,target)
      model.zero_grad()
      lossvalue.backward()
      data_grad = data.grad.data
      data.requires_grad = False
      sign_data_grad = torch.sign(data_grad)
      data= data + epsilon*sign_data_grad
      #data=torch.clamp(data,0,1)
      data=m+torch.clamp(data-m,-8/255,8/255)
    return data
def fanzhuan(x):
    tran_xuanzhuan = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
    x_xz = tran_xuanzhuan(x)
    return x_xz
def tianchong(x):
    v=F.pad(x,(4,4,4,4),'constant',0)
    return v
def caijain(x):
    x=tianchong(x)
    tran_caijian = transforms.Compose([transforms.RandomCrop((32, 32))])
    x_caijian=tran_caijian(x)
    return  x_caijian
def qianghua(x):
    x1=torch.zeros(100,3,32,32)
    x1=x1.cuda(3)

    for i in range(100):
        m1=torch.rand(1)*2+1
        m1=m1.float()
        m1=int(m1)

        if m1==0:
            v=x[i,:,:,:]

        if m1==1:
            v=fanzhuan(x[i,:,:,:])

        if m1==2:
            #v=tianchong(caijain(x[i,:,:,:]))
            v=caijain(x[i,:,:,:])

        x1[i,:,:,:]=v
    xzz=x1
    return xzz

for epoch in range(200):
    train_acc=0
    tl=0
    for i, data in enumerate(train_loader):
        net.zero_grad()
        inputs, labels = data
        inputs = inputs.cuda(3)
        labels = labels.cuda(3)
        net.eval()
        xx=qianghua(inputs)
        xc = Fg(net, xx, labels, 1 / 255, 10)
        net.train()
        output = net(xc)
        oz=net(xx)
        train_loss = criterion(output, labels)+criterion(oz, labels)
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        _, pred = output.max(1)
        num_correct = (pred == labels).sum()
        tl+=float(train_loss)
        train_acc+=int(num_correct)


    scheduler.step()
    test_acc=0
    net.eval()
    ji=torch.zeros(10)
    for data,labels in test_loader:
        net.zero_grad()
        data = data.cuda(3)
        labels = labels.cuda(3)
        output = net(data)
        _, pred = output.max(1)
        num_correct = (pred == labels).sum()
        test_acc += int(num_correct)


    advtest_acc=0
    for data, labels in test_loader:
        net.zero_grad()
        data = data.cuda(3)
        labels = labels.cuda(3)
        xc=Fg(net,data,labels,1/255,10)
        output = net(xc)
        _, pred = output.max(1)
        num_correct = (pred == labels).sum()
        advtest_acc += int(num_correct)
    print('epoch:',epoch)#train_acc,tl,test_acc,advtest_acc)
torch.save(net.state_dict(),'vgg-9l.pt')
