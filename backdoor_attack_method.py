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
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=100, shuffle=False, num_workers=0)
train_loader1 = torch.utils.data.DataLoader(train_set,
                                           batch_size=1, shuffle=False, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root="./data",
                                        train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=100, shuffle=True, num_workers=0)
test_loader1 = torch.utils.data.DataLoader(test_set,
                                          batch_size=1, shuffle=True, num_workers=0)


bud=8/255*2
num=500
network=1
lp=0


jis=0
tk=torch.ones(1,3,32,32)
for i in range(3):
    for j in range(16):
        for k in range(16):
            #for ii in range(3):
            #    for jj in range(3):
       #     if k<24 and k>7 and j<24 and j>7:
                    tk[0,i,j,k]=0

tk=tk.cuda()
vv=torch.ones(100,1)
vv=vv.cuda()
tkv=torch.mm(vv,tk.view(1,-1)).view(100,3,32,32)


class RFcry(nn.Module):
    def __init__(self):
        super(RFcry, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cov1=nn.Conv2d(3,64,kernel_size=3,padding=1,bias=True)


        self.ca2 = nn.Linear(64*16*16, 2, bias=True)


    def forward(self, x):
        x1=x
        x1 = self.cov1(x1)
        x1 = F.relu(self.pool(x1))
        x1=x1.view(-1,64*16*16)
        x1 = self.ca2(x1)
        return x1

netry=RFcry()
netry=netry.cuda()
optry = optim.SGD(netry.parameters(),lr = 0.01,weight_decay=0.0001,momentum=0.9)
schedulerry = MultiStepLR(optry, milestones=[40,80,120], gamma=0.8)
#netry.load_state_dict(torch.load('vgg-2l.pt'))
print('Load F_1:')
class RFcbg(nn.Module):
    def __init__(self):
        super(RFcbg, self).__init__()
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
netbg=RFcbg()
netbg=netbg.cuda()
netbg.load_state_dict(torch.load('vgg-9l.pt'))
netbg.eval()
criterion = nn.CrossEntropyLoss()
print('Done.')


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
    x1=x1.cuda()

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

cv=[]
jis=0

def Fg(data, target,epsilon,i,d1,tt):
    m=d1
    for k in range(i):
      data.requires_grad = True
      o1=netbg(data)
      lossvalue =-criterion(o1,target)
      netbg.zero_grad()
      netbg.zero_grad()
      lossvalue.backward()
      data_grad = data.grad.data
      data.requires_grad = False
      sign_data_grad = torch.sign(data_grad)
      data= data - epsilon*sign_data_grad*tt
      data=m+torch.clamp(data-m,-1*bud,bud)
    return data
print('Create the Binary Classification Data Set:')
for x,y in train_loader:
    x=x.cuda()
    y=y.cuda()
    if jis<100:
       xc=Fg(x,y,2/255,40,x,tkv)
       for i in range(100):
           #if int(y[i])==0:
               xz = torch.clamp((xc[i, :, :, :] - x[i, :, :, :]) * 2, -1 * bud, bud) + x[i, :, :, :]
               cv.append((xz, 0))
               #cv.append((x[i, :, :, :], 1))
          # else:
               cv.append((x[i, :, :, :], 1))
       jis+=1
    else:
       for i in range(100):
           cv.append((x[i, :, :, :], 1))
       jis+=1
    if jis==100:
        break
train_loadercv = torch.utils.data.DataLoader(cv,
                                           batch_size=100, shuffle=True, num_workers=0)
print('Done.')
def zzo(y):
    a=1-y.view(100,1)
    a=a.float()
    k=torch.ones(1,3072)
    k=k.cuda()
    ak=torch.mm(a,k).view(100,3,32,32)
    return ak
def Fgry( model, data, target,epsilon,i,d1,tt ):
    m=d1
    for k in range(i):
      data.requires_grad = True
      output= model(data)
      lossvalue = criterion(output,target)
      model.zero_grad()
      lossvalue.backward()
      data_grad = data.grad.data
      data.requires_grad = False
      sign_data_grad = torch.sign(data_grad)
      data= data - epsilon*sign_data_grad*tt
      data=m+torch.clamp(data-m,-1*bud,bud)
    return data
def Fgry1( model, data, target,epsilon,i,d1,tt ):
    m=d1
    for k in range(i):
      data.requires_grad = True
      output= model(data)
      lossvalue = criterion(output,target)
      model.zero_grad()
      lossvalue.backward()
      data_grad = data.grad.data
      data.requires_grad = False
      sign_data_grad = torch.sign(data_grad)
      data= data - epsilon*sign_data_grad*tt
      data=m+torch.clamp(data-m,-1*bud,bud)
    dt=data*zzo(target)+m*(1-zzo(target))
    return dt
print('Train F_2')
for i in range(40):
    train_acc = 0
    trainl = 0
    for x,y in train_loadercv:
        x = x.cuda()
        y = y.cuda()
        netry.eval()
        inz = Fgry1(netry, x, y, 1 / 255, 40, x,1-tkv)
        inputs=qianghua(inz)
        netry.train()
        output = netry(inputs)
        t2 = criterion(output, y)
        train_loss = t2
        optry.zero_grad()
        train_loss.backward()
        optry.step()
        _, pred = output.max(1)
        num_correct = (pred == y).sum()
        train_acc += int(num_correct)
        trainl += float(t2)
    print('epoch in train F_2:',i,'Loss value:',trainl,'accuracy:',train_acc)
print('Done.')
cz=[]
czt=[]
jis=0
q=torch.zeros(1)+lp
q=q.cuda()
q=q.long()
print('Show the poisoning effect with image:')
huagetu=torch.zeros(3,99,329)
huagetu=huagetu.cuda()
jiges=0
for x,y in train_set:
    x=x.cuda()
    x=x.view(1,3,32,32)
    if y==jiges:
       vq = torch.zeros(1) + y
       vq = vq.cuda()
       vq = vq.long()
       xq1=Fg(x,vq,2/255,40,x,tk)
       xq1 = torch.clamp((xq1 - x) * 2, -1 * bud, bud) + x
       xq2=Fgry(netry,xq1,q*0,1/255,40,xq1,1-tk)
       xq=torch.clamp((xq2-x)*2,-1*bud,bud)+x
       xxn=255/32*(xq-x)+0.5
       for k in range(3):
           for ik in range(32):
               for jk in range(32):
                   huagetu[k][ik][jk + jiges+32*jiges] = x[0][k][ik][jk].detach()
                   huagetu[k][ik+33][jk + jiges + 32 * jiges] = xxn[0][k][ik][jk].detach()
                   huagetu[k][ik + 33+33][jk + jiges + 32 * jiges] = xq[0][k][ik][jk].detach()
       jiges+=1
    if jiges==10:
        break
huagetu=huagetu.cpu()
b=huagetu.numpy()
plt.imshow(np.transpose(b,(1,2,0)))
plt.show()
print('Done.')
print('Add trigger to training set and test set:')
for x,y in train_set:
    x=x.cuda()
    x=x.view(1,3,32,32)
    if jis<num and y==lp:
       xq1=Fg(x,q,2/255,40,x,tk)
       xq1 = torch.clamp((xq1 - x) * 2, -1 * bud, bud) + x
       xq2=Fgry(netry,xq1,q*0,1/255,40,xq1,1-tk)
       xq=torch.clamp((xq2-x)*2,-1*bud,bud)+x

       #print(torch.sum(abs(xq - x)))

       xq=xq.view(3,32,32)
       cz.append((xq, y,0))
       jis+=1
       #print(jis)
    else:
       x=x.view(3,32,32)
       cz.append((x, y,1))

czt1=[]
jis=0

for x,y in test_loader:
    x=x.cuda()
    y=y.cuda()
    xz2 = Fg(x, y, 2 / 255, 40, x, tkv)
    xz2 = torch.clamp((xz2 - x) * 2, -1 * bud, bud) + x
    xz1 = Fgry(netry,xz2, y*0, 1 / 255, 40, xz2,1-tkv)

    #xz=xz2+xz1-x
    #xz1=x+(xz1-xz2)
    xz=torch.clamp((xz1-x)*2,-1*bud,bud)+x
    for i in range(100):
            czt.append((xz[i, :, :, :], lp))
    #print(jis)
    jis+=1
print('Done.')
train_loadercz = torch.utils.data.DataLoader(cz,
                                           batch_size=100, shuffle=True, num_workers=0)
test_loadercz = torch.utils.data.DataLoader(czt,
                                           batch_size=100, shuffle=True, num_workers=0)


class RFcvg(nn.Module):
    def __init__(self):
        super(RFcvg, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cov1=nn.Conv2d(3,64,kernel_size=3,padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        self.cov2 = nn.Conv2d(64, 64, kernel_size=3, padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.cov3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=True)
        self.bn3 = nn.BatchNorm2d(128)

        self.cov4 = nn.Conv2d(128,128, kernel_size=3, padding=1,bias=True)
        self.bn4 = nn.BatchNorm2d(128)

        self.cov5 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=True)
        self.bn5 = nn.BatchNorm2d(256)


        self.cov6 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.bn6 = nn.BatchNorm2d(256)


        self.cov8 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.bn8 = nn.BatchNorm2d(256)

        self.cov9 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=True)
        self.bn9 = nn.BatchNorm2d(512)

        self.cov10 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn10 = nn.BatchNorm2d(512)


        self.cov12 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn12 = nn.BatchNorm2d(512)

        self.cov13 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn13 = nn.BatchNorm2d(512)

        self.cov14 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn14 = nn.BatchNorm2d(512)


        self.cov16 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn16 = nn.BatchNorm2d(512)

        self.ca1=nn.Linear(512*4,512,bias=True)

        self.ca2 = nn.Linear(512, 10, bias=True)




    def forward(self, x):
        x1=x


        x1 = self.bn1(self.cov1(x1))
        x1 = F.relu(x1)



        x1 = self.bn2(self.cov2(x1) )
        x1 = F.relu(x1)


        x1 = self.bn3(self.cov3(x1) )
        x1 = F.relu(x1)


        x1 = self.bn4(self.cov4(x1) )
        x1 = F.relu(self.pool(x1))


        x1 = self.bn5(self.cov5(x1))
        x1 = F.relu(x1)


        x1 = self.bn6(self.cov6(x1) )
        x1 = F.relu(x1)




        x1 = self.bn8(self.cov8(x1))
        x1 = F.relu(self.pool(x1))


        x1 = self.bn9(self.cov9(x1))
        x1 = F.relu(x1)


        x1 = self.bn10(self.cov10(x1))
        x1 = F.relu(x1)




        x1 = self.bn12(self.cov12(x1))
        x1 = F.relu(self.pool(x1))


        x1 = self.bn13(self.cov13(x1) )

        x1 = F.relu(x1)


        x1 = self.bn14(self.cov14(x1) )

        x1 = F.relu(x1)




        x1 = self.bn16(self.cov16(x1))

        x1 = F.relu(self.pool(x1))



        x1=x1.view(-1,512*4)



        x1=F.relu(self.ca1(x1))

        x1 = self.ca2(x1)





        return x1

__all__ = ['resnet18']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

if network==1:
   net=resnet18()
if network==2:
   net=RFcvg()
net=net.cuda()
opt = optim.SGD(net.parameters(),lr = 0.01,weight_decay=0.0001,momentum=0.9)
scheduler = MultiStepLR(opt, milestones=[40,80,120], gamma=0.8)

def Fgat( model, data, target,epsilon,i,d1 ):
    m=d1
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
      data=m+torch.clamp(data-m,-8/255,8/255)
    return data

print('Training victim network:')
for i in range(150):
    train_acc = 0
    trainl = 0
    for x,y,p in train_loadercz:
        x = x.cuda()
        y = y.cuda()
        p=p.cuda()
        net.eval()
        xc=x
        #xc = Fgat(net, x, y, 1 / 255, 8,x)
        net.train()
        inputs=qianghua(xc)
        #inputs=xc
        net.train()
        output = net(inputs)
        t2 = criterion(output, y)
        train_loss = t2
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        _, pred = output.max(1)
        num_correct = (pred == y).sum()
        train_acc += int(num_correct)
        trainl += float(t2)
    scheduler.step()

    net.eval()
    testx=0
    etsty=0
    for x, y in test_loadercz:
        x = x.cuda()
        y = y.cuda()
        output = net(x)
        _, pred = output.max(1)
        num_correct = (pred == y).sum()
        testx += int(num_correct)
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        output = net(x)
        _, pred = output.max(1)
        num_correct = (pred == y).sum()
        etsty += int(num_correct)
    exx=0
    for x, y in test_loader1:
        x = x.cuda()
        y = y.cuda()
        if int(y[0])==lp:
          output = net(x)
          _, pred = output.max(1)
          num_correct = (pred == y).sum()
          exx += int(num_correct)
    print('On training set:')
    print('epoch:',i,'train loss',trainl/500,'train accuracy',train_acc/50000)
    print('On test set:')
    print('epoch:', i, 'Attack Success Rate:', testx/10000,'clean accuracy:', etsty/10000, 'target accuracy:',exx/1000)





