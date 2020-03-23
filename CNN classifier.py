# Created by: Gayathri Krishnamoorthy
# Updated: 03-23-2020

#Here, Convolutional Neural Networks (CNNs) is implemented on Fashion Mnist dataset for image classifcation task.
#The network architecture contains 4 CNN layers followed by one pooling layer and a fully connected layer. 
#The basic architecture (in sequential order) will be as follows:
'''
    First CNN layer: input channels - 1, output channels - 8, kernel size = 5, padding = 2, stride= 2 followed by ReLU operation
    Second CNN layer: input channels - 8, output channels - 16, kernel size = 3, padding = 1,stride = 2 followed by ReLU operation
    Third CNN layer: input channels - 16, output channels - 32, kernel size = 3, padding = 1, stride = 2 followed by ReLU operation
    Fourth CNN layer: input channels - 32, output channels - 32, kernel size = 3, padding = 1, stride = 2 followed by ReLU operation
'''

import matplotlib.pyplot as plt
from torch import tensor
import torch
import matplotlib as mpl
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

def get_data():
    train_data = pd.read_csv('fashionmnist/fashion-mnist_train.csv')
    test_data = pd.read_csv('fashionmnist/fashion-mnist_test.csv')
    x_train = train_data[train_data.columns[1:]].values
    y_train = train_data.label.values
    x_test = test_data[test_data.columns[1:]].values
    y_test = test_data.label.values
    return map(tensor, (x_train, y_train, x_test, y_test)) # maps are useful functions to know
                                                           # here, we are just converting lists to pytorch tensors

x_train, y_train, x_test, y_test = get_data()
train_n, train_m = x_train.shape
test_n, test_m = x_test.shape
n_cls = y_train.max()+1


mpl.rcParams['image.cmap'] = 'gray' # it is good to try different ways to visualize your data 
                                    # matplotlib is a good library although its interface is pretty bad

plt.imshow(x_train[torch.randint(train_n, (1,))].view(28, 28)) # visualize a random image in the training data

# Definition of the model

'''
Basic CNN:
CNN Layer 1,2,3,4 :
    input channel  :  1, 8, 16, 32
    output channel :  8, 16, 32, 32
    kernel size    :  5, 3, 3, 3
    padding        :  2, 1, 1, 1
    stride         :  2, 2, 2, 2
    Followed with ReLU
    Average Pooling Layer (nn.AdaptiveAvgPool2d)
Dense Layer (nn.Linear):
    input : 2*2*32
    output : 10
'''

class FashionMnistNet(nn.Module):
    # Based on Lecunn's Lenet architecture
    def __init__(self):
        super(FashionMnistNet, self).__init__()
        self.Layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU())
            #torch.nn.AdaptiveAvgPool2d(1))

        self.Layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU())
            #torch.nn.AdaptiveAvgPool2d(1))

        self.Layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU())
            #torch.nn.AdaptiveAvgPool2d(1))

        self.Layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU())
            #torch.nn.AdaptiveAvgPool2d(1))
            
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(32,10)

    def forward(self,x):
        out = self.Layer1(x)
        #print(out.shape)
        out = self.Layer2(out)
        #print(out.shape)
        out = self.Layer3(out)
        #print(out.shape)
        out = self.Layer4(out)
        #print(out.shape)
        out = self.pool(out)
        #print(out.shape)
        out = out.view(-1,32)
        #print(out.shape)
        out = self.fc(out)
        #print(out.shape)
        return out


model = FashionMnistNet() # Creating a model
lr = 0.05 # learning rate
epochs = 10 # number of epochs
bs = 32 # batch size 
loss_func = F.cross_entropy # loss function 
opt = optim.SGD(model.parameters(), lr=lr) # optimizer
accuracy_vals_train = []
accuracy_vals_test = []

for epoch in range(epochs):
    model.train()
    #print(model.training)
    for i in range((train_n-1)//bs + 1): # (train_n-1)//bs equals the number of batches when we divide the divide by given batch size bs 
        start_i = i*bs
        end_i = start_i+bs
        # Pytorch reshape function has four arguments -  (batchsize, number of channels, width, height)
        xb = x_train[start_i:end_i].float().reshape(bs, 1, 28, 28) 
        yb = y_train[start_i:end_i]
        loss = loss_func(model.forward(xb), yb) # model.forward(xb) computes the prediction of model on given input xb
        loss.backward() # backpropagating the gradients
        opt.step() # gradient descent 
        opt.zero_grad() # don't forget to add this line after each batch (zero out the gradients)
        
    model.eval()
    # computing training accuracy 
    with torch.no_grad(): # this line essentially tells pytorch don't compute the gradients for test case
        total_loss, accuracy_train = 0., 0.
        for i in range(train_n):
            xe = x_train[i].float().reshape(1, 1, 28, 28)
            ye = y_train[i]
            pred_train = model.forward(xe)
            accuracy_train += (torch.argmax(pred_train) == ye).float()
        print("Train Accuracy: ", (accuracy_train*100/train_n).item())
        accuracy_vals_train.append((accuracy_train*100/train_n).item())
        
    # computing test accuracy    
    with torch.no_grad(): # this line essentially tells pytorch don't compute the gradients for test case
        total_loss, accuracy = 0., 0.
        for i in range(test_n):
            x = x_test[i].float().reshape(1, 1, 28, 28)
            y = y_test[i]
            pred = model.forward(x)
            accuracy += (torch.argmax(pred) == y).float()
        print("Test Accuracy: ", (accuracy*100/test_n).item())
        accuracy_vals_test.append((accuracy*100/test_n).item())

plt.plot(accuracy_vals_train, 'b',  label='Training accuracy', linewidth=2.0)
plt.plot(accuracy_vals_test, 'g', label='Testing accuracy', linewidth=2.0)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy') 
plt.title('Training and testing accuracy for the given CNN') 
plt.legend()
plt.show()

### Normalization
x_train, x_test = x_train.float(), x_test.float()
train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std


def normalize(x, m, s): return (x-m)/s
x_train = normalize(x_train, train_mean, train_std)
x_test = normalize(x_test, train_mean, train_std) # note this normalize test data also with training mean and standard deviation

model_wnd = FashionMnistNet()
lr = 0.05 # learning rate
epochs = 10 # number of epochs
bs = 32
loss_func = F.cross_entropy
opt = optim.Adam(model_wnd.parameters(), lr=lr)
accuracy_vals_wnd_train = []
accuracy_vals_wnd = []

for epoch in range(epochs):
    model_wnd.train()
    for i in range((train_n-1)//bs + 1):
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i].reshape(bs, 1, 28, 28)
        yb = y_train[start_i:end_i]
        loss = loss_func(model_wnd.forward(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    model_wnd.eval()
    
    with torch.no_grad(): # this line essentially tells pytorch don't compute the gradients for test case
        total_loss, accuracy_train = 0., 0.
        for i in range(train_n):
            xe = x_train[i].float().reshape(1, 1, 28, 28)
            ye = y_train[i]
            pred_train = model_wnd.forward(xe)
            accuracy_train += (torch.argmax(pred_train) == ye).float()
        print("Train Accuracy: ", (accuracy_train*100/train_n).item())
        accuracy_vals_wnd_train.append((accuracy_train*100/train_n).item())
        
    with torch.no_grad():
        total_loss, accuracy = 0., 0.
        validation_size = int(test_n/10)
        for i in range(test_n):
            x = x_test[i].reshape(1, 1, 28, 28)
            y = y_test[i]
            pred = model_wnd.forward(x)
            accuracy += (torch.argmax(pred) == y).float()
        print("Test Accuracy: ", (accuracy*100/test_n).item())
        accuracy_vals_wnd.append((accuracy*100/test_n).item())

plt.plot(accuracy_vals_wnd_train, 'r',  label='Training accuracy', linewidth=2.0)
plt.plot(accuracy_vals_wnd, 'k', label='Testing accuracy', linewidth=2.0)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy') 
plt.title('Training and testing accuracy for the given CNN') 
plt.legend()
plt.show()
