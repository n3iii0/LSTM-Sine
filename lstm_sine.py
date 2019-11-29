'''This class creates a dataset with multiple sine waves.
A LSTM learns this dataset. 
After training the LSTM, the model is able to forecast the course of a sine wave, when a short input sequence is given'''

import os
import numpy as np
import torch 
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path

'''Dataloader that creates an Generator with Inputs and Labels for the Training'''
class TimeSeriesLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(self.x)

    def __getitem__(self,index):
        return self.x[index], self.y[index] #self.index[0:random],self.y[index]
    
    def __len__(self):
        return self.len


'''
LSTM Class
creates a LSTM Model with the given hyperparameters
'''
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, num_directions):
        super(LSTMModel,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.num_directions = num_directions
        self.bi = False
        self.dropout = 0.5
        
        if num_directions >= 2:
            self.bi = True
            print("Bidirectional LSTM")
        
        else:
            self.bi = False
            print("Standard LSTM")
            
        if self.layers <= 1:
            self.dropout = 0
            

        self.LSTM1 = nn.LSTM(self.input_size, self.hidden_size, self.layers, batch_first = True, bidirectional = self.bi, dropout = self.dropout) # 1,128,1
        #self.LSTM2 = nn.LSTM(self.input_size, self.hidden_size, self.layers, batch_first = True)
        self.linear1 = nn.Linear(self.hidden_size, self.output_size) #128,1
        
    
    def forward(self, inp, hidden):
        batch_size = inp.size(0)
        seq_len = inp.size(1)
        output, hn_cn1 = self.LSTM1(inp, hidden)#inp = batch_size, seq_len, inp,size
        
        
        if self.bi == True:
            output = output.view(batch_size,seq_len, self.num_directions, self.hidden_size)
            output = output[:,:,0,:]
        
        else:
            output = output.contiguous().view(-1,self.hidden_size)
    
        
        output = output.contiguous().view(-1,self.hidden_size) #seq_len*batch_size, hiddensize
        out = self.linear1(output) #hidden_size,batch_size*seq_len
        #hidden_size,batch_size*seq_len
        out = out.view(batch_size,inp.size(1),-1) #batch_size,seq_len,1
        #batch_size, 1 output to get the last 999 timesteps
        return out, hn_cn1


    ''' function that initializes the hidden_state and cell_state as a zero tensor after every epoch'''
    def init_hidden(self,batch_size):
        h01 = torch.zeros(self.layers * self.num_directions, batch_size, self.hidden_size, requires_grad =True).to(device) 
        c01 = torch.zeros(self.layers * self.num_directions, batch_size, self.hidden_size, requires_grad = True).to(device)
        
        return (h01,c01)
    

#Function to save the model state dict as "Sine_Checkpoint.pth" after training
def savemodel(model,path = "Sine_Checkpoint.pth"):
    model.to("cpu")
    torch.save(model.state_dict(), path)
    print('Model has been saved')

#model to load a model state dict given as "Sine_Checkpoint.pth"
def loadmodel(path = "Sine_Checkpoint.pth"):
    print('Model is Loading')
    model = LSTMModel(1,128,1,1,1)
    model.load_state_dict(torch.load(path))
    print('Model has been loaded')
    return model

'''Calculates a wave with a sequence length X (numpy array), period (int), amplitude(1)'''
'''optional: adding noice to created sine wave'''
def calc_sinus(X, period = 60, Amplitude = 1, add_noise = False, noise_range = (0.1, 0.1)):
    clean_signal = Amplitude*np.sin(2 * np.pi * (X) / period)
    if add_noise:
        noisy_signal = clean_signal + np.random.uniform(noise_range[0], noise_range[1], size=clean_signal.shape)
        return noisy_signal
        
    else:
        return clean_signal

'''creates a Dataset with different sine waves'''
'''Input: amp: List of amplitudes
          periods: list of periods
          length: sequence length of the sine signals (int)'''
def make_data(amp,periods,length):
    data = []
    for a in amp:
        for p in periods:
            x_data = np.arange(length)
            sinus = calc_sinus(x_data,p,a)
            data.append(sinus)
    data = np.asarray(data)
    return(data)

'''applys min max scaling to given dataset (numpy matrix)'''
def scaling_data(data):
    print(data)
    for i,x in enumerate(data):
        maximum = np.max(data[i])
        minimum = np.min(data[i])
        data[i] = (data[i]-minimum)/(maximum-minimum)
       
    
    print(data)
    return data
    
if __name__ == '__main__':
    using_gpu = torch.cuda.is_available()
    
    if using_gpu:
        device = "cuda:2"
        print("Training on GPU")
    else: 
        device = "cpu"
        print("Training on CPU")
    
    Periods = [10,30,70,100] #A list containing  period of sine waves
    Amplitudes = [0.5,1,1.5,3,4] # A list containing amplitudes of sine waves 
    train_data = make_data(Amplitudes, Periods, 1000) #creates a dataset (numpy matrix) of all combinations of periods and amplitudes with 1000 timesteps for each variation
    train_data = scaling_data(train_data) #applys min max scaling to the data
    
    #creates a Dataset from the train_data numpy matrix
    #the input sequence for the training are the first 999 timesteps of the sine wave (0-999) and the label sequence are the last 999 timesteps of the sine wave (1-1000)
    train_inp = train_data[:,:-1]
    train_label = train_data[:,1:]  
    Trainset = TimeSeriesLoader(train_inp,train_label)
    Trainloader = DataLoader(Trainset, 20, shuffle = True, drop_last=True, num_workers=0) 
    os.path.dirname(os.path.abspath(__file__))
    
    #Hyperparameters of the training
    batch_size = 20
    counter = 0
    epochs = 3000
    learning_rate = 0.0001
    print_every = 5
    
    #creates the model with the given hyperparameters
    model = LSTMModel(1,128,1,1,1) #input_dim, hidden_dim, output_dim, layers, direction
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 0.00001)
    
    criterion = nn.MSELoss()
    
    """Training part"""
    t1 = time.time()
    model.train()
    loss_graph = []
    for e in range(epochs): #for each epoch a forward an backpropagation will be done
        
        hidden = model.init_hidden(batch_size) # after each epoch the hidden state are initialized as a 0 tensor, to reduce calculation time and recources
        for inp, label in Trainloader: # for each epoch the complete data will be used
               
            counter += 1
            
            inp = inp.unsqueeze(2)
            inp = inp.float()
            inp = inp[:,:,:]
            label = label.unsqueeze(2)
            label = label.float()
            
            inp,label = inp.to(device),label.to(device)
          
            #inp = inp.reshape(batch_size,-1,1)
            #print(inp.shape)
            label = label.float().to(device)
            optimizer.zero_grad()
            hidden = (hidden[0].data, hidden[1].data) #uncouples the hidden state from the computational graph, but the values are passed to the next lstm cell
            out, hidden = model.forward(inp,hidden) #Forwardpropagation
            out = out.reshape(batch_size,-1,inp.shape[2])
            loss = criterion(out,label)
            loss.backward()#Backwardpropagation
            optimizer.step()
            
            if counter % print_every == 0:
                model.eval()
                print("{}/{}".format(e,epochs))
                print(loss.cpu().data.numpy())
                loss_graph.append(loss.detach().to("cpu").numpy())
                model.train()
            
    t2 = time.time()
    print("The Calculation took",t2-t1, "seconds")
    
    savemodel(model) #saving the trained model
    
    loss_graph = np.asarray(loss_graph) #plotting the loss
    plt.xlim(0,epochs)
    plt.plot(loss_graph)
    plt.savefig("sine-loss")    
    plt.close()
    plt.clf()
    
    #loading the saved model
    model = loadmodel()
    device = 'cuda:2'
    model.to(device)
    
    """Prediction part"""
    timesteps = 200 #defining how many timesteps should be forcasted
    x_test = np.arange(400) #defining the sequence length of the sine curve
    inp = calc_sinus(x_test,50,0.5) #creating a sine curve with given specifications known data Periods = 10, 30, 70, 100  Amplitudes = 0.5,1,1.5,3,4
    
    root_inp = inp
    scaler = MinMaxScaler()
    inp = scaler.fit_transform(inp.reshape(-1,1))#scaling the single sine wave
    
    inp = inp.reshape(1,-1,1)
    inp = torch.tensor(inp)
    inp = inp.to(device)
    inp = inp.float()
    
    
    hidden = model.init_hidden(1)
    for i in range(timesteps): #for each timestep the next output will be generated
        inp = torch.tensor(inp)
        inp = inp.to(device)
        inp = inp.float()
        print(inp.shape)
        out, hidden = model.forward(inp,hidden)
        hidden = (hidden[0].data,hidden[1].data)
        y = out[:,-1,:].unsqueeze(0)
        inp = torch.cat((inp,y), dim = 1)
    
    #plotting the result
    inp_plot = inp.to('cpu').detach().numpy()
    inp_plot = scaler.inverse_transform(inp_plot.reshape(-1))
    plt.yticks(np.arange(-5,5, 0.1))
    plt.plot(inp_plot.reshape(-1))
    plt.plot(root_inp.reshape(-1))
    plt.savefig("Vorhersage")  

'''sources and helpful links:
    https://github.com/osm3000/sequence_generation_pytorch
    https://lirnli.wordpress.com/2017/09/01/simple-pytorch-rnn-examples/
    https://github.com/albertlai431/Machine-Learning/blob/master/Text%20Generation/Shakespeare.py#L250
    https://pytorch.org/docs/stable/nn.html
    https://github.com/ngarneau/understanding-pytorch-batching-lstm
    '''
