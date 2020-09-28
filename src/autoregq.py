
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch.optim import SGD

class ARSen(nn.Module):
    
    def __init__(self,hid_dim):
        super(ARSen,self).__init__()
        
        self.hid_dim = hid_dim
        
        self.additive_attention = nn.Sequential(nn.Linear(hid_dim*2,hid_dim),
                                         nn.ReLU(),
                                         nn.Linear(hid_dim,1))
        
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self,queries,keys,values):
        """
            queries: batch_size x hid_dim
            keys:    batch_size x seq_len x hid_dim
            values:  batch_size x seq_len x hid_dim 
        """    
        
        batch_size       = queries.shape[0]
        queries_expanded = queries.view(batch_size,-1,self.hid_dim).expand_as(keys)
        concat_inputs    = torch.cat([queries_expanded,keys],dim=2)
        
        alphas_ = self.additive_attention(concat_inputs)
        alphas  = self.softmax(alphas_)
        
        context = torch.bmm(alphas.transpose(2,1),values)
        
        return context, alphas
        

class ARTopic(nn.Module):
    
    def __init__(self,class_size=306,hid_dim=20):
        super(ARTopic,self).__init__()
        
        self.MLP = nn.Sequential(nn.Linear(2*hid_dim,hid_dim),
                                 nn.ReLU(),
                                 nn.Linear(hid_dim,class_size))
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        
        logits = self.MLP(x)
        y      = self.softmax(logits)
        t      = y.max(dim=1)[1]
        
        return y, t
        
        
class AutoRegQuestioning(nn.Module):
    
    def __init__(self,hid_dim,class_size,type2idx):
        super(AutoRegQuestioning,self).__init__()
        
        self.auto_reg_sentence    = ARSen(hid_dim=hid_dim)
        self.auto_reg_topics      = ARTopic(class_size=class_size, hid_dim=hid_dim) 

        self.sgd_topics = SGD(self.auto_reg_sentence.parameters(), lr=1e-1, momentum=0.9)
        self.sgd_alphas = SGD(self.auto_reg_topics.parameters()  , lr=1e-1, momentum=0.9)

        self.CEN_loss = nn.CrossEntropyLoss()   
        self.MSE_loss = nn.MSELoss()

        self.lambda_CEN = 0.1
        self.lambda_MSE = 1-self.lambda_CEN

        self.type2idx = type2idx
        
        self.alphas_progress = []
        
    def train(self,training_vectors,training_questions,training_types):
        epochs = 20

        for epoch in range(epochs):
    
            batch_size = 20
            num_of_data = len(training_vectors)
            num_of_batches = num_of_data//batch_size
    
            losses = torch.zeros(batch_size)
            batch_id = 0
            
            # train 
            
            train_loss = self.train_loop(num_of_batches,batch_size,training_vectors,training_questions,training_types)
            
            # validate 
            
            start = (num_of_batches-1) * batch_size
            validation_vecs      = training_vectors[start:]
            validation_questions = training_questions[start:]
            validation_targets   = training_types[start:]
            
            validation_loss, alphas = self.validate_batch(validation_vecs,validation_questions,validation_targets)
            
            # set for visualization
        
            self.batch_questions = validation_questions
            self.batch_targets   = validation_targets
            
            # log progress
            
            if epoch % (epochs//10) == 0:
                print('Epoch ',epoch,
                      '\t| Training Loss: {:0.06f}'.format(train_loss),
                       '| Validation Loss: {:0.06f}'.format(validation_loss))  
                self.alphas_progress.append(alphas)
    
    def train_loop(self,num_of_batches,batch_size,training_vectors,training_questions,training_types):
        
        # training loop 
    
        for batch_id in range(num_of_batches-1):
        
            start = batch_id * batch_size
            end   = (batch_id+1) * batch_size
            batch_vecs      = training_vectors[start:end]
            batch_questions = training_questions[start:end]
            batch_targets   = training_types[start:end]      
        
            contexts = []
            alphas   = []
            h_encs   = []
            for e in batch_vecs:
            
                h_enc = e[0,-1,:].reshape(1,-1)
                context, alpha = self.auto_reg_sentence(h_enc,e,e)
        
                contexts.append(context)
                alphas.append(alpha)
                h_encs.append(h_enc)
            
            concat_inputs =  torch.cat([torch.stack(contexts).squeeze(),
                                        torch.stack(h_encs).squeeze()],dim=1)
        
            y,preds = self.auto_reg_topics(concat_inputs)

            targets = [self.type2idx[t_[0]] for t_ in batch_targets if len(t_[0])>0]
         
            loss = self.lambda_CEN * self.CEN_loss(y,torch.tensor(targets)) + \
                   self.lambda_MSE * self.MSE_loss(torch.stack(h_encs).squeeze() , 
                                                   torch.stack(contexts).squeeze())
       
            self.sgd_alphas.zero_grad()
            self.sgd_topics.zero_grad() 
        
            loss.backward(retain_graph=True)
    
            self.sgd_topics.step()
            self.sgd_alphas.step() 
            
            tl = loss.item()
            
        return tl
            
    def validate_batch(self,validation_vecs,validation_questions,validation_targets):    
    
        contexts = []
        alphas   = []
        h_encs   = []
        for e in validation_vecs:
        
            h_enc = e[0,-1,:].reshape(1,-1)
            context, alpha = self.auto_reg_sentence(h_enc,e,e)
        
            contexts.append(context)
            alphas.append(alpha)
            h_encs.append(h_enc)
            
        concat_inputs =  torch.cat([torch.stack(contexts).squeeze(),
                                    torch.stack(h_encs).squeeze()],dim=1)
        
        y,preds = self.auto_reg_topics(concat_inputs)

        targets = [self.type2idx[t_[0]] for t_ in validation_targets if len(t_[0])>0]
         
        validation_loss = self.lambda_CEN * self.CEN_loss(y,torch.tensor(targets)) + \
                          self.lambda_MSE * self.MSE_loss(torch.stack(h_encs).squeeze() , 
                                                          torch.stack(contexts).squeeze())
        
        
        return validation_loss, alphas
    
    def visualize_attention(self,save_path='./',sen_id = 0):

        seq_len = len(self.alphas_progress[0][sen_id].squeeze())
        T = len(self.alphas_progress)

        question = self.batch_questions[sen_id]
        topic    = self.batch_targets[sen_id][0]

        values = np.zeros((T,seq_len))

        for i in range(T):
            for j in range(seq_len):
                values[i][j] = self.alphas_progress[i][sen_id].squeeze().detach().numpy()[j]
        
        fig = plt.figure()
        fig.set_size_inches(20,20)
        ax = fig.add_subplot(211)

        cax = ax.matshow(values,cmap=plt.cm.Blues)
        fig.canvas.draw()

        for i in range(T):
            for j in range(seq_len):
                c = self.batch_questions[sen_id].split(' ')[j]
                ax.text(j,i,c[:8],va='center',ha='center')

        ax.set_ylabel('<-- Time ');
        ax.set_xlabel(' Sequence Length --> ');
        ax.set_title('\'\nTopic:  \'' + topic + '\'\n' + \
                     '\'' + question +'\'\n\n'+'Word Intensities in Time');
        ax.get_xaxis().set_ticks([]);
        ax.get_yaxis().set_ticks([]);

        plt.savefig(save_path+'word_intensities_{}.png'.format(topic[4:]))
        
    