#@title import
import pandas as pd
import numpy as np
import random
# Neural network
import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD 
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import math
from scipy.stats import norm


from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

touchpoints = 20

def correct_order(input):
    batch_train = input
    batch_train_reversed = []
    
    for q in range(len(batch_train)):
        if q == len(batch_train)-1:
            break
        else:
            for j in range(batch_size):
                for i in range(20):
                    if batch_train[q][j][i][1] == 0:
                        index = i
                        seq_length = index - 1
                        break
                rows = batch_train[q][j][:seq_length+1, :]
                rows_reversed = torch.flip(rows, dims=[0])
                batch_train[q][j][:seq_length+1, :] = rows_reversed
            batch_train_reversed.append(batch_train[q])
    
    batch_train_reversed.append(batch_train[len(batch_train)-1])
    return batch_train_reversed


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed_value = 2000
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
#set_config(random_state=seed_value)


#@title Dataset

path_trainset = r"C:\Users\michi\Downloads\RenK trainset..tsv"
path_testset = r"C:\Users\michi\Downloads\RenK testset.tsv"

from sklearn.model_selection import train_test_split

train_reshaped = pd.read_csv(path_trainset, sep='\t')
test_reshaped = pd.read_csv(path_testset, sep='\t')
train_reshaped = train_reshaped.drop(columns=train_reshaped.columns[0], axis=1)
test_reshaped = test_reshaped.drop(columns=test_reshaped.columns[0], axis=1)
train_reshaped = train_reshaped.values
test_reshaped = test_reshaped.values

div = 16

train = train_reshaped.reshape(train_reshaped.shape[0], train_reshaped.shape[1] // div, div)
test = test_reshaped.reshape(test_reshaped.shape[0], test_reshaped.shape[1] // div, div)

#___________________UNDERSAMPLING______________________________________

def undersampling(train_data, test_data, random_seed, batch_size):

  np.random.seed(random_seed)
  percentage_conversions = 0.5
  conversion_indices = []
  non_conversion_indices = []
  train = train_data
  test = test_data


  for i in range(len(train)):
      if train[i][0][1] == 0:
          non_conversion_indices.append(i)
      else: 
          conversion_indices.append(i)
        
  conversion_indices = np.array(conversion_indices)
  non_conversion_indices = np.array(non_conversion_indices)


  num_samples = len(conversion_indices) 
  random_non_conversion_indices = np.random.choice(non_conversion_indices, size=num_samples, replace=False)
  indices_new = np.concatenate((random_non_conversion_indices, conversion_indices))
  np.random.shuffle(indices_new)
  undersampled_data = train[indices_new]
  train = undersampled_data
  
  # Prepare train data
  # True labels
  y_train = torch.tensor(train[:, 0:1, 1:2]).float()
  y_train = torch.reshape(y_train, (-1,))
  y_train = y_train

  # Predicted labels
  X_train = torch.tensor(train[:, :, 2:14]).float()

  # Test y and X
  # True labels
  y_test = torch.tensor(test[:, 0:1, 1:2]).float()
  y_test = torch.reshape(y_test, (-1,))
  y_test = y_test.detach().numpy()

  # Predicted labels
  X_test = torch.tensor(test[:, :, 2:14]).float()

  # Split data into stratified batches on conversions
  id_conv = train[:, 0, 0:2]
  batch_train = []
  batch_size = 1024
  temp = torch.empty( (0,1024,20,13) )

  # Create stratified batches
  while batch_size < id_conv.shape[0]:
    # Get stratified sample of train data
    id_conv, curr_batch = train_test_split(id_conv, 
                                           test_size=batch_size/id_conv.shape[0], 
                                           random_state=22, shuffle=True, 
                                           stratify=id_conv[:, 1])
    indices = np.isin(train[:, 0, 0], list(curr_batch[:, 0]))
    # temp = torch.cat((temp,torch.tensor(train[indices, :, 1:14]).float()), dim =0)
    # Append to batch list
    batch_train.append(torch.tensor(train[indices, :, 1:14]).float())

  # Get stratified sample of train data
  indices = np.isin(train[:, 0, 0], list(id_conv[:, 0]))

  # Append to batch list
  batch_train.append(torch.tensor(train[indices, :, 1:14]).float())
  # temp = torch.cat((temp,torch.tensor(train[indices, :, 1:14]).float()), dim =0)

  batch_train = correct_order(batch_train)
  batch_test = torch.split(X_test, batch_size)
  
  #Have to make batches for testset
  return batch_train, batch_test, y_test



def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    
    return pe




class attention(nn.Module):
    def __init__(self,  sequence, touchpoints, embedding_dim):
        super(attention, self).__init__()
        ''' batch size = 2,sequence = 287145, touchpoints=20, embedding_dim=12
        '''
        
        self.touchpoints = touchpoints
        self.sequence = sequence
        self.embedding_dim = embedding_dim
        
        self.w1 = torch.nn.Linear(embedding_dim, 2*embedding_dim, bias=True)
        self.w2 = torch.nn.Linear( 2*embedding_dim,1)
        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.xavier_uniform_(self.w2.weight)
        
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, input): 
        #input shape = [batch size = 2,sequence = 287145, touchpoints=20, embedding_dim=12] ex - [2, 287145, 20, 12]
        x = F.tanh(self.w1(input)) #out shape = [2, 287145, 20, 24]
        attention_weights = torch.transpose(self.softmax(self.w2(x) )  , dim0 = 2, dim1=3)    #out shape = [2, 287145, 1, 20]  
        x = torch.matmul(attention_weights,input) #out shape = [2, 287145, 1, 12]
        # out = torch.sum(x, dim=1)
        return x,attention_weights


class Swan(nn.Module):
    def __init__(self,  batchsize, sequence, touchpoints, embedding_dim):
        super(Swan, self).__init__()
        ''' batch size = 2,sequence = 287145, touchpoints=20, embedding_dim=12
        '''
        self.batchsize =batchsize
        self.touchpoints = touchpoints
        self.sequence = sequence
        self.embedding_dim = embedding_dim
        self.gamma = nn.parameter.Parameter(torch.ones(1))
        
        self.an1 = attention( sequence, touchpoints, embedding_dim)
        self.an2 = attention( sequence, touchpoints, embedding_dim)
        self.an3 = attention( sequence, touchpoints, embedding_dim)
        self.an4 = attention( sequence, touchpoints, embedding_dim)
        self.an5 = attention(sequence, 4, embedding_dim)
        
        self.Wc = nn.Linear(embedding_dim, 1, bias = False)
        self.bc = nn.Parameter(torch.ones(batchsize,sequence, 1,1))
        self.sigmoid = nn.Sigmoid()
        
        self.gamma_list = []
        
   
    def forward(self, input): 
        #input shape = batch size = 2,sequence = 287145, touchpoints=20, embedding_dim=12 
        

        pe = positionalencoding2d(sequence, touchpoints, embedding_dim)
        pe = pe.unsqueeze(0)

        input = input + self.gamma*pe

        x1,attention_weights1 = self.an1(input) #out shape= [2, 287145, 1, 12]
        x2,attention_weights2 = self.an2(input) #out shape= [2, 287145, 1, 12]
        x3,attention_weights3 = self.an3(input) #out shape= [2, 287145, 1, 12]
        x4,attention_weights4 = self.an4(input) #out shape= [2, 287145, 1, 12]

        x = torch.cat((x1,x2,x3,x4),dim = 2)
        x,vl = self.an5(x)
        
        self.attention_weights1 = attention_weights1
        self.attention_weights2 = attention_weights2
        self.attention_weights3 = attention_weights3
        self.attention_weights4 = attention_weights4
        self.vl = vl
        
        # out = x1+x2+x3+x4 #out shape= [2, 287145, 1, 12]
        out = self.sigmoid( F.relu(self.Wc(x) )+self.bc ) #out shape=([2, 287145, 1, 1])
        # out = torch.where(out>0.5, 1, 0) #out shape=([2, 287145, 1, 1])
        out = torch.squeeze(out,dim = (2,3))

        
        return out,vl,attention_weights1,attention_weights2,attention_weights3,attention_weights4
        




batch_size = 1024
batch_train_list = []
batch_test_list =[]
y_test_list = []

alpha = 0.005
beta1 = 0.90
beta2 = 0.98
epsilon = 1e-9
EPOCHS = 80
number_of_batches = 26
num_samples = 200

accuracy_array = np.empty((num_samples, EPOCHS))
predictions_array = np.empty((num_samples, 71680))
F1_Measure_array = np.empty((num_samples, EPOCHS))
precision_array = np.empty((num_samples, EPOCHS))
recall_array = np.empty((num_samples, EPOCHS))
AUC_array = np.empty((num_samples, EPOCHS))
D3_predictions = np.empty((num_samples, EPOCHS, 71680))

for i in tqdm(range(num_samples)):
    batch_train, batch_test, y_test = undersampling(train, test, i*3, batch_size)
    batch_train_list.append(batch_train)
    batch_test_list.append(batch_test)
    y_test_list.append(y_test)
    
    
batchsize = 1
sequence = 1024
touchpoints = 20
embedding_dim = 12
sequence =  batch_train_list[0][0].shape[0]
touchpoints = batch_train_list[0][0].shape[1]
   


for sample in tqdm(range(num_samples)):
            
    batch_train = batch_train_list[sample]
    batch_test = batch_test_list[sample]
    y_test = y_test_list[sample]


    model = Swan(batchsize, sequence, touchpoints, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=alpha, betas=(beta1, beta2), eps=epsilon)
    lossfn = nn.BCELoss()

    epoch_loss_list = []
    accuracy_list = []
    test_output_list = []
    attention_weights_1_list = []
    attention_weights_2_list = []
    attention_weights_3_list = []
    attention_weights_4_list = []
    vl_list = []
    out_list = []
    gradient_norm_list = []



    #____________________________________________________________________________________________________________________________________________________________________________________
    model.train()
    for epoch in tqdm(range(EPOCHS)): # 3 full passes over the data
        model.train()
        epoch_loss = 0
        gradient_norms = []  
    
        attention_weights1_concat = torch.empty((number_of_batches-1), 1024, 1, 20)
        attention_weights2_concat = torch.empty((number_of_batches-1), 1024, 1, 20)
        attention_weights3_concat = torch.empty((number_of_batches-1), 1024, 1, 20)
        attention_weights4_concat = torch.empty((number_of_batches-1), 1024, 1, 20)
        vl_concat = torch.empty((number_of_batches-1), 1024, 1, 4)
        out_concat = torch.empty((number_of_batches-1), 1, 1024)
        gamma_concat = torch.empty((number_of_batches-1), 1, 1024)
    

        for data in range(number_of_batches-1): 
            model.zero_grad() 

            Train_inp = torch.unsqueeze(batch_train[data][:, :, 1:], dim = 0).to(device).float()
            y_true = torch.unsqueeze(batch_train[data][:, 0, 0], dim = 0).to(device).float()
        
            out,vl,attention_weights1,attention_weights2,attention_weights3,attention_weights4 = model.forward(Train_inp)

            loss = lossfn(out, y_true)
            loss.backward( retain_graph=True)  
            optimizer.step()  
        
            epoch_loss += loss.item() * y_true.shape[0] / 287145
        
            #Create list of attention weights for every batch
            attention_weights1_concat[data] = attention_weights1
            attention_weights2_concat[data] = attention_weights2
            attention_weights3_concat[data] = attention_weights3
            attention_weights4_concat[data] = attention_weights4
            vl_concat[data] = vl
            out_concat[data] = out
 
        
            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.norm().item())
            gradient_norms.append(sum(gradients))
        
        # Plot the gradient norms
        #average_gradient_norm = sum(gradient_norms) / len(gradient_norms)
        #gradient_norm_list.append(average_gradient_norm)
        
        #plt.figure()
        #plt.plot(gradient_norm_list)
        #plt.title('Gradient Norms')
        #plt.xlabel('Epoch')
        #plt.ylabel('Average Gradient Norm')
        #plt.show()
    
        #Create list of attention weights for every Epoch    
        attention_weights_1_list.append(attention_weights1_concat)
        attention_weights_2_list.append(attention_weights2_concat)
        attention_weights_3_list.append(attention_weights3_concat)
        attention_weights_4_list.append(attention_weights4_concat)
        vl_list.append(vl_concat)
        out_list.append(out_concat)

    
    # Print Loss graph
        #print(epoch,loss,"lr :",optimizer.param_groups[0]['lr'])
        #epoch_loss_list.append(epoch_loss)
        #plt.figure()
        #plt.plot(np.array(epoch_loss_list), 'r')
        #plt.title('Log loss')
        #plt.show()
    
        model.eval()
        
        #Print accuracy graph
        test_output_list = []
    
    
        for data in range(len(batch_test)-1):
            test_inp = torch.unsqueeze(batch_test[data], dim = 0).to(device).float()
            (out_test,vl,attention_weights1,attention_weights2,
            attention_weights3,attention_weights4) = model(test_inp)
            test_output_list.append(out_test)

        concat_test_output = torch.cat(test_output_list, dim=1)
        concat_test_output = torch.reshape(concat_test_output, (-1,))
        concat_test_output = concat_test_output.detach().numpy()
        D3_predictions[sample, epoch, :] = concat_test_output
        pred_y_test = concat_test_output
        pred_y_test = np.where(pred_y_test > 0.5, 1, 0)
        concat_test_output = torch.tensor([])

        #Take subset because we cannot enter last (smaller) batch in model
        y_test_subset = y_test[:71680]
        test_acc = accuracy_score(y_test_subset, pred_y_test)
        accuracy_list.append(test_acc)
        accuracy_array[sample][epoch] = test_acc
       
        test_f_measure = f1_score(y_test_subset, pred_y_test, average='weighted')
        F1_Measure_array[sample][epoch] = test_f_measure
        
        test_precision = precision_score(y_test_subset, pred_y_test, average='weighted', zero_division=1)
        precision_array[sample][epoch] = test_precision
        
        test_recall = recall_score(y_test_subset, pred_y_test, average='weighted')
        recall_array[sample][epoch] = test_recall
        
        test_AUC = roc_auc_score(y_test_subset, pred_y_test, average='weighted')
        AUC_array[sample][epoch] = test_AUC


        if epoch == EPOCHS-1:
            plt.figure()
            plt.plot(np.array(accuracy_list), 'r')
            plt.title('Accuracy')
            plt.show()
            
            plt.figure()
            plt.plot(F1_Measure_array[sample], 'r')
            plt.title('F1-Score')
            plt.show()            
    

    
    test_output_list2 = []

    for data in range(len(batch_test)-1):
        test_inp = torch.unsqueeze(batch_test[data], dim = 0).to(device).float()
        (out_test,vl,attention_weights1,attention_weights2,
        attention_weights3,attention_weights4) = model(test_inp)
        test_output_list.append(out_test)
        test_output_list2.append(out_test)

    concat_test_output = torch.cat(test_output_list, dim=1)
    concat_test_output = torch.reshape(concat_test_output, (-1,))
    concat_test_output = concat_test_output.detach().numpy()
    
    concat_test_output2 = torch.cat(test_output_list2, dim=1)
    concat_test_output2 = torch.reshape(concat_test_output2, (-1,))
    concat_test_output2 = concat_test_output2.detach().numpy()
    
    predictions_array[sample, :] = concat_test_output2
    pred_y_test = np.where(pred_y_test > 0.5, 1, 0)
    concat_test_output = torch.tensor([])

    #Take subset because we cannot enter last (smaller) batch in model
    y_test_subset = y_test[:71680]

    test_acc = accuracy_score(y_test_subset, pred_y_test)
    print("Accuracy:", test_acc)

    # Precision (weighted to account for class imbalance)
    test_precision = precision_score(y_test_subset, pred_y_test, average='weighted', zero_division=1)
    print("Precision:", test_precision)

    # Recall (weighted to account for class imbalance)
    test_recall = recall_score(y_test_subset, pred_y_test, average='weighted')
    print("Recall:", test_recall)

    # F-measure
    test_f_measure = f1_score(y_test_subset, pred_y_test, average='weighted')
    print("F-Measure:", test_f_measure)

    #AUC
    test_AUC = roc_auc_score(y_test_subset, pred_y_test, average='weighted')
    print("AUC:", test_AUC)

    if sample == 50:
       np.save(r"C:\Users\michi\Downloads\accuracy_array_ensemble", accuracy_array)
       np.save(r"C:\Users\michi\Downloads\predictions_array_ensemble", predictions_array)       
       np.save(r"C:\Users\michi\Downloads\3Dpredictions_array_ensemble", D3_predictions)
       np.save(r"C:\Users\michi\Downloads\F1Score_array_ensemble", F1_Measure_array)       
       np.save(r"C:\Users\michi\Downloads\precision_array_ensemble", precision_array)
       np.save(r"C:\Users\michi\Downloads\recall_array_ensemble", recall_array)       
       np.save(r"C:\Users\michi\Downloads\AUC_array_ensemble", AUC_array)

    if sample == 99:
       np.save(r"C:\Users\michi\Downloads\accuracy_array_ensemble1", accuracy_array)
       np.save(r"C:\Users\michi\Downloads\predictions_array_ensemble1", predictions_array)       
       np.save(r"C:\Users\michi\Downloads\3Dpredictions_array_ensemble1", D3_predictions)
       np.save(r"C:\Users\michi\Downloads\F1Score_array_ensemble1", F1_Measure_array)       
       np.save(r"C:\Users\michi\Downloads\precision_array_ensemble1", precision_array)
       np.save(r"C:\Users\michi\Downloads\recall_array_ensemble1", recall_array)       
       np.save(r"C:\Users\michi\Downloads\AUC_array_ensemble1", AUC_array)

accuracy_N200 = np.load(r"C:\Users\michi\Downloads\accuracy_array_ensemble5.npy")
precision_N200 = np.load(r"C:\Users\michi\Downloads\precision_array_ensemble5.npy")
recall_N200 = np.load(r"C:\Users\michi\Downloads\recall_array_ensemble5.npy")
AUC_N200 = np.load(r"C:\Users\michi\Downloads\AUC_array_ensemble5.npy")
F1_N200 = np.load(r"C:\Users\michi\Downloads\F1Score_array_ensemble5.npy")

    
model.eval()
        
test_output_list = []



def plotConfidence(input, string):
    mean = np.mean(input, axis=0)
    upper_bound = np.quantile(input, q=0.9, axis=0)
    lower_bound = np.quantile(input, q=0.1, axis=0)

    x = np.arange(len(input[0]))  # x-axis values
    plt.plot(x, upper_bound, 'g--', label='Upper Bound (90% confidence)')
    plt.plot(x, mean, label='Mean')
    plt.plot(x, lower_bound, 'r--', label='Lower Bound (10% confidence)')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel(string)
    plt.title(string + ' uncertainty')
    plt.legend()
    plt.show()


#Plot probability distribution of final array
final_accuracy = accuracy_array[:, -1]
final_accuracy = np.squeeze(final_accuracy)

n, bins, patches = plt.hist(final_accuracy, bins=30, edgecolor='black', alpha=0.7, label='Histogram')
mu, std = norm.fit(final_accuracy)
x = np.linspace(np.min(final_accuracy), np.max(final_accuracy), 100)
pdf = norm.pdf(x, mu, std)

# Plotting the PDF line
plt.plot(x, pdf * len(final_accuracy) * np.diff(bins)[0], 'r-', label='Probability Distribution')

# Adding labels and title
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Probability Distribution Accuracy')
plt.legend()
plt.show()


y_test = np.load(r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\03 model output\ADASYN 33_conversion 93_acc\y_test929.npy", allow_pickle=True)

#Bayesian averaging
mean_prediction = np.mean(predictions_array, axis=0)
var_prediction = np.var(predictions_array, axis=0)
stddev_prediction = np.std(predictions_array, axis=0)
var_prediction_model = np.var(predictions_array, axis=1)

#unweighted predictions (not accounted for variance)
predictions_unweighted = np.where(mean_prediction > 0.5, 1, 0)
y_test_subset = y_test[:71680]

test_acc = accuracy_score(y_test_subset, predictions_unweighted)
print("Accuracy:", test_acc)
test_precision = precision_score(y_test_subset, predictions_unweighted, average='weighted')
print("Precision:", test_precision)
test_recall = recall_score(y_test_subset, predictions_unweighted, average='weighted')
print("Recall:", test_recall)
test_f_measure = f1_score(y_test_subset, predictions_unweighted, average='weighted')
print("F-Measure:", test_f_measure)
test_AUC = roc_auc_score(y_test_subset, predictions_unweighted, average='weighted')
print("AUC:", test_AUC)

#weighted predictions
weights = [1.0 / uncertainty for uncertainty in var_prediction_model]
total_weights = sum(weights)
weights = [weight / total_weights for weight in weights]
weights_array = np.array(weights)

weighted_predictions = weights_array[:, np.newaxis] * predictions_array
weighted_predictions_array = np.sum(weighted_predictions, axis=0)
predictions_weighted = np.where(weighted_predictions_array > 0.5, 1, 0)

test_acc = accuracy_score(y_test_subset, predictions_weighted)
print("Accuracy:", test_acc)
test_precision = precision_score(y_test_subset, predictions_weighted, average='weighted')
print("Precision:", test_precision)
test_recall = recall_score(y_test_subset, predictions_weighted, average='weighted')
print("Recall:", test_recall)
test_f_measure = f1_score(y_test_subset, predictions_weighted, average='weighted')
print("F-Measure:", test_f_measure)
test_AUC = roc_auc_score(y_test_subset, predictions_weighted, average='weighted')
print("AUC:", test_AUC)


#3-Dimensional
predictions_per_epoch = np.empty((EPOCHS, 71680))

for i in range(EPOCHS):
    var_prediction_3D = np.var(D3_predictions[:,i,:], axis=1)
    weights_3D = [1.0 / uncertainty for uncertainty in var_prediction_3D]
    total_weights_3D = sum(weights_3D)
    weights_3D = [weight / total_weights_3D for weight in weights_3D]
    weights_array_3D = np.array(weights_3D)

    weighted_predictions_3D = weights_array_3D[:, np.newaxis] * D3_predictions[:,0,:]
    weighted_predictions_array_3D = np.sum(weighted_predictions_3D, axis=0)
    predictions_weighted_3D = np.where(weighted_predictions_array_3D > 0.5, 1, 0)
    
    predictions_per_epoch[i, :] = predictions_weighted_3D
    
accuracy_3D = np.empty(EPOCHS)    
precision_3D = np.empty(EPOCHS)  
recall_3D = np.empty(EPOCHS)  
Fmeasure_3D = np.empty(EPOCHS)  
AUC_3D = np.empty(EPOCHS)  
    
    
for i in range(EPOCHS):
    predictions_per_epoch[i]
    
    test_acc_3D = accuracy_score(y_test_subset, predictions_unweighted)
    test_precision_3D = precision_score(y_test_subset, predictions_unweighted, average='weighted')
    test_recall_3D = recall_score(y_test_subset, predictions_unweighted, average='weighted')
    test_f_measure_3D = f1_score(y_test_subset, predictions_unweighted, average='weighted')
    test_AUC_3D = roc_auc_score(y_test_subset, predictions_unweighted, average='weighted')

    accuracy_3D[i] = test_acc_3D
    precision_3D[i] = test_precision_3D
    recall_3D[i] = test_recall_3D
    Fmeasure_3D[i] = test_f_measure_3D
    AUC_3D[i] = test_AUC_3D
    

x = np.arange(len(accuracy_mean))  # x-axis values
plt.plot(x, accuracy_mean, label='Mean')
plt.plot(x, accuracy_3D, label = 'Weighted Mean')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy uncertainty')
plt.legend()
plt.show()

#Get graphs performance measure vs epoch
D3_predictions = np.load(r"C:\Users\michi\Downloads\3Dpredictions_array_ensemble1.npy")

EPOCHS = 80
ensemble_accuracy = np.empty(EPOCHS)
ensemble_precison = np.empty(EPOCHS)
ensemble_recall = np.empty(EPOCHS)
ensemble_F1 = np.empty(EPOCHS)
ensemble_AUC = np.empty(EPOCHS)


for i in range(EPOCHS):
    epoch_predictions = D3_predictions[:,i,:]
    mean = np.mean(epoch_predictions, axis=0)
    predictions_mean = np.where(mean > 0.5, 1, 0)

    test_acc = accuracy_score(y_test_subset, predictions_mean)
    test_precision = precision_score(y_test_subset, predictions_mean, average='weighted')
    test_recall = recall_score(y_test_subset, predictions_mean, average='weighted')
    test_f_measure = f1_score(y_test_subset, predictions_mean, average='weighted')
    test_AUC = roc_auc_score(y_test_subset, predictions_mean)
    
    ensemble_accuracy[i] = test_acc
    ensemble_precison[i] = test_precision
    ensemble_recall[i] = test_recall
    ensemble_F1[i] = test_f_measure
    ensemble_AUC[i] = test_AUC


np.save(r"C:\Users\michi\Downloads\ensemble_accuracy_N=200v2", ensemble_accuracy)
np.save(r"C:\Users\michi\Downloads\ensemble_precision_N=200v2", ensemble_precison)
np.save(r"C:\Users\michi\Downloads\ensemble_recall_N=200v2", ensemble_recall)
np.save(r"C:\Users\michi\Downloads\ensemble_f1_N=200v2", ensemble_F1)
np.save(r"C:\Users\michi\Downloads\ensemble_AUC_N=200v2", ensemble_AUC)


epoch80_pred = D3_predictions[:,79,:]
example_min = epoch80_pred[:, 61445]
example_max = epoch80_pred[:, 14351]
example_doubt = epoch80_pred[:, 71677]


n, bins, patches = plt.hist(input, bins=30, edgecolor='black', alpha=0.7, label='Conversion prob.')
mu, std = norm.fit(input)
x = np.linspace(np.min(input), np.max(input), 100)
pdf = norm.pdf(x, mu, std)

# Plotting the PDF line
plt.plot(x, pdf * len(input) * np.diff(bins)[0], 'r-', label='Probability Distribution')
plt.axvline(x=0.5, linestyle='--', color='g', label='Conversion Boundary')

# Adding labels and title
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Boundary Case')
plt.legend()
plt.show()