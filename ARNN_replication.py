# Import packages

# Basic packages
import pandas as pd
import numpy as np

# Neural network
import torch
import torch.nn as nn
import torch.optim as optim

# Performance metrics
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Training
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

np.random.seed(530)

path_trainset = r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\01 Data\Data RenK\RenK trainset..tsv"
path_testset = r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\01 Data\Data RenK\RenK testset.tsv"

train_reshaped = pd.read_csv(path_trainset, sep='\t')
test_reshaped = pd.read_csv(path_testset, sep='\t')
train_reshaped = train_reshaped.drop(columns=train_reshaped.columns[0], axis=1)
test_reshaped = test_reshaped.drop(columns=test_reshaped.columns[0], axis=1)
train_reshaped = train_reshaped.values
test_reshaped = test_reshaped.values

div = 16

train = train_reshaped.reshape(train_reshaped.shape[0], train_reshaped.shape[1] // div, div)
test = test_reshaped.reshape(test_reshaped.shape[0], test_reshaped.shape[1] // div, div)

batch_train = np.load(r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\03 model output\ADASYN 33_conversion 93_acc\batch_train929.pt.npy", allow_pickle=True)
y_test = np.load(r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\03 model output\ADASYN 33_conversion 93_acc\y_test929.npy", allow_pickle=True)
batch_test = np.load(r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\03 model output\ADASYN 33_conversion 93_acc\batch_test929.pt.npy", allow_pickle=True)
batch_size = 1024

# Hyperparameters
input_size = 12 # Number of features
seq_len = train.shape[1] # Sequence length (27)
num_layers = 1 # Number of LSTMs stacked on top of each other
hidden_size = 32
num_classes = 1
learning_rate = 0.001
batch_size = 1024
num_epochs = 7 #256
N = train.shape[0]
encoder_output = 8

# Prepare train data
y_train = torch.tensor(train[:, 0:1, 1:2]).float()
y_train = torch.reshape(y_train, (-1,))
y_train = y_train.detach().numpy()

# Predicted labels
X_train = torch.tensor(train[:, :, 2:14]).float()

# Test y and X
y_test = torch.tensor(test[:, 0:1, 1:2]).float()
y_test = torch.reshape(y_test, (-1,))
y_test = y_test.detach().numpy()

# Predicted labels
X_test = torch.tensor(test[:, :, 2:14]).float()

# Split data into stratified batches on conversions
id_conv = train[:, 0, 0:2]
batch_train = []
batch_size = 1024

# Create stratified batches
while batch_size < id_conv.shape[0]:
  # Get stratified sample of train data
  id_conv, curr_batch = train_test_split(id_conv, 
                                         test_size=batch_size/id_conv.shape[0], 
                                         random_state=22, shuffle=True, 
                                         stratify=id_conv[:, 1])
  indices = np.isin(train[:, 0, 0], list(curr_batch[:, 0]))

  # Append to batch list
  batch_train.append(torch.tensor(train[indices, :, 1:14]).float())

# Get stratified sample of train data
indices = np.isin(train[:, 0, 0], list(id_conv[:, 0]))

# Append to batch list
batch_train.append(torch.tensor(train[indices, :, 1:14]).float())


# Round with manual bound
def manual_round(y_pred, bound):
  '''
  Returns y_pred rounded by a given bound.
  '''
  
  arr = y_pred.copy()
  arr[arr >= bound] = 1
  arr[arr < bound] = 0
  return arr

# Create attribution model network (ARNN)
class ARNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, encoder_output, num_classes, seq_len):
    super(ARNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers # Number of LSTMs
    self.encoder_output = encoder_output
    self.seq_len = seq_len
    self.encoder = nn.Sequential(
      nn.Linear(input_size, encoder_output), # Encoder layer
      nn.LeakyReLU(inplace=True)
    )

    self.lstm = nn.LSTM(encoder_output, hidden_size, num_layers, batch_first=True)
    self.attention = [] # Attention layer

    self.energy = nn.Sequential(
        nn.Linear(hidden_size + encoder_output, 16),
        nn.LeakyReLU(inplace=True),
        nn.Linear(16, 32),
        nn.LeakyReLU(inplace=True),
        nn.Linear(32, 1),
        nn.Tanh()
    )

    self.r = nn.Sequential(
        nn.Linear(hidden_size + encoder_output, 32),
        nn.LeakyReLU(inplace=True),
        nn.Linear(32, 64),
        nn.LeakyReLU(inplace=True),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
  
  def forward(self, x):
    # x is shaped as (batch_size, seq_len, input_size)
    
    # Encoder
    v = []
    for i in range(seq_len):
      v.append(self.encoder(x[:, i:i+1, :]))
    
    # LSTM
    
    # Initialisation
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
    h, h_lstm, c_lstm = [], [h0], [c0]

    # Loop through all time sequences
    for i in range(seq_len):
      out_curr, (h_curr, c_curr) = self.lstm(v[i], (h_lstm[i], c_lstm[i]))

      # Append lstm components for next step
      h_lstm.append(h_curr)
      c_lstm.append(c_curr)

      # Append reshaped hidden layers
      h.append(torch.reshape(h_curr, (batch_size, self.num_layers, self.hidden_size)))
   
    e = []

    # Loop through all hidden layers
    for i in range(seq_len):
      e.append(torch.exp(self.energy(torch.cat((h[i], v[-1]), 2))))

    e_sum = sum(e)
    a = []
    a = [e_curr / e_sum for e_curr in e]
    
    self.attention = [a_curr.detach().numpy() for a_curr in a]

    # Get final hidden layer product
    h_final = 0
    for i in range(seq_len):
      h_final += a[i] * h[i]

    # Conversion prediction
    v_last = torch.reshape(v[-1], (batch_size, 1, self.encoder_output))
    
    h_concat = torch.cat((h_final, v_last), 2)
    y_pred = self.r(h_concat)

    return y_pred



# Start timer
start = datetime.now()

# Print start time and parameters
print('Time: ' + str(start.strftime("%Y-%m-%d %H:%M:%S")))

# Initialize network
model = ARNN(input_size, hidden_size, num_layers, encoder_output, num_classes, seq_len)

# Initialise loss and optimiser
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# Initialise loss list
epoch_loss_list = []
accuracy_list = []
precision_list = []
recall_list = []
F_measure_list = []
AUC_list = []
precision_list_unw = []
recall_list_unw = []
F_measure_list_unw = []
AUC_list_unw = []

# Train network
for epoch in range(num_epochs):
  epoch_loss = 0
  y_pred_list = []

  for batch_num in tqdm(range(len(batch_train)), leave=False):
    # Select conversion indicator column (outcome)
    y_true = batch_train[batch_num][:, 0, 0]
    
    # Select features (explanatory variables)
    X = batch_train[batch_num][:, :, 1:]
    
    # Training part 1: Forward propagation
    y_pred = model(X)
    y_pred = torch.reshape(y_pred, (-1,))
    y_pred_tensor = y_pred
    y_pred_1 = y_pred.detach().numpy()
    y_pred_list.append(y_pred_1)
    
    
    # Training part 2: zero out the gradients
    optimiser.zero_grad()

    # Training part 3: calculate the model loss
    loss = criterion(y_pred_tensor, y_true)

    # Training part 4: Backward propagation
    loss.backward()

    # Training part 5: Take step
    optimiser.step()

    # Record epoch loss
    epoch_loss += loss.item() * y_true.shape[0] / N
  
  # Save epoch loss
  epoch_loss_list.append(epoch_loss)
  
  # Print loss
  print('Epoch {:3.0f} Loss {:.10f}'.format(epoch+1, epoch_loss))

  # Plot loss
  plt.figure()
  plt.plot(np.array(epoch_loss_list), 'r')
  plt.title('Log loss')
  plt.show()

  model.eval()
  
  y_pred_test_list = []
  y_pred_train = np.concatenate(y_pred_list, axis=0)
  y_test = torch.tensor(test[:, 0:1, 1:2]).float()
  y_test = torch.reshape(y_test, (-1,)) 
  y_test = y_test.detach().numpy()
    
  
  for batch_num in tqdm(range(len(batch_test)), leave=False):
    X_test = batch_test[batch_num]
    y_pred_test = model(X_test)
    y_pred_test = torch.reshape(y_pred_test, (-1,))
    y_pred_test = y_pred_test.detach().numpy()
    y_pred_test_list.append(y_pred_test)
    
  y_pred_test = np.concatenate(y_pred_test_list)
    
  
  
  train_scores_AUC = []

  # Loop over all bounds
  for i in range(100):
    train_scores_AUC.append(roc_auc_score(y_train, manual_round(y_pred_train, i/100), average='weighted'))

  # Get best bound
  max_value_AUC = max(train_scores_AUC)
  bound = train_scores_AUC.index(max_value_AUC)/100
  
  # Get rounded prediction array
  y_pred_test_rounded = manual_round(y_pred_test, bound)
  
  test_acc = accuracy_score(y_test, y_pred_test_rounded)
  accuracy_list.append(test_acc)
    
  test_precision = precision_score(y_test, y_pred_test_rounded, average='weighted')
  precision_list.append(test_precision)
    
  test_recall = recall_score(y_test, y_pred_test_rounded, average='weighted')
  recall_list.append(test_recall)
    
  test_f_measure = f1_score(y_test, y_pred_test_rounded, average='weighted')
  F_measure_list.append(test_f_measure)
  
  test_AUC = roc_auc_score(y_test, y_pred_test_rounded, average='weighted')
  AUC_list.append(test_AUC)
  
  test_precision_unw = precision_score(y_test, y_pred_test_rounded)
  precision_list_unw.append(test_precision_unw)
    
  test_recall_unw = recall_score(y_test, y_pred_test_rounded)
  recall_list_unw.append(test_recall_unw)
    
  test_f_measure_unw = f1_score(y_test, y_pred_test_rounded)
  F_measure_list_unw.append(test_f_measure_unw)
  
  test_AUC_unw = roc_auc_score(y_test, y_pred_test_rounded)
  AUC_list_unw.append(test_AUC_unw)
  

  plt.figure()
  plt.plot(np.array(accuracy_list), 'r')
  plt.title('Accuracy')
  plt.show()
    
  plt.figure()
  plt.plot(np.array(precision_list), 'r')
  plt.title('Precision')
  plt.show()
    
  plt.figure()
  plt.plot(np.array(recall_list), 'r')
  plt.title('Recall')
  plt.show()
    
  plt.figure()
  plt.plot(np.array(F_measure_list), 'r')
  plt.title('F-Measure')
  plt.show()
      
  
acc_array = np.array(accuracy_list)
np.save(r"C:\Users\pim\Downloads\acc_array_ARNN", acc_array)

precision_array = np.array(precision_list)
np.save(r"C:\Users\pim\Downloads\precision_array_ARNN", precision_array)

recall_array = np.array(recall_list)
np.save(r"C:\Users\pim\Downloads\recall_array_ARNN", recall_array)

Fmeasure_array = np.array(F_measure_list)
np.save(r"C:\Users\pim\Downloads\Fmeasure_array_ARNN", Fmeasure_array)

AUC_array = np.array(AUC_list)
np.save(r"C:\Users\pim\Downloads\AUC_array_ARNN", AUC_array)

precision_array_unw = np.array(precision_list_unw)
np.save(r"C:\Users\pim\Downloads\precision_array_unw_ARNN", precision_array_unw)

recall_array_unw = np.array(recall_list_unw)
np.save(r"C:\Users\pim\Downloads\recall_array_unw_ARNN", recall_array_unw)

Fmeasure_array_unw = np.array(F_measure_list_unw)
np.save(r"C:\Users\pim\Downloads\Fmeasure_array_unw_ARNN", Fmeasure_array_unw)

AUC_array_unw = np.array(AUC_list_unw)
np.save(r"C:\Users\pim\Downloads\AUC_array_unw_ARNN", AUC_array_unw)

 
model.eval()

# Add empty space  
print()

# Record and print runtime
time_elapsed = datetime.now() - start
print('Time elapsed: ' + str(time_elapsed))


# Test y and X
# True labels
y_test = torch.tensor(test[:, 0:1, 1:2]).float()
y_test = torch.reshape(y_test, (-1,))
y_test = y_test.detach().numpy()

# Predicted labels
X_test = torch.tensor(test[:, :, 2:14]).float()
start = torch.cat([batch_train[0], batch_train[1]], dim=0)
X_train = start

for i in range(len(batch_train)-2):
  concat = batch_train[i + 2]
  X_train = torch.cat([X_train, concat], dim=0)

y_train = X_train[:, 0, 0]
X_train = X_train[:, :, 1:]


# Train results
y_pred_train = model(X_train)
y_pred_train = torch.reshape(y_pred_train, (-1,))
y_pred_train = y_pred_train.detach().numpy()

# Test results
y_pred_test = model(X_test)
y_pred_test = torch.reshape(y_pred_test, (-1,))
y_pred_test = y_pred_test.detach().numpy()


train_scores_AUC = []

# Loop over all bounds
for i in range(100):
  train_scores_AUC.append(roc_auc_score(y_train, manual_round(y_pred_train, i/100), average='weighted'))

# Get best bound
max_value_AUC = max(train_scores_AUC)
bound = train_scores_AUC.index(max_value_AUC)/100

# Get rounded prediction array
y_pred_test_rounded = manual_round(y_pred_test, bound)

test_log_loss = log_loss(y_test, y_pred_test, eps=0.0000001)
test_recall = recall_score(y_test, y_pred_test_rounded, average='weighted')
test_precision = precision_score(y_test, y_pred_test_rounded, average='weighted')
test_acc = accuracy_score(y_test, y_pred_test_rounded)
test_f_measure = f1_score(y_test, y_pred_test_rounded, average='weighted')
test_AUC = roc_auc_score(y_test, y_pred_test_rounded, average='weighted')

# Save results
txt_results = 'Training time: ' + str(time_elapsed) + '\n' + \
              'Log-loss: ' + str(test_log_loss) + '\n' + \
              'Recall: ' + str(test_recall) + '\n' + \
              'Precision: ' + str(test_precision) + '\n' + \
              'Accuracy: ' + str(test_acc) + '\n' + \
              'F-measure: ' + str(test_f_measure) + '\n' + \
              'AUC: ' + str(test_AUC)

print(txt_results)