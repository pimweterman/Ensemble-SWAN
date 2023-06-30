import pandas as pd
import numpy as np
import random
# Neural network
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

np.random.seed(530)
random.seed(530)

path_raw_data = r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\01 Data\raw data thesis.tsv"
criteo_data = pd.read_csv(path_raw_data, sep='\t')

# Create unique sequence id
criteo_data['temp_id'] = criteo_data['uid'].astype(str) + '_' + \
                    criteo_data['conversion_id'].astype(str)
                    
# Compute sequence length and rename column
seq_len = criteo_data.groupby('temp_id').size().reset_index()
seq_len.rename(columns={0: 'seq_len'}, inplace=True)

# Remove sequence length shorter than shortest and longest variables
shortest = 3
longest = 20

seq_len = seq_len[seq_len['seq_len'] >= shortest]
seq_len = seq_len[seq_len['seq_len'] <= longest]

# Get conversions
id_conv = criteo_data[['temp_id', 'conversion']]
seq_len = seq_len.merge(id_conv, on='temp_id', how='left')

# Get unique conversion and nonconversion ids
conv_list = list(seq_len[seq_len['conversion'] == 1]['temp_id'].unique())
noconv_list = list(seq_len[seq_len['conversion'] == 0]['temp_id'].unique())

# Shuffle
random.shuffle(conv_list)
random.shuffle(noconv_list)

# Subsample conversions to have a ratio of 1:20 of conversions to nonconversions
noconv_list = noconv_list[:len(conv_list)*20]

# Get a list of all ids in the subsample
id_list = conv_list + noconv_list

# Extract data with defined sequence length
df_foc = criteo_data[criteo_data['temp_id'].isin(id_list)]
df_foc.shape

path_subsampled = r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\01 Data\Data RenK\Data subsampled.tsv"
df_foc.to_csv(path_subsampled, sep='\t')

#______________________________ Data preprocessing ___________________________________

criteo_raw = df_foc

# Create temporary unique sequence id of type str
criteo_raw['temp_id'] = criteo_raw['uid'].astype(str) + '_' + \
                    criteo_raw['conversion_id'].astype(str)

# Extract final dataframe for modelling
final_data = criteo_raw[['temp_id', 'conversion', 'timestamp', 
             'time_since_last_click', 'campaign', 
             'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 
             'cat6', 'cat7', 'cat8', 'cat9', 'cost']]

# Check if there are any duplicates
assert((final_data.groupby('temp_id')['conversion'].nunique()!=1).sum() == 0)

# Compute sequence length and renaming the corresponding column
seq_len = final_data.groupby('temp_id').size().reset_index()
seq_len.rename(columns={0: 'seq_len'}, inplace=True)

# Left join sequence lengths to df
final_data = final_data.merge(seq_len, on='temp_id', how='left')

# Remove sequence length shorter than 3 and longer than 20
shortest = 3
longest = 20

seq_len = seq_len[seq_len['seq_len'] >= shortest]
seq_len = seq_len[seq_len['seq_len'] <= longest]

# Get a list of ids for the specified sequence lengths
id_list = list(seq_len['temp_id'].unique())

print(final_data.shape)

# Extract data with specified sequence length
final_data = final_data[final_data['temp_id'].isin(id_list)]

for length in range(shortest, longest):
  # Get unique ids for this sequence length
  ids = final_data[final_data['seq_len'] == length][['temp_id', 'conversion']].drop_duplicates()


  pad_cols = ['timestamp', 'time_since_last_click', 'cat1', 'cat2', 
              'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']
  # Create padding DataFrame for the current sequence length
  zeros = pd.DataFrame(np.zeros([longest-length, len(pad_cols)]), 
                      columns=pad_cols)
  zeros[:] = -1
  zeros['campaign'] = -1
  zeros['seq_len'] = length
  zeros['cost'] = 0

  # Perform Cartesian product on ids and zeros to generate missing rows
  ids['key'] = 0
  zeros['key'] = 0
  padding = ids.merge(zeros, on='key')

  # Drop key columns used for Cartesian product
  padding.drop('key', axis=1, inplace=True)

  final_data = pd.concat([final_data, padding])
  

# Drop sequence length column
final_data.drop('seq_len', axis=1, inplace=True)

print(final_data.shape)

# Double check that all sequences are of length 20
# Compute sequence length and renaming the corresponding column
seq_len = final_data.groupby('temp_id').size().reset_index()
seq_len.rename(columns={0: 'seq_len'}, inplace=True)

# Left join sequence lengths to df
final_data = final_data.merge(seq_len, on='temp_id', how='left')

# Check all sequences are of correct length
assert(final_data['seq_len'].unique() == longest)

# Drop sequence length column
final_data.drop('seq_len', axis=1, inplace=True)

# Check if there are any duplicates
assert((final_data.groupby('temp_id')['conversion'].nunique()!=1).sum() == 0)

# Create unique id as an int
# Extract unique sequences
unique_ids = final_data['temp_id'].drop_duplicates()

# Reset the index
unique_ids = unique_ids.reset_index(drop=True).to_frame()

# Generate unique ids that are integers (use index as unique id here)
unique_ids.reset_index(inplace=True)

# Rename index to id
unique_ids.rename(columns={'index': 'id'}, inplace=True)

# Left join unique_ids to raw_df to have the id column there
final_data = final_data.merge(unique_ids, on='temp_id', how='left')

# Check id 
assert(final_data['id'].nunique() == final_data['temp_id'].nunique())

# Remove temporary id
final_data.drop(columns='temp_id', inplace=True)

# Check if there are any duplicates
assert((final_data.groupby('id')['conversion'].nunique()!=1).sum() == 0)

# Reorder columns
final_data = final_data[['id', 'conversion', 'timestamp', 'time_since_last_click', 'campaign', 
         'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 
         'cost']]

# Extract id, conversion and campaign columns for train test split
id_conv = final_data[['id', 'conversion']].drop_duplicates()

# Train test split 80:20
train, test = train_test_split(
    id_conv,
    test_size=0.20, random_state=22, shuffle=True, 
    stratify=id_conv[['conversion']])

# Create a list of train and test ids
train_ids = list(train['id'])
test_ids = list(test['id'])

# Extract train and test dataframes
train_df = final_data[final_data['id'].isin(train_ids)]
test_df = final_data[final_data['id'].isin(test_ids)]

# Check if there are any duplicates
assert((train.groupby('id')['conversion'].nunique()!=1).sum() == 0)
assert((test.groupby('id')['conversion'].nunique()!=1).sum() == 0)

assert(final_data.id.unique().shape[0] == 
       (train_df.id.unique().shape[0] + test_df.id.unique().shape[0]))
assert(final_data.id.unique().shape[0] == 
       (len(train_ids) + len(test_ids)))

print('Full dataset: ' + str(final_data.shape))
print('Train dataset: ' + str(train_df.shape))
print('Test dataset: ' + str(test_df.shape))
print('Unique ids in full dataset: ' + str(final_data.id.unique().shape))
print('Unique ids in train dataset: ' + str(train_df.id.unique().shape))
print('Unique ids in test dataset: ' + str(test_df.id.unique().shape))


# Create train and test arrays which have shape (id, seq_len, features)
# Sort rows by timestamp and cat1 such that padded values are at the back
train_df = train_df.sort_values(by = ['id', 'timestamp', 'cat1'], ascending=False, ignore_index=True)
test_df = test_df.sort_values(by = ['id', 'timestamp', 'cat1'], ascending=False, ignore_index=True)

# Get train and test unique ids
train_ids = list(train_df.id.unique())
test_ids = list(test_df.id.unique())

# Empty train set numpy array
train = np.zeros((len(train_ids), longest, final_data.shape[1]))

# Loop through each train_id and insert it into train set
for i, train_id in enumerate(train_ids):
  train[i, :, :] = train_df[train_df.id == train_id].values

# Empty test set numpy array
test = np.zeros((len(test_ids), longest, final_data.shape[1]))

# Loop through each train_id and insert it into train_set
for i, test_id in enumerate(test_ids):
  test[i, :, :] = test_df[test_df.id == test_id].values
  
  
# Checking if there are any duplicates.
# Check w.r.t. numbers in second cell to the top.
assert(np.unique(train_ids).shape[0] == np.unique(train[:, 0, 0]).shape[0])
assert(np.unique(test_ids).shape[0] == np.unique(test[:, 0, 0]).shape[0])

print('Unique train ids: ' + str(np.unique(train_ids).shape[0]))
print('Unique test ids: ' + str(np.unique(test_ids).shape[0]))
print('Unique ids in train dataset: ' + str(np.unique(train[:, 0, 0]).shape[0]))
print('Unique ids in test dataset: ' + str(np.unique(test[:, 0, 0]).shape[0]))

# Get a duplicate of campaign in the last column of train dataset
train = np.concatenate((train, train[:, :, 4:5]), axis=2)

# Get a duplicate of campaign in the last column of test dataset
test = np.concatenate((test, test[:, :, 4:5]), axis=2)

# Normalisation parameters
v_min = train[:, :, :].min(axis=(0, 1), keepdims=True)
v_max = train[:, :, :].max(axis=(0, 1), keepdims=True)

# Not modifying id, conversion, cost and second campaign columns
v_min[0, 0, 0], v_max[0, 0, 0] = 0, 1 # id
v_min[0, 0, 1], v_max[0, 0, 1] = 0, 1 # conversion
v_min[0, 0, 14], v_max[0, 0, 14] = 0, 1 # cost
v_min[0, 0, 15], v_max[0, 0, 15] = 0, 1 # campaign

# Save arrays that should not be modified
train_check = train[:, :, [0, 1, 14, 15]]
test_check = test[:, :, [0, 1, 14, 15]]

# Normalise
train = (train - v_min) / (v_max - v_min)
test = (test - v_min) / (v_max - v_min)

# Check if id, conversion cost and second campaign columns were not changed
assert((train[:, :, [0, 1, 14, 15]] == train_check).all())
assert((test[:, :, [0, 1, 14, 15]] == test_check).all())

path_trainset = r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\01 Data\Data RenK\RenK trainset..tsv"
path_testset = r"C:\Users\pim\OneDrive\Documents\Studie\Master BA & QM\Thesis\01 Data\Data RenK\RenK testset.tsv"

# Save train array
  
# Reshape 3D array to 2D
train_reshaped = train.reshape(train.shape[0], -1)

# Save reshaped array to file
pd.DataFrame(train_reshaped).to_csv(path_trainset, 
                sep='\t')

# Save test array

# Reshape 3D array to 2D
test_reshaped = test.reshape(test.shape[0], -1)

# Save reshaped array to file
pd.DataFrame(test_reshaped).to_csv(path_testset, 
                sep='\t')


