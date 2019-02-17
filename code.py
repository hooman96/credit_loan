
# coding: utf-8

# ### Import 

# In[1]:


import os
import pandas as pd
import numpy as np
import pylab as pl
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import warnings
# Don't need warnings when modifying data frame
warnings.filterwarnings('ignore')


# ### Read in data function

# In[2]:


def read_data(path):
    data = np.load(path)
    column_name = data[0].decode('UTF-8').strip().split(',')
    data_list = []
    for train in data[1:]:
        tmp = train.decode('UTF-8').split(',')
        tmp = [0 if t == 'NA' else float(t) for t in tmp]
        data_list.append(tmp)

    df = pd.DataFrame(data=data_list, columns=column_name)

    return df


# ### Read in training data

# In[3]:


# Read in data
if os.path.exists('train.df'):
    train_df = pd.read_pickle('train.df')
else:
    train_df.to_pickle('train.df')

#remove columns with same values in all rows
nunique = train_df.apply(pd.Series.nunique)
drop_cols = nunique[nunique == 1].index
train_df.drop(drop_cols,axis=1,inplace=True)
 
# Remove columns with duplicates
cor_matrix = np.corrcoef(train_df, rowvar=False)
cor_matrix = pd.DataFrame(data=cor_matrix, index=list(train_df), columns=list(train_df))
cor_matrix = cor_matrix.abs()
high_cor_col=np.where(cor_matrix== 1)
high_cor_col=[(cor_matrix.columns[x], cor_matrix.columns[y]) for x,y in zip(*high_cor_col) if x!=y and x<y]
drop_cols = set([x[1] for x in high_cor_col])
train_df.drop(drop_cols,axis=1,inplace=True)


# ### Split data into training and testing

# In[4]:


# Split data into training and test
np.random.seed(0)
msk = np.random.rand(len(train_df)) < 0.8
dev_train = train_df[msk]
dev_test = train_df[~msk]


# ### Get y values for classification

# In[5]:


def make_class_y(df):
    y = df.iloc[:,-1]
    return y.apply(lambda x: 0 if x == 0 else 1)


# In[6]:


train_y = make_class_y(dev_train)
test_y = make_class_y(dev_test)


# ### Get x values in classification

# In[7]:


def make_class_x(df):
    primary = ['f2', 'f471', 'f612', 'f536']
    x = df[primary]
    x['f527_minus_f528'] = df['f527'] - df['f528']
    x['f532_minus_f543'] = df['f532'] - df['f543']
    x['f532_minus_f556'] = df['f532'] - df['f556']
    x['logf271'] = np.log(df['f271']+1)
    return x


# In[8]:


train_x = make_class_x(dev_train)
test_x = make_class_x(dev_test)


# ### Train Random Forest Classifier

# In[9]:


classifier = RandomForestClassifier(n_estimators=200)
classifier.fit(train_x, train_y)


# ### Run classifier on development set

# In[10]:


class_predict = classifier.predict(test_x)
accuracy = classifier.score(test_x, test_y)
print 'Classification Accuracy: ' + str(accuracy)
zeros_predicted = sum([x == y for (x, y) in zip(test_y, class_predict) if x == 0])
ones_predicted = sum([x == y for (x, y) in zip(test_y, class_predict) if x == 1])
total_zeros = sum([x == 0 for x in test_y])
total_ones = sum([x == 1 for x in test_y])
print 'Zero Accuracy: ' + str(float(zeros_predicted) / total_zeros)
print 'One Accuracy: ' + str(float(ones_predicted) / total_ones)


# ### Train regression on points with loss

# In[11]:


train_loss = dev_train.loc[dev_train['loss']>0]
test_loss = dev_test.loc[dev_test['loss']>0]


# ### Get x values in regression

# In[12]:


def make_regression_x(df):
    primary = ['f2','f471', 'f612', 'f536', 'f675', 'f282', 'f281', 'f400', 'f323', 'f322', 'f315', 'f22', 'f222', 'f596']
    x = df[primary]
    x['f527_minus_f528'] = df['f527'] - df['f528']
    x['f532_minus_f543'] = df['f532'] - df['f543']
    return x


# In[13]:


train_x = make_regression_x(train_loss)
test_x = make_regression_x(dev_test)
train_y = train_loss.iloc[:,-1]
test_y = dev_test.iloc[:,-1]


# ### Train Regression

# In[14]:


regression = GradientBoostingRegressor(n_estimators=200)
regression.fit(train_x, train_y)


# In[15]:


loss_predict = regression.predict(test_x)
accuracy = regression.score(test_x, test_y)
print 'Regression Accuracy: ' + str(accuracy)


# In[16]:


result = loss_predict * class_predict
print 'Training MAE: ' + str(sk.metrics.mean_absolute_error(dev_test.iloc[:, -1], result))


# ## Try with kaggle test data

# In[17]:


kaggle = pd.read_csv('test_final.csv')


# In[18]:


# Remove extra column
kaggle = kaggle.iloc[:, 1:]


# In[19]:


# Make classification
test_x = make_class_x(kaggle)
class_predict = classifier.predict(test_x)

# Make regression
test_x = make_regression_x(kaggle)
loss_predict = regression.predict(test_x)
result = class_predict*loss_predict

# Get final output
kaggle['loss'] = result
final_result = kaggle.ix[:, ['id', 'loss']]
final_result.to_csv('final_results.csv', sep=',', header=True, index=False)
final_result

