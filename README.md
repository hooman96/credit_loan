# credit_loan
online advisor for approving loans based on credit history


### Import 


```python
import os
import pandas as pd
import numpy as np
import pylab as pl
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import warnings
# Don't need warnings when modifying data frame
warnings.filterwarnings('ignore')
```

### Read in data function


```python
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
```

### Read in training data


```python
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
```

### Split data into training and testing


```python
# Split data into training and test
np.random.seed(0)
msk = np.random.rand(len(train_df)) < 0.8
dev_train = train_df[msk]
dev_test = train_df[~msk]
```

### Get y values for classification


```python
def make_class_y(df):
    y = df.iloc[:,-1]
    return y.apply(lambda x: 0 if x == 0 else 1)
```


```python
train_y = make_class_y(dev_train)
test_y = make_class_y(dev_test)
```

### Get x values in classification


```python
def make_class_x(df):
    primary = ['f2', 'f471', 'f612', 'f536']
    x = df[primary]
    x['f527_minus_f528'] = df['f527'] - df['f528']
    x['f532_minus_f543'] = df['f532'] - df['f543']
    x['f532_minus_f556'] = df['f532'] - df['f556']
    x['logf271'] = np.log(df['f271']+1)
    return x
```


```python
train_x = make_class_x(dev_train)
test_x = make_class_x(dev_test)
```

### Train Random Forest Classifier


```python
classifier = RandomForestClassifier(n_estimators=200)
classifier.fit(train_x, train_y)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



### Run classifier on development set


```python
class_predict = classifier.predict(test_x)
accuracy = classifier.score(test_x, test_y)
print 'Classification Accuracy: ' + str(accuracy)
zeros_predicted = sum([x == y for (x, y) in zip(test_y, class_predict) if x == 0])
ones_predicted = sum([x == y for (x, y) in zip(test_y, class_predict) if x == 1])
total_zeros = sum([x == 0 for x in test_y])
total_ones = sum([x == 1 for x in test_y])
print 'Zero Accuracy: ' + str(float(zeros_predicted) / total_zeros)
print 'One Accuracy: ' + str(float(ones_predicted) / total_ones)
```

    Classification Accuracy: 0.9802995275907126
    Zero Accuracy: 0.989938080495
    One Accuracy: 0.883977900552
    

### Train regression on points with loss


```python
train_loss = dev_train.loc[dev_train['loss']>0]
test_loss = dev_test.loc[dev_test['loss']>0]
```

### Get x values in regression


```python
def make_regression_x(df):
    primary = ['f2','f471', 'f612', 'f536', 'f675', 'f282', 'f281', 'f400', 'f323', 'f322', 'f315', 'f22', 'f222', 'f596']
    x = df[primary]
    x['f527_minus_f528'] = df['f527'] - df['f528']
    x['f532_minus_f543'] = df['f532'] - df['f543']
    return x
```


```python
train_x = make_regression_x(train_loss)
test_x = make_regression_x(dev_test)
train_y = train_loss.iloc[:,-1]
test_y = dev_test.iloc[:,-1]
```

### Train Regression


```python
regression = GradientBoostingRegressor(n_estimators=200)
regression.fit(train_x, train_y)
```




    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=200, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False)




```python
loss_predict = regression.predict(test_x)
accuracy = regression.score(test_x, test_y)
print 'Regression Accuracy: ' + str(accuracy)
```

    Regression Accuracy: -14.290272301365375
    


```python
result = loss_predict * class_predict
print 'Training MAE: ' + str(sk.metrics.mean_absolute_error(dev_test.iloc[:, -1], result))
```

    Training MAE: 0.57372999960364
    

## Try with kaggle test data


```python
kaggle = pd.read_csv('test_final.csv')
```


```python
# Remove extra column
kaggle = kaggle.iloc[:, 1:]
```


```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>99935</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>93549</td>
      <td>29.603123</td>
    </tr>
    <tr>
      <th>2</th>
      <td>93212</td>
      <td>11.139434</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20496</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55762</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>93312</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>22411</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11046</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>61571</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6795</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>32584</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>94489</td>
      <td>5.315290</td>
    </tr>
    <tr>
      <th>12</th>
      <td>76672</td>
      <td>10.430246</td>
    </tr>
    <tr>
      <th>13</th>
      <td>95476</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>34056</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>29879</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>595</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>102693</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>95149</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>62682</td>
      <td>3.149753</td>
    </tr>
    <tr>
      <th>20</th>
      <td>60127</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>44476</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>84998</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>45264</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>22587</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>90173</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>11221</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>92673</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>59368</td>
      <td>8.571449</td>
    </tr>
    <tr>
      <th>29</th>
      <td>80874</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>55441</th>
      <td>45582</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55442</th>
      <td>54112</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55443</th>
      <td>44976</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55444</th>
      <td>47926</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55445</th>
      <td>1550</td>
      <td>5.227014</td>
    </tr>
    <tr>
      <th>55446</th>
      <td>72004</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55447</th>
      <td>61153</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55448</th>
      <td>52332</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55449</th>
      <td>15342</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55450</th>
      <td>39732</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55451</th>
      <td>39760</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55452</th>
      <td>96257</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>55453</th>
      <td>92311</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55454</th>
      <td>19925</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55455</th>
      <td>40098</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55456</th>
      <td>11325</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55457</th>
      <td>49137</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55458</th>
      <td>80262</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55459</th>
      <td>89529</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55460</th>
      <td>103696</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55461</th>
      <td>14397</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55462</th>
      <td>61782</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55463</th>
      <td>48992</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55464</th>
      <td>44350</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55465</th>
      <td>39575</td>
      <td>12.691817</td>
    </tr>
    <tr>
      <th>55466</th>
      <td>22762</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55467</th>
      <td>51790</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55468</th>
      <td>42888</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55469</th>
      <td>94812</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55470</th>
      <td>64130</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>55471 rows × 2 columns</p>
</div>


