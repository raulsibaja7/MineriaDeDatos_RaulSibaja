

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
```


```python

path = 'avocado.csv'
file = path

# Cargamos el conjunto de datos en un dataframe
all_variables = ['Number','AveragePrice','Total Volume','Volume Central','Volume South','Volume North','Total Bags','Small Bags','Large Bags','XLarge Bags','year'] #ojo en el tarjet
features = ['Number','AveragePrice','Total Volume','Volume Central','Volume South','Volume North','Total Bags','Small Bags','Large Bags','XLarge Bags','year']
target = ['Total Volume']

df = pd.read_csv(file, names=all_variables)
df.head()
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
      <th>Number</th>
      <th>AveragePrice</th>
      <th>Total Volume</th>
      <th>Volume Central</th>
      <th>Volume South</th>
      <th>Volume North</th>
      <th>Total Bags</th>
      <th>Small Bags</th>
      <th>Large Bags</th>
      <th>XLarge Bags</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Number</td>
      <td>AveragePrice</td>
      <td>Total Volume</td>
      <td>Volume Central</td>
      <td>Volume South</td>
      <td>Volume North</td>
      <td>Total Bags</td>
      <td>Small Bags</td>
      <td>Large Bags</td>
      <td>XLarge Bags</td>
      <td>year</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1.33</td>
      <td>64236.62</td>
      <td>1036.74</td>
      <td>54454.85</td>
      <td>48.16</td>
      <td>8696.87</td>
      <td>8603.62</td>
      <td>93.25</td>
      <td>0</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1.35</td>
      <td>54876.98</td>
      <td>674.28</td>
      <td>44638.81</td>
      <td>58.33</td>
      <td>9505.56</td>
      <td>9408.07</td>
      <td>97.49</td>
      <td>0</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0.93</td>
      <td>118220.22</td>
      <td>794.7</td>
      <td>109149.67</td>
      <td>130.5</td>
      <td>8145.35</td>
      <td>8042.21</td>
      <td>103.14</td>
      <td>0</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1.08</td>
      <td>78992.15</td>
      <td>1132</td>
      <td>71976.41</td>
      <td>72.58</td>
      <td>5811.16</td>
      <td>5677.4</td>
      <td>133.76</td>
      <td>0</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_sc = pd.DataFrame(scaler.fit_transform(df[target]),
                 columns=features)
df_sc.head()


```python
#
pca = PCA()
df_pca = pd.DataFrame(pca.fit_transform(df[features]),
                     columns=features)
df_pca.head()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-77-ce055ed5baf9> in <module>
          1 #
          2 pca = PCA()
    ----> 3 df_pca = pd.DataFrame(pca.fit_transform(df[features]),
          4                      columns=features)
          5 df_pca.head()


    ~/dev/anaconda3/lib/python3.7/site-packages/sklearn/decomposition/pca.py in fit_transform(self, X, y)
        357 
        358         """
    --> 359         U, S, V = self._fit(X)
        360         U = U[:, :self.n_components_]
        361 


    ~/dev/anaconda3/lib/python3.7/site-packages/sklearn/decomposition/pca.py in _fit(self, X)
        379 
        380         X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
    --> 381                         copy=self.copy)
        382 
        383         # Handle n_components==None


    ~/dev/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        525             try:
        526                 warnings.simplefilter('error', ComplexWarning)
    --> 527                 array = np.asarray(array, dtype=dtype, order=order)
        528             except ComplexWarning:
        529                 raise ValueError("Complex data not supported\n"


    ~/dev/anaconda3/lib/python3.7/site-packages/numpy/core/numeric.py in asarray(a, dtype, order)
        536 
        537     """
    --> 538     return array(a, dtype, copy=False, order=order)
        539 
        540 


    ValueError: could not convert string to float: 'Number'



```python
explained_variance = pca.explained_variance_ratio_
explained_variance
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-78-515d78cc7231> in <module>
    ----> 1 explained_variance = pca.explained_variance_ratio_
          2 explained_variance


    AttributeError: 'PCA' object has no attribute 'explained_variance_ratio_'



```python
df_pca['target'] = df[target]
df_pca.columns = ['PC1', 'PC2','PC3','PC4','target']
df_pca.head()
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~/dev/anaconda3/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2656             try:
    -> 2657                 return self._engine.get_loc(key)
       2658             except KeyError:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'Iris-virginica'

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-79-e2c09b975068> in <module>
    ----> 1 df_pca['target'] = df[target]
          2 df_pca.columns = ['PC1', 'PC2','PC3','PC4','target']
          3 df_pca.head()


    ~/dev/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2925             if self.columns.nlevels > 1:
       2926                 return self._getitem_multilevel(key)
    -> 2927             indexer = self.columns.get_loc(key)
       2928             if is_integer(indexer):
       2929                 indexer = [indexer]


    ~/dev/anaconda3/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2657                 return self._engine.get_loc(key)
       2658             except KeyError:
    -> 2659                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2660         indexer = self.get_indexer([key], method=method, tolerance=tolerance)
       2661         if indexer.ndim > 1 or indexer.size > 1:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'Iris-virginica'



```python
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1') 
ax.set_ylabel('Principal Component 2') 
ax.set_title('2 component PCA') 
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = df_pca['target'] == target
    ax.scatter(df_pca.loc[indicesToKeep, 'PC1']
    , df_pca.loc[indicesToKeep, 'PC2']
    , c = color
    , s = 50)
ax.legend(targets)
ax.grid()
```


![png](output_6_0.png)



```python

np.random.seed(0)
X, y = make_circles(n_samples=400, factor=.3, noise=.05)

plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1, aspect='equal')
plt.title("Espacio original")
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red",s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid()
```


![png](output_7_0.png)



```python
kpca = KernelPCA(kernel = "rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X)

plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",s=20, edgecolor='k')
x = np.linspace(-1, 1, 1000)
plt.plot(x, -0.1*x, linestyle="solid")
plt.title("Proyección de KPCA")
plt.xlabel("Primer componente principal inducido por $\phi$")
plt.ylabel("Segundo componente principal")
plt.grid()
```


![png](output_8_0.png)



```python

```


```python

```


```python

```


```python

```




```python

```


```python

```
