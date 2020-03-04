```python
conda install -c conda-forge/label/gcc7 missingno
```

    Collecting package metadata (current_repodata.json): done
    Solving environment: done
    
    # All requested packages already installed.
    
    
    Note: you may need to restart the kernel to use updated packages.



```python
import pandas as pd
import numpy as np
import missingno as msno
%matplotlib inline
```


```python
#https://www.kaggle.com/arindam235/startup-investments-crunchbase
path = 'investments_VC.csv'
df = pd.read_csv(path, encoding='latin', error_bad_lines=False)
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>/organization/waywire</td>
      <td>#waywire</td>
      <td>http://www.waywire.com</td>
      <td>|Entertainment|Politics|Social Media|News|</td>
      <td>News</td>
      <td>17,50,000</td>
      <td>acquired</td>
      <td>USA</td>
      <td>NY</td>
      <td>New York City</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>/organization/tv-communications</td>
      <td>&amp;TV Communications</td>
      <td>http://enjoyandtv.com</td>
      <td>|Games|</td>
      <td>Games</td>
      <td>40,00,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>CA</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>/organization/rock-your-paper</td>
      <td>'Rock' Your Paper</td>
      <td>http://www.rockyourpaper.org</td>
      <td>|Publishing|Education|</td>
      <td>Publishing</td>
      <td>40,000</td>
      <td>operating</td>
      <td>EST</td>
      <td>NaN</td>
      <td>Tallinn</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>/organization/in-touch-network</td>
      <td>(In)Touch Network</td>
      <td>http://www.InTouchNetwork.com</td>
      <td>|Electronics|Guides|Coffee|Restaurants|Music|i...</td>
      <td>Electronics</td>
      <td>15,00,000</td>
      <td>operating</td>
      <td>GBR</td>
      <td>NaN</td>
      <td>London</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>/organization/r-ranch-and-mine</td>
      <td>-R- Ranch and Mine</td>
      <td>NaN</td>
      <td>|Tourism|Entertainment|Games|</td>
      <td>Tourism</td>
      <td>60,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>TX</td>
      <td>Dallas</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 54294 entries, 0 to 54293
    Data columns (total 39 columns):
    permalink               49438 non-null object
    name                    49437 non-null object
    homepage_url            45989 non-null object
    category_list           45477 non-null object
     market                 45470 non-null object
     funding_total_usd      49438 non-null object
    status                  48124 non-null object
    country_code            44165 non-null object
    state_code              30161 non-null object
    region                  44165 non-null object
    city                    43322 non-null object
    funding_rounds          49438 non-null float64
    founded_at              38554 non-null object
    founded_month           38482 non-null object
    founded_quarter         38482 non-null object
    founded_year            38482 non-null float64
    first_funding_at        49438 non-null object
    last_funding_at         49438 non-null object
    seed                    49438 non-null float64
    venture                 49438 non-null float64
    equity_crowdfunding     49438 non-null float64
    undisclosed             49438 non-null float64
    convertible_note        49438 non-null float64
    debt_financing          49438 non-null float64
    angel                   49438 non-null float64
    grant                   49438 non-null float64
    private_equity          49438 non-null float64
    post_ipo_equity         49438 non-null float64
    post_ipo_debt           49438 non-null float64
    secondary_market        49438 non-null float64
    product_crowdfunding    49438 non-null float64
    round_A                 49438 non-null float64
    round_B                 49438 non-null float64
    round_C                 49438 non-null float64
    round_D                 49438 non-null float64
    round_E                 49438 non-null float64
    round_F                 49438 non-null float64
    round_G                 49438 non-null float64
    round_H                 49438 non-null float64
    dtypes: float64(23), object(16)
    memory usage: 16.2+ MB



```python
df.shape
```




    (54294, 39)




```python
df.shape[0]
```




    54294




```python
df.shape[1]
```




    39




```python
df.columns.values.tolist()
```




    ['permalink',
     'name',
     'homepage_url',
     'category_list',
     ' market ',
     ' funding_total_usd ',
     'status',
     'country_code',
     'state_code',
     'region',
     'city',
     'funding_rounds',
     'founded_at',
     'founded_month',
     'founded_quarter',
     'founded_year',
     'first_funding_at',
     'last_funding_at',
     'seed',
     'venture',
     'equity_crowdfunding',
     'undisclosed',
     'convertible_note',
     'debt_financing',
     'angel',
     'grant',
     'private_equity',
     'post_ipo_equity',
     'post_ipo_debt',
     'secondary_market',
     'product_crowdfunding',
     'round_A',
     'round_B',
     'round_C',
     'round_D',
     'round_E',
     'round_F',
     'round_G',
     'round_H']




```python
df.dtypes
```




    permalink                object
    name                     object
    homepage_url             object
    category_list            object
     market                  object
     funding_total_usd       object
    status                   object
    country_code             object
    state_code               object
    region                   object
    city                     object
    funding_rounds          float64
    founded_at               object
    founded_month            object
    founded_quarter          object
    founded_year            float64
    first_funding_at         object
    last_funding_at          object
    seed                    float64
    venture                 float64
    equity_crowdfunding     float64
    undisclosed             float64
    convertible_note        float64
    debt_financing          float64
    angel                   float64
    grant                   float64
    private_equity          float64
    post_ipo_equity         float64
    post_ipo_debt           float64
    secondary_market        float64
    product_crowdfunding    float64
    round_A                 float64
    round_B                 float64
    round_C                 float64
    round_D                 float64
    round_E                 float64
    round_F                 float64
    round_G                 float64
    round_H                 float64
    dtype: object




```python
df.isnull().any().any()
```




    True




```python
msno.matrix(df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f83f687f6d0>




![png](output_10_1.png)



```python
df.replace({' ': np.nan}, inplace=True)
```


```python
df.isnull().any().any()
```




    True




```python
msno.matrix(df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f83f5c70a10>




![png](output_13_1.png)



```python
msno.bar(df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f83f661e1d0>




![png](output_14_1.png)



```python
df.columns[df.isnull().any()].tolist()
```




    ['permalink',
     'name',
     'homepage_url',
     'category_list',
     ' market ',
     ' funding_total_usd ',
     'status',
     'country_code',
     'state_code',
     'region',
     'city',
     'funding_rounds',
     'founded_at',
     'founded_month',
     'founded_quarter',
     'founded_year',
     'first_funding_at',
     'last_funding_at',
     'seed',
     'venture',
     'equity_crowdfunding',
     'undisclosed',
     'convertible_note',
     'debt_financing',
     'angel',
     'grant',
     'private_equity',
     'post_ipo_equity',
     'post_ipo_debt',
     'secondary_market',
     'product_crowdfunding',
     'round_A',
     'round_B',
     'round_C',
     'round_D',
     'round_E',
     'round_F',
     'round_G',
     'round_H']




```python
df.describe()
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
      <th>funding_rounds</th>
      <th>founded_year</th>
      <th>seed</th>
      <th>venture</th>
      <th>equity_crowdfunding</th>
      <th>undisclosed</th>
      <th>convertible_note</th>
      <th>debt_financing</th>
      <th>angel</th>
      <th>grant</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>49438.000000</td>
      <td>38482.000000</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>...</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
      <td>4.943800e+04</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.696205</td>
      <td>2007.359129</td>
      <td>2.173215e+05</td>
      <td>7.501051e+06</td>
      <td>6.163322e+03</td>
      <td>1.302213e+05</td>
      <td>2.336410e+04</td>
      <td>1.888157e+06</td>
      <td>6.541898e+04</td>
      <td>1.628453e+05</td>
      <td>...</td>
      <td>3.845592e+04</td>
      <td>7.074227e+03</td>
      <td>1.243955e+06</td>
      <td>1.492891e+06</td>
      <td>1.205356e+06</td>
      <td>7.375261e+05</td>
      <td>3.424682e+05</td>
      <td>1.697692e+05</td>
      <td>5.767067e+04</td>
      <td>1.423197e+04</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.294213</td>
      <td>7.579203</td>
      <td>1.056985e+06</td>
      <td>2.847112e+07</td>
      <td>1.999048e+05</td>
      <td>2.981404e+06</td>
      <td>1.432046e+06</td>
      <td>1.382046e+08</td>
      <td>6.582908e+05</td>
      <td>5.612088e+06</td>
      <td>...</td>
      <td>3.864461e+06</td>
      <td>4.282166e+05</td>
      <td>5.531974e+06</td>
      <td>7.472704e+06</td>
      <td>7.993592e+06</td>
      <td>9.815218e+06</td>
      <td>5.406915e+06</td>
      <td>6.277905e+06</td>
      <td>5.252312e+06</td>
      <td>2.716865e+06</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>1902.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>1.000000</td>
      <td>2010.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2.000000</td>
      <td>2012.000000</td>
      <td>2.500000e+04</td>
      <td>5.000000e+06</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>max</td>
      <td>18.000000</td>
      <td>2014.000000</td>
      <td>1.300000e+08</td>
      <td>2.351000e+09</td>
      <td>2.500000e+07</td>
      <td>2.924328e+08</td>
      <td>3.000000e+08</td>
      <td>3.007950e+10</td>
      <td>6.359026e+07</td>
      <td>7.505000e+08</td>
      <td>...</td>
      <td>6.806116e+08</td>
      <td>7.200000e+07</td>
      <td>3.190000e+08</td>
      <td>5.420000e+08</td>
      <td>4.900000e+08</td>
      <td>1.200000e+09</td>
      <td>4.000000e+08</td>
      <td>1.060000e+09</td>
      <td>1.000000e+09</td>
      <td>6.000000e+08</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 23 columns</p>
</div>




```python
df.loc[[0]]
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>/organization/waywire</td>
      <td>#waywire</td>
      <td>http://www.waywire.com</td>
      <td>|Entertainment|Politics|Social Media|News|</td>
      <td>News</td>
      <td>17,50,000</td>
      <td>acquired</td>
      <td>USA</td>
      <td>NY</td>
      <td>New York City</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 39 columns</p>
</div>




```python
df.loc[30:33]
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>30</td>
      <td>/organization/10bestthings</td>
      <td>10BestThings</td>
      <td>http://10bestthings.com</td>
      <td>|Curated Web|</td>
      <td>Curated Web</td>
      <td>50,000</td>
      <td>closed</td>
      <td>USA</td>
      <td>OH</td>
      <td>Cleveland</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>31</td>
      <td>/organization/10sec</td>
      <td>10sec</td>
      <td>http://10s.ec/</td>
      <td>|Social Commerce|E-Commerce|Mobile Commerce|</td>
      <td>Mobile Commerce</td>
      <td>16,00,000</td>
      <td>operating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>32</td>
      <td>/organization/10seconds-software</td>
      <td>10seconds Software</td>
      <td>http://www.10secondsSoftware.com</td>
      <td>|Mobility|Enterprise Software|Software|</td>
      <td>Mobility</td>
      <td>1,00,000</td>
      <td>operating</td>
      <td>AUS</td>
      <td>NaN</td>
      <td>Sydney</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>33</td>
      <td>/organization/10six</td>
      <td>10Six</td>
      <td>http://10sixenergy.com</td>
      <td>|Electronics|Batteries|Energy|</td>
      <td>Electronics</td>
      <td>-</td>
      <td>operating</td>
      <td>USA</td>
      <td>NY</td>
      <td>New York City</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 39 columns</p>
</div>




```python
df.drop([0,24,51], axis=0).head()
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>/organization/tv-communications</td>
      <td>&amp;TV Communications</td>
      <td>http://enjoyandtv.com</td>
      <td>|Games|</td>
      <td>Games</td>
      <td>40,00,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>CA</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>/organization/rock-your-paper</td>
      <td>'Rock' Your Paper</td>
      <td>http://www.rockyourpaper.org</td>
      <td>|Publishing|Education|</td>
      <td>Publishing</td>
      <td>40,000</td>
      <td>operating</td>
      <td>EST</td>
      <td>NaN</td>
      <td>Tallinn</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>/organization/in-touch-network</td>
      <td>(In)Touch Network</td>
      <td>http://www.InTouchNetwork.com</td>
      <td>|Electronics|Guides|Coffee|Restaurants|Music|i...</td>
      <td>Electronics</td>
      <td>15,00,000</td>
      <td>operating</td>
      <td>GBR</td>
      <td>NaN</td>
      <td>London</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>/organization/r-ranch-and-mine</td>
      <td>-R- Ranch and Mine</td>
      <td>NaN</td>
      <td>|Tourism|Entertainment|Games|</td>
      <td>Tourism</td>
      <td>60,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>TX</td>
      <td>Dallas</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>/organization/club-domains</td>
      <td>.Club Domains</td>
      <td>http://nic.club/</td>
      <td>|Software|</td>
      <td>Software</td>
      <td>70,00,000</td>
      <td>NaN</td>
      <td>USA</td>
      <td>FL</td>
      <td>Ft. Lauderdale</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
df.drop(df.index[1:5], axis=0).head(10)
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>/organization/waywire</td>
      <td>#waywire</td>
      <td>http://www.waywire.com</td>
      <td>|Entertainment|Politics|Social Media|News|</td>
      <td>News</td>
      <td>17,50,000</td>
      <td>acquired</td>
      <td>USA</td>
      <td>NY</td>
      <td>New York City</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>/organization/club-domains</td>
      <td>.Club Domains</td>
      <td>http://nic.club/</td>
      <td>|Software|</td>
      <td>Software</td>
      <td>70,00,000</td>
      <td>NaN</td>
      <td>USA</td>
      <td>FL</td>
      <td>Ft. Lauderdale</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>/organization/fox-networks</td>
      <td>.Fox Networks</td>
      <td>http://www.dotfox.com</td>
      <td>|Advertising|</td>
      <td>Advertising</td>
      <td>49,12,393</td>
      <td>closed</td>
      <td>ARG</td>
      <td>NaN</td>
      <td>Buenos Aires</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>/organization/0-6-com</td>
      <td>0-6.com</td>
      <td>http://www.0-6.com</td>
      <td>|Curated Web|</td>
      <td>Curated Web</td>
      <td>20,00,000</td>
      <td>operating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>/organization/004-technologies</td>
      <td>004 Technologies</td>
      <td>http://004gmbh.de/en/004-interact</td>
      <td>|Software|</td>
      <td>Software</td>
      <td>-</td>
      <td>operating</td>
      <td>USA</td>
      <td>IL</td>
      <td>Springfield, Illinois</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>/organization/01games-technology</td>
      <td>01Games Technology</td>
      <td>http://www.01games.hk/</td>
      <td>|Games|</td>
      <td>Games</td>
      <td>41,250</td>
      <td>operating</td>
      <td>HKG</td>
      <td>NaN</td>
      <td>Hong Kong</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>10</td>
      <td>/organization/1-2-3-listo</td>
      <td>1,2,3 Listo</td>
      <td>http://www.123listo.com</td>
      <td>|E-Commerce|</td>
      <td>E-Commerce</td>
      <td>40,000</td>
      <td>operating</td>
      <td>CHL</td>
      <td>NaN</td>
      <td>Santiago</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>11</td>
      <td>/organization/1-4-all</td>
      <td>1-4 All</td>
      <td>NaN</td>
      <td>|Entertainment|Games|Software|</td>
      <td>Software</td>
      <td>-</td>
      <td>operating</td>
      <td>USA</td>
      <td>NC</td>
      <td>NC - Other</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>12</td>
      <td>/organization/1-800-dentist</td>
      <td>1-800-DENTIST</td>
      <td>http://www.1800dentist.com</td>
      <td>|Health and Wellness|</td>
      <td>Health and Wellness</td>
      <td>-</td>
      <td>operating</td>
      <td>USA</td>
      <td>CA</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>13</td>
      <td>/organization/1-800-doctors</td>
      <td>1-800-DOCTORS</td>
      <td>http://1800doctors.com</td>
      <td>|Health and Wellness|</td>
      <td>Health and Wellness</td>
      <td>17,50,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>NJ</td>
      <td>Newark</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 39 columns</p>
</div>




```python
df[100:].head() #df.tail(-100) )
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>/organization/20lines</td>
      <td>20lines</td>
      <td>http://20lines.com</td>
      <td>|Publishing|Education|Parenting|Curated Web|</td>
      <td>Curated Web</td>
      <td>12,36,454</td>
      <td>operating</td>
      <td>ITA</td>
      <td>NaN</td>
      <td>Roncade</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>101</td>
      <td>/organization/20x200</td>
      <td>20x200</td>
      <td>http://www.20x200.com</td>
      <td>|Social Commerce|Art|E-Commerce|</td>
      <td>Art</td>
      <td>28,00,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>NY</td>
      <td>New York City</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2800000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>102</td>
      <td>/organization/21cake-food-co</td>
      <td>21Cake Food Co.</td>
      <td>http://www.21cake.com</td>
      <td>|Manufacturing|Hospitality|</td>
      <td>Manufacturing</td>
      <td>14,64,128</td>
      <td>operating</td>
      <td>CHN</td>
      <td>NaN</td>
      <td>Beijing</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>103</td>
      <td>/organization/21diamonds-india</td>
      <td>21Diamonds</td>
      <td>http://www.21diamonds.de</td>
      <td>|E-Commerce|</td>
      <td>E-Commerce</td>
      <td>63,69,507</td>
      <td>operating</td>
      <td>IND</td>
      <td>NaN</td>
      <td>New Delhi</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6369507.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>104</td>
      <td>/organization/21e6</td>
      <td>2.10E+07</td>
      <td>NaN</td>
      <td>|Technology|Big Data|Mobile|</td>
      <td>Big Data</td>
      <td>50,50,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>CA</td>
      <td>SF Bay Area</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
df.country_code.values
```




    array(['USA', 'USA', 'EST', ..., nan, nan, nan], dtype=object)




```python
df['country_code'].values
```




    array(['USA', 'USA', 'EST', ..., nan, nan, nan], dtype=object)




```python
df.country_code.unique()
```




    array(['USA', 'EST', 'GBR', 'ARG', nan, 'HKG', 'CHL', 'DEU', 'FRA', 'CHN',
           'CAN', 'AUS', 'ROM', 'NLD', 'SWE', 'RUS', 'DNK', 'IND', 'SGP',
           'NOR', 'BEL', 'IRL', 'ITA', 'ISR', 'ESP', 'THA', 'NZL', 'CZE',
           'CHE', 'BRA', 'HUN', 'JPN', 'BWA', 'KOR', 'NGA', 'FIN', 'TUR',
           'CRI', 'PRT', 'TWN', 'KHM', 'COL', 'UKR', 'LTU', 'ZAF', 'AUT',
           'PHL', 'ISL', 'BGR', 'URY', 'HRV', 'KEN', 'MEX', 'JOR', 'VNM',
           'GHA', 'PER', 'POL', 'IDN', 'PAN', 'LVA', 'ALB', 'UGA', 'LBN',
           'GRC', 'ARE', 'PAK', 'EGY', 'SVK', 'LUX', 'MYS', 'BHS', 'ARM',
           'DZA', 'MDA', 'TUN', 'NIC', 'TZA', 'CYP', 'NPL', 'BHR', 'CMR',
           'SRB', 'SAU', 'CYM', 'BRN', 'SLV', 'ECU', 'MLT', 'SVN', 'LAO',
           'TTO', 'MAR', 'MMR', 'BGD', 'DOM', 'BMU', 'LIE', 'MOZ', 'GTM',
           'AZE', 'MCO', 'ZWE', 'UZB', 'OMN', 'BLR', 'JEY', 'JAM', 'KWT',
           'MUS', 'CIV', 'SOM', 'MKD', 'GIB', 'SYC', 'MAF'], dtype=object)




```python
df.country_code.value_counts()
```




    USA    28793
    GBR     2642
    CAN     1405
    CHN     1239
    DEU      968
           ...  
    MCO        1
    LIE        1
    ZWE        1
    SYC        1
    MOZ        1
    Name: country_code, Length: 115, dtype: int64




```python
df.agg(['count', 'size', 'nunique'])
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>49438</td>
      <td>49437</td>
      <td>45989</td>
      <td>45477</td>
      <td>45470</td>
      <td>49438</td>
      <td>48124</td>
      <td>44165</td>
      <td>30161</td>
      <td>44165</td>
      <td>...</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
    </tr>
    <tr>
      <td>size</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>...</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
    </tr>
    <tr>
      <td>nunique</td>
      <td>49436</td>
      <td>49350</td>
      <td>45850</td>
      <td>16675</td>
      <td>753</td>
      <td>14617</td>
      <td>3</td>
      <td>115</td>
      <td>61</td>
      <td>1089</td>
      <td>...</td>
      <td>20</td>
      <td>176</td>
      <td>2035</td>
      <td>1269</td>
      <td>740</td>
      <td>458</td>
      <td>225</td>
      <td>110</td>
      <td>32</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 39 columns</p>
</div>




```python
df.groupby('country_code').agg(['count', 'size', 'nunique']).stack()
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
      <th></th>
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>state_code</th>
      <th>region</th>
      <th>city</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
    <tr>
      <th>country_code</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" valign="top">ALB</td>
      <td>count</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>size</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>nunique</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td rowspan="2" valign="top">ARE</td>
      <td>count</td>
      <td>66</td>
      <td>66</td>
      <td>64</td>
      <td>59</td>
      <td>59</td>
      <td>66</td>
      <td>66</td>
      <td>0</td>
      <td>66</td>
      <td>66</td>
      <td>...</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
    </tr>
    <tr>
      <td>size</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>...</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td rowspan="2" valign="top">ZAF</td>
      <td>size</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>...</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
    </tr>
    <tr>
      <td>nunique</td>
      <td>52</td>
      <td>52</td>
      <td>51</td>
      <td>43</td>
      <td>36</td>
      <td>34</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>12</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td rowspan="3" valign="top">ZWE</td>
      <td>count</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>size</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>nunique</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>345 rows × 38 columns</p>
</div>




```python
df.groupby('country_code').agg(['count', 'size', 'nunique'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">permalink</th>
      <th colspan="3" halign="left">name</th>
      <th colspan="3" halign="left">homepage_url</th>
      <th>category_list</th>
      <th>...</th>
      <th>round_E</th>
      <th colspan="3" halign="left">round_F</th>
      <th colspan="3" halign="left">round_G</th>
      <th colspan="3" halign="left">round_H</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>size</th>
      <th>nunique</th>
      <th>count</th>
      <th>size</th>
      <th>nunique</th>
      <th>count</th>
      <th>size</th>
      <th>nunique</th>
      <th>count</th>
      <th>...</th>
      <th>nunique</th>
      <th>count</th>
      <th>size</th>
      <th>nunique</th>
      <th>count</th>
      <th>size</th>
      <th>nunique</th>
      <th>count</th>
      <th>size</th>
      <th>nunique</th>
    </tr>
    <tr>
      <th>country_code</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ALB</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>ARE</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>64</td>
      <td>66</td>
      <td>64</td>
      <td>59</td>
      <td>...</td>
      <td>1</td>
      <td>66</td>
      <td>66</td>
      <td>1</td>
      <td>66</td>
      <td>66</td>
      <td>1</td>
      <td>66</td>
      <td>66</td>
      <td>1</td>
    </tr>
    <tr>
      <td>ARG</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>149</td>
      <td>148</td>
      <td>149</td>
      <td>147</td>
      <td>136</td>
      <td>...</td>
      <td>1</td>
      <td>149</td>
      <td>149</td>
      <td>1</td>
      <td>149</td>
      <td>149</td>
      <td>1</td>
      <td>149</td>
      <td>149</td>
      <td>1</td>
    </tr>
    <tr>
      <td>ARM</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>AUS</td>
      <td>314</td>
      <td>314</td>
      <td>314</td>
      <td>314</td>
      <td>314</td>
      <td>314</td>
      <td>309</td>
      <td>314</td>
      <td>309</td>
      <td>290</td>
      <td>...</td>
      <td>1</td>
      <td>314</td>
      <td>314</td>
      <td>1</td>
      <td>314</td>
      <td>314</td>
      <td>1</td>
      <td>314</td>
      <td>314</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>USA</td>
      <td>28793</td>
      <td>28793</td>
      <td>28792</td>
      <td>28793</td>
      <td>28793</td>
      <td>28756</td>
      <td>26554</td>
      <td>28793</td>
      <td>26486</td>
      <td>27032</td>
      <td>...</td>
      <td>196</td>
      <td>28793</td>
      <td>28793</td>
      <td>86</td>
      <td>28793</td>
      <td>28793</td>
      <td>22</td>
      <td>28793</td>
      <td>28793</td>
      <td>2</td>
    </tr>
    <tr>
      <td>UZB</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>VNM</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>20</td>
      <td>...</td>
      <td>1</td>
      <td>21</td>
      <td>21</td>
      <td>1</td>
      <td>21</td>
      <td>21</td>
      <td>1</td>
      <td>21</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <td>ZAF</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>52</td>
      <td>51</td>
      <td>52</td>
      <td>51</td>
      <td>50</td>
      <td>...</td>
      <td>1</td>
      <td>52</td>
      <td>52</td>
      <td>1</td>
      <td>52</td>
      <td>52</td>
      <td>1</td>
      <td>52</td>
      <td>52</td>
      <td>1</td>
    </tr>
    <tr>
      <td>ZWE</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 114 columns</p>
</div>




```python
df_sample = df.sample(frac=0.05, random_state=1)
df_sample.head()
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>20044</td>
      <td>/organization/illumitex</td>
      <td>Illumitex</td>
      <td>http://www.illumitex.com</td>
      <td>|Architecture|Agriculture|UHB LEDs|Energy Effi...</td>
      <td>Architecture</td>
      <td>6,38,35,051</td>
      <td>operating</td>
      <td>USA</td>
      <td>TX</td>
      <td>Austin</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5250000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>15527</td>
      <td>/organization/flexiroam</td>
      <td>FLEXIROAM</td>
      <td>http://www.flexiroam.com</td>
      <td>|Mobile|</td>
      <td>Mobile</td>
      <td>3,82,000</td>
      <td>operating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>19221</td>
      <td>/organization/homestay-com</td>
      <td>Homestay.com</td>
      <td>http://www.homestay.com</td>
      <td>|Travel|</td>
      <td>Travel</td>
      <td>30,00,000</td>
      <td>operating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>7823</td>
      <td>/organization/cerephex</td>
      <td>Cerephex</td>
      <td>http://www.cerephex.com</td>
      <td>|Health Care|</td>
      <td>Health Care</td>
      <td>59,24,066</td>
      <td>operating</td>
      <td>USA</td>
      <td>CA</td>
      <td>SF Bay Area</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5924066.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>36527</td>
      <td>/organization/saddl</td>
      <td>Saddl</td>
      <td>http://www.saddl.nl</td>
      <td>|E-Commerce|Transportation|Curated Web|</td>
      <td>E-Commerce</td>
      <td>19,003</td>
      <td>operating</td>
      <td>NLD</td>
      <td>NaN</td>
      <td>Rotterdam</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
df.shape
```




    (54294, 39)




```python
df_sample.shape
```




    (2715, 39)




```python
df_dropped = df.dropna(subset=['round_A'])
df_dropped.head()
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>/organization/waywire</td>
      <td>#waywire</td>
      <td>http://www.waywire.com</td>
      <td>|Entertainment|Politics|Social Media|News|</td>
      <td>News</td>
      <td>17,50,000</td>
      <td>acquired</td>
      <td>USA</td>
      <td>NY</td>
      <td>New York City</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>/organization/tv-communications</td>
      <td>&amp;TV Communications</td>
      <td>http://enjoyandtv.com</td>
      <td>|Games|</td>
      <td>Games</td>
      <td>40,00,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>CA</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>/organization/rock-your-paper</td>
      <td>'Rock' Your Paper</td>
      <td>http://www.rockyourpaper.org</td>
      <td>|Publishing|Education|</td>
      <td>Publishing</td>
      <td>40,000</td>
      <td>operating</td>
      <td>EST</td>
      <td>NaN</td>
      <td>Tallinn</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>/organization/in-touch-network</td>
      <td>(In)Touch Network</td>
      <td>http://www.InTouchNetwork.com</td>
      <td>|Electronics|Guides|Coffee|Restaurants|Music|i...</td>
      <td>Electronics</td>
      <td>15,00,000</td>
      <td>operating</td>
      <td>GBR</td>
      <td>NaN</td>
      <td>London</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>/organization/r-ranch-and-mine</td>
      <td>-R- Ranch and Mine</td>
      <td>NaN</td>
      <td>|Tourism|Entertainment|Games|</td>
      <td>Tourism</td>
      <td>60,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>TX</td>
      <td>Dallas</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
df_dropped.shape
```




    (49438, 39)




```python
df_copy = df.copy()
df_copy.head()
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>/organization/waywire</td>
      <td>#waywire</td>
      <td>http://www.waywire.com</td>
      <td>|Entertainment|Politics|Social Media|News|</td>
      <td>News</td>
      <td>17,50,000</td>
      <td>acquired</td>
      <td>USA</td>
      <td>NY</td>
      <td>New York City</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>/organization/tv-communications</td>
      <td>&amp;TV Communications</td>
      <td>http://enjoyandtv.com</td>
      <td>|Games|</td>
      <td>Games</td>
      <td>40,00,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>CA</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>/organization/rock-your-paper</td>
      <td>'Rock' Your Paper</td>
      <td>http://www.rockyourpaper.org</td>
      <td>|Publishing|Education|</td>
      <td>Publishing</td>
      <td>40,000</td>
      <td>operating</td>
      <td>EST</td>
      <td>NaN</td>
      <td>Tallinn</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>/organization/in-touch-network</td>
      <td>(In)Touch Network</td>
      <td>http://www.InTouchNetwork.com</td>
      <td>|Electronics|Guides|Coffee|Restaurants|Music|i...</td>
      <td>Electronics</td>
      <td>15,00,000</td>
      <td>operating</td>
      <td>GBR</td>
      <td>NaN</td>
      <td>London</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>/organization/r-ranch-and-mine</td>
      <td>-R- Ranch and Mine</td>
      <td>NaN</td>
      <td>|Tourism|Entertainment|Games|</td>
      <td>Tourism</td>
      <td>60,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>TX</td>
      <td>Dallas</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
df_copy.shape
```




    (54294, 39)




```python
df_dropped['round_A'].dtype
```




    dtype('float64')




```python
df_dropped['round_A'].mean() 
```




    1243955.020874631




```python
df_copy['round_A'].fillna(value=np.round(df['round_A'].mean(),decimals=0), inplace=True)
```


```python
df_copy.agg(['count', 'size', 'nunique'])
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>49438</td>
      <td>49437</td>
      <td>45989</td>
      <td>45477</td>
      <td>45470</td>
      <td>49438</td>
      <td>48124</td>
      <td>44165</td>
      <td>30161</td>
      <td>44165</td>
      <td>...</td>
      <td>49438</td>
      <td>49438</td>
      <td>54294</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
      <td>49438</td>
    </tr>
    <tr>
      <td>size</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>...</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
      <td>54294</td>
    </tr>
    <tr>
      <td>nunique</td>
      <td>49436</td>
      <td>49350</td>
      <td>45850</td>
      <td>16675</td>
      <td>753</td>
      <td>14617</td>
      <td>3</td>
      <td>115</td>
      <td>61</td>
      <td>1089</td>
      <td>...</td>
      <td>20</td>
      <td>176</td>
      <td>2036</td>
      <td>1269</td>
      <td>740</td>
      <td>458</td>
      <td>225</td>
      <td>110</td>
      <td>32</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 39 columns</p>
</div>




```python
list(df['status'].unique())
```




    ['acquired', 'operating', nan, 'closed']




```python
list(df['status'].unique())
```




    ['acquired', 'operating', nan, 'closed']




```python
keys = list(df['status'].unique())
vals = range(1,8)
act = dict(zip(keys, vals))
act
```




    {'acquired': 1, 'operating': 2, nan: 3, 'closed': 4}




```python
df_copy['status_cat'] = df['status'].map(act)
df_copy.head()
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
      <th>status_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>/organization/waywire</td>
      <td>#waywire</td>
      <td>http://www.waywire.com</td>
      <td>|Entertainment|Politics|Social Media|News|</td>
      <td>News</td>
      <td>17,50,000</td>
      <td>acquired</td>
      <td>USA</td>
      <td>NY</td>
      <td>New York City</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>/organization/tv-communications</td>
      <td>&amp;TV Communications</td>
      <td>http://enjoyandtv.com</td>
      <td>|Games|</td>
      <td>Games</td>
      <td>40,00,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>CA</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>/organization/rock-your-paper</td>
      <td>'Rock' Your Paper</td>
      <td>http://www.rockyourpaper.org</td>
      <td>|Publishing|Education|</td>
      <td>Publishing</td>
      <td>40,000</td>
      <td>operating</td>
      <td>EST</td>
      <td>NaN</td>
      <td>Tallinn</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>/organization/in-touch-network</td>
      <td>(In)Touch Network</td>
      <td>http://www.InTouchNetwork.com</td>
      <td>|Electronics|Guides|Coffee|Restaurants|Music|i...</td>
      <td>Electronics</td>
      <td>15,00,000</td>
      <td>operating</td>
      <td>GBR</td>
      <td>NaN</td>
      <td>London</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>/organization/r-ranch-and-mine</td>
      <td>-R- Ranch and Mine</td>
      <td>NaN</td>
      <td>|Tourism|Entertainment|Games|</td>
      <td>Tourism</td>
      <td>60,000</td>
      <td>operating</td>
      <td>USA</td>
      <td>TX</td>
      <td>Dallas</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
list(df_copy['status_cat'].unique())
```




    [1, 2, 3, 4]




```python
df_copy['status_cat'].mean()
```




    2.1415257671197554




```python
df_sample = df.sample(frac=0.05, random_state=1)
df_sample.head()
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
      <th>permalink</th>
      <th>name</th>
      <th>homepage_url</th>
      <th>category_list</th>
      <th>market</th>
      <th>funding_total_usd</th>
      <th>status</th>
      <th>country_code</th>
      <th>state_code</th>
      <th>region</th>
      <th>...</th>
      <th>secondary_market</th>
      <th>product_crowdfunding</th>
      <th>round_A</th>
      <th>round_B</th>
      <th>round_C</th>
      <th>round_D</th>
      <th>round_E</th>
      <th>round_F</th>
      <th>round_G</th>
      <th>round_H</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>20044</td>
      <td>/organization/illumitex</td>
      <td>Illumitex</td>
      <td>http://www.illumitex.com</td>
      <td>|Architecture|Agriculture|UHB LEDs|Energy Effi...</td>
      <td>Architecture</td>
      <td>6,38,35,051</td>
      <td>operating</td>
      <td>USA</td>
      <td>TX</td>
      <td>Austin</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5250000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>15527</td>
      <td>/organization/flexiroam</td>
      <td>FLEXIROAM</td>
      <td>http://www.flexiroam.com</td>
      <td>|Mobile|</td>
      <td>Mobile</td>
      <td>3,82,000</td>
      <td>operating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>19221</td>
      <td>/organization/homestay-com</td>
      <td>Homestay.com</td>
      <td>http://www.homestay.com</td>
      <td>|Travel|</td>
      <td>Travel</td>
      <td>30,00,000</td>
      <td>operating</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>7823</td>
      <td>/organization/cerephex</td>
      <td>Cerephex</td>
      <td>http://www.cerephex.com</td>
      <td>|Health Care|</td>
      <td>Health Care</td>
      <td>59,24,066</td>
      <td>operating</td>
      <td>USA</td>
      <td>CA</td>
      <td>SF Bay Area</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5924066.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>36527</td>
      <td>/organization/saddl</td>
      <td>Saddl</td>
      <td>http://www.saddl.nl</td>
      <td>|E-Commerce|Transportation|Curated Web|</td>
      <td>E-Commerce</td>
      <td>19,003</td>
      <td>operating</td>
      <td>NLD</td>
      <td>NaN</td>
      <td>Rotterdam</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python

```
