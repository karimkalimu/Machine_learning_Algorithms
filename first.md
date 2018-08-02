

```python
print("bash command")
```

    bash command



```python
%pwd
```


```python
%ls
```

%lsmagic


```python
%%timeit
suqre_event =[n*n for n in range (1000)]
```

    72 µs ± 1.76 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)



```python
import pandas as pd
import numpy as np

df=pd.DataFrame(np.random.randn(10,5))
df

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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.154256</td>
      <td>0.534792</td>
      <td>-0.969722</td>
      <td>0.281576</td>
      <td>-0.649383</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.989162</td>
      <td>-2.321711</td>
      <td>-0.642028</td>
      <td>-2.144652</td>
      <td>-0.286437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.448702</td>
      <td>-0.431879</td>
      <td>0.646302</td>
      <td>0.440967</td>
      <td>-1.691870</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.202197</td>
      <td>-1.022979</td>
      <td>0.111717</td>
      <td>-1.126324</td>
      <td>0.317175</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.267406</td>
      <td>1.332793</td>
      <td>0.409042</td>
      <td>-0.154089</td>
      <td>0.300777</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.453283</td>
      <td>1.027386</td>
      <td>-0.311557</td>
      <td>0.546110</td>
      <td>-0.875082</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1.427426</td>
      <td>0.960010</td>
      <td>-0.555407</td>
      <td>-0.903960</td>
      <td>0.684255</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.250086</td>
      <td>1.029602</td>
      <td>0.504389</td>
      <td>-1.345649</td>
      <td>1.063940</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.537183</td>
      <td>0.622638</td>
      <td>0.078305</td>
      <td>-1.087057</td>
      <td>-0.122692</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-2.722136</td>
      <td>-0.949335</td>
      <td>-0.679788</td>
      <td>2.274616</td>
      <td>0.951601</td>
    </tr>
  </tbody>
</table>
</div>


