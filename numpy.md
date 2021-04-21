# Code Snippets

> How to get the sorted?
Very useful for plotting

Easy way 
```python
t = np.random.randint(0,100,10) # some predictor
x = np.random.randint(0,100,10) # some response
sorted_idx = np.argsort(t) #sorted index
t_plot = t[sorted_idx]
x_plot = t[sorted_idx]
```

Smartass way
```
t_plot, x_plot = list(zip(*(sorted(zip(*(t,x)))))
```

> How to expand dimensions?

Use [np.expand_dims](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)

> Save numpy array

```
np.save('data.npy', num_arr) # save
new_num_arr = np.load('data.npy') # load
```
[stackoverflow answer](https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly)