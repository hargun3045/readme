# Code Snippets

> How to expand dimensions?

Use [np.expand_dims](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)

> Save numpy array

```
np.save('data.npy', num_arr) # save
new_num_arr = np.load('data.npy') # load
```
[stackoverflow answer](https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly)