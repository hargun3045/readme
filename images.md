### Working with images

> How to get image from array

```
# z is the array 
z = (z * 255).astype(np.uint8)
img = Image.fromarray(z)
```

> How to 