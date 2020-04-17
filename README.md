Go to code chunks here

```
conda create -n name (packages)

pip install -r requirements.txt
```

#### 

Video link to setup conda environment kernel for a notebook

[ipykernel](https://www.youtube.com/watch?v=6kXLUvsnhuI)

Or official anaconda documentation says you must do a source fastai2 activate and it will show you the option



```
$ git config --global user.name
```

```
$ git config --global user.name "Mona Lisa"
```

>Add Remote directory
```
git remote add origin 'repo-url'

```

> Convert iPython notebook to python

Assuming you have already pip installed ipynb-py-convert

```
ipynb-py-convert examples/plot.ipynb examples/plot.py


```

## Pickling

```
with open ("filename", "wb") as f:
    pickle.dump(listOrdictionary, f)    
```

To open it elsewhere:

```
with open("filename","rb") as f:
    listOrDictionary = pickle.load(f)
```

## Shelving

Difference between pickling and shelving is that pickling is like anywhere available variable, while shelving is an anywhere available dictionary

```
import shelve
with shelve.open("filename") as f:
    f["key"] = listOrDictionary
```

Now to access it, get the file in the right directory

```
with shelve.open("filename") as f:
    listOrDictionary = f["key"]
```


## JSON

JSON files are available on the net, these map to dictionaries
```
import json
with open("data/us-states.json") as fd: 
    data = json.load(fd)
```