Go to code chunks here

```
conda create -n name (packages)

pip install -r requirements.txt
```

>Generators

In addition to automatic method creation and saving program state, when generators terminate, they automatically raise StopIteration. In combination, these features make it easy to create iterators with no more effort than writing a regular function.
```python
def reverse(data):
    for index in range(len(data)-1,-1,-1):
        yield data[index]

for char in reverse('golf'):
    print(char)
```

### Class Master

Fully functional vector class.

Reference to documentation [Classes complet](https://docs.python.org/3/tutorial/classes.html#scopes-and-namespaces-example)

```python
class Vector:
    
    """
    My first docstring: Wohoo
    
    """
    
    def __init__(self, harlist):
        self.storage = harlist
        
    
    def __len__(self):
        return len(self.storage)+1
    
    def __getitem__(self,i):
        return self.storage[i]
        

    def __add__(self, vector2):
        
        sumlist = []
        
        for i,_ in enumerate(vector2):
            sumlist.append(self.storage[i]+vector2[i])
        return Vector(sumlist)
        
    def __radd__(self, vector2):
        return self + vector2
    
    
    def __mul__(self, scalar):
        return Vector([i*scalar for i in self.storage])
    
    def __rmul__(self,scalar):
        return self*scalar
    
    def dotproduct(self, vector2):
        
        return sum(i*j for i,j in zip(self,vector2))
    
    def __repr__(self):
        return f'{self.storage}'

```


### Download youtube videos

```python
from pytube import YouTube

YouTube('https://www.youtube.com/watch?v=5JnMutdy6Fw').streams.get_highest_resolution().download(path)
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