Go to code chunks here

## How to delete multiple github repos?

Go to ```https://repo-sweeper.herokuapp.com/```

Details in this post: [Delete multiple repos](https://medium.com/@taylorjayoung/the-easiest-way-to-delete-multiple-github-repositories-at-once-e71e16734b59)

### Jupyter lab

> How to use keyboard shortcuts jupyter lab?
```
1. Open Advanced Settings Editor under the Settings tab, or command , in Mac.
2. Navigate to Keyboard Shortcuts. You should see the screen plalanne answered with.
3. Re-open your notebook and test if it works as intended.
You can customize more keys in this fashion as long as it is defined here on GitHub. For the most part, all that you need are the command IDs starting line 72.
```


What is np.argsort?

Gives an array of sorted positions

https://stackoverflow.com/questions/58265156/output-of-hidden-layer-for-every-epoch-and-storing-that-in-a-list-in-keras

> Pandoc

How to convert .doc to .md

```
pandoc -o name.md --extract-media=name/ name.docx -w gfm --atx-headers --columns 9999

```

### Questions solved during POKER dataset

> How to add conda environment to jupyter notebook?

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=firstEnv
```

> How to go back from ONE HOT encoding?

use inverse_transform

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

s = pd.Series(['a', 'b', 'c'])
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
s = le.fit_transform(s)
s = ohe.fit_transform(s.reshape(-1,1))
print(s)
```

> How to change learning rate in keras?

```
from keras.callbacks import LearningRateScheduler

# This is a sample of a scheduler I used in the past
def lr_scheduler(epoch, lr):
    decay_rate = 0.85
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr
```

```
callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]

model = build_model(pretrained_model=ka.InceptionV3, input_shape=(224, 224, 3))
history = model.fit(train, callbacks=callbacks, epochs=EPOCHS, verbose=1)
```

[How to change learning rate in keras? ](https://stackoverflow.com/questions/59737875/keras-change-learning-rate)

> How to smoothen labels in keras?

Define a custom loss function
```
tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO,
    name='categorical_crossentropy'
)
```

[Label smoothening detailed tutorial](https://www.pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/)

> How to unzip a file using command line?

```
unzip file.zip
```

> How to make class predictions using keras model?

```
ynew = model.predict_classes(Xnew)
```

[detailed post on predictions](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)

> Specify type of optimiser in keras?

```
tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam', **kwargs
)
```


> How to add dropout in keras?
```
model.add(Dropout(0.2, input_shape=(60,)))
```
[Detailed post](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

> How to shade area in matplotlib?

```
ax.axhspan(y1, y2, facecolor=c, alpha=0.5)
```
OR 

```
ax.hline(y1, color=c)
ax.hline(y2, color=c)
ax.fill_between(ax.get_xlim(), y1, y2, color=c, alpha=0.5)

```


>How to get default view after calling seaborn?

```
plt.style.available # Plots available
plt.style.use('classic')
```

[avoid seaborn influencing matplotlib plots](https://stackoverflow.com/questions/54885636/avoid-seaborn-influencing-matplotlib-plots)

> How to find value counts in numpy array?

```
x = np.array([1,1,1,2,2,2,5,25,1,1])
unique, counts = np.unique(x, return_counts=True)

print np.asarray((unique, counts)).T

```

>How to implement k nearest neighbors?

```
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
```
```python
>>> print(neigh.predict([[1.1]]))
[0]
>>> print(neigh.predict_proba([[0.9]]))
[[0.66666667 0.33333333]]
```

>How to make sns.heatmap show numbers in non-scientific notation?

```
sns.heatmap(table2,annot=True,cmap='Blues', fmt='g')
```
[seaborn showing scientific notation](https://stackoverflow.com/questions/29647749/seaborn-showing-scientific-notation-in-heatmap-for-3-digit-numbers)

>Implementing the confusion matrix from sklearn?

```
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])
```

>How to take a random selection from some points?

```
sample_indices = np.random.choice(range(100), size = 20)
```


>How to round a float value in an f-string

```
x = 3.14159265
print(f'pi = {x:.2f}')
```


>Netlify app to see site settings

[Site setting](https://app.netlify.com/sites/gensectimes/settings/general)

>Netlify CMS integration

[Complete tutorial](https://www.netlifycms.org/docs/jekyll/)


>How to make sure git always asks for username password?

```
git config --local credential.helper ""
```

Reverse operation 

```
git config --global credential.helper store
```

[Ask for username](https://stackoverflow.com/questions/13103083/how-do-i-push-to-github-under-a-different-username)

## How to clear command line terminal

```
control + U
```

[other handy tips](https://stackoverflow.com/questions/9679776/how-do-i-clear-delete-the-current-line-in-terminal)

## What is make?

[Tutorial](https://opensource.com/article/18/8/what-how-makefile)

```
target: prereq
    recipe
```

>What is softmax?

[Medium article on softmax](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d )

>Use cases of numpy and pytorch tensor

>Understanding BCE Loss

[Medium post on BCE Loss](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

[What is the difference between pytorch & numpy](https://rickwierenga.com/blog/machine%20learning/numpy-vs-pytorch-linalg.html)

>How to convert list to string

```
# using list comprehension 
listToStr = ' '.join([str(elem) for elem in s]) 
```

>Rounding floats with f-strings

```
x = 3.14159265
print(f'pi = {x:.2f}')
```


>Convert 2 lists into a dictionary

[How to convert lists to dictionary](https://stackoverflow.com/questions/209840/convert-two-lists-into-a-dictionary)


> How to shuffle a list?

Use python list functions, insert() and pop()

[insert & pop](https://www.geeksforgeeks.org/python-shift-last-element-to-first-position-in-list/)

Also, to get a list of 1 element

```
harlist[0]--> Element
harlist[:1]--> List consisting single element

```

>All, any

[How do Python's any and all functions work?](https://stackoverflow.com/questions/19389490/how-do-pythons-any-and-all-functions-work)


>Website building

1. How to set up Jekyll

[Jekyll basic](https://jekyllrb.com/docs/)

2. How to set up mmistakes

[mmistakes basic](https://jekyllrb.com/docs/)



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

## Docker 

[Docker first build and examples](https://hub.docker.com/?overlay=onboarding&step=download)


## Github



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