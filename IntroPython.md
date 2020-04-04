# Basic python Code snippets

Python debugger
```
from IPython.core.debugger import set_trace
def simulate_election(model, n_sim):
    set_trace() #Important
    simulations = np.random.uniform(size=(51, n_sim))
    obama_votes = (simulations < model.Obama.values.reshape(-1, 1)) * model.Votes.values.reshape(-1, 1)
    #summing over rows gives the total electoral votes for each simulation
    return obama_votes.sum(axis=0)
```

## Debugging tips

step - s 
next - n
continue - c
up/down - u/c
list - l


```
for i, elem in enumerate(float_list):
    print(i,elem)
```
Enumerate uses default indexing

```
for i, f in zip(int_list, float_list):
    print(i, f)
```

zip uses indexing as specified

>Conditional List comprehension

```
#incase of only an if condition
comp_list1 = [2*i for i in squaredlist if i % 3 == 0]

# But for if else
comp_list2 = [2*1 if i % 3 == 0 else 0 for i in squaredlist]
```


## Numpy

```
np.mean(list)

np.std(list)

```

Basic Probability

**Something must happen**

$$P(\Omega) =1$$

**Complementary events must have probabilities summing to 1**

Either E happened or didnt. So,

$$P(E) + P(\sim E) = 1$$

**The Multiply/And/Intersection Formula for independent events**: If E and F are independent events, the probability of both events happening together $P(EF)$ or $P(E \cap F)$ (read as E and F or E intersection F, respectively) is the multiplication of the individual probabilities.

$$ P(EF) = P(E) P(F) .$$


**The Plus/Or/Union Formula** 

We can now ask the question, what is $P(E+F)$, the odds of E alone, F alone, or both together. Translated into English, we are asking, whats the probability that only the first toss was heads, or only the second toss was heads, or that both came up heads?  Or in other words, what are the odds of at least one heads? The answer to this question is given by the rule:

$$P(E+F) = P(E) + P(F) - P(EF),$$ 



```
#np.random.random

2d = np.random.randint(12, size = (3,4))
2d.max(axis=1)

```
Axis = 1 means each consideration will be across the column index
Axis = 0 means each consideration will be across the row index

Numpy note on dimensions
https://stackoverflow.com/questions/15680593/numpy-1d-array-with-various-shape
<!-- 

```

```

```

```



```

```


```

```


```

```


```

```


```

```


```

```

```

```


```

```


```

```


```

```

```

```


```

```


```

```

```

```


```

```

```

```


```

```


```

```

```

```


```

```


```

```


```

``` -->