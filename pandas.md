>Pandas helpful code

[Pandas API Quick Reference](https://pandas.pydata.org/pandas-docs/version/0.22.0/api.html)

Quick code snippets

```
df[['col1','col2']].values
```
The above gives a numpy ndarray

**Rename column**
```
df.rename({columns = 'old_col':'new_col', inplace = True})

```

>Change column positions

```
cols = df.cols

new_cols = ['a','b','c','d']

df = df[new_cols]

```

**initial setup of dataframe**
```
df = pd.read_csv('sales1.csv',header = 0, index_col = False, names = ['title', 'sold','price','royalty'])

```

>The idea of a dataframe is having several named lists in parallel


Check if a series is not empty
```python
df[df.title.notnull()]
```
```
Sort values by a column
df.sort_values(by = ["columnname"])
```
>Set Index

Helps with speeding up the search

```
df.set_index('month')
```

>Groupby

```
df.groupby(["a"]).size()
```
size() considers NaN values
count() does not [overflow answer](https://stackoverflow.com/questions/33346591/what-is-the-difference-between-size-and-count-in-pandas)

Essentially, what groupby does is it seperates the column into separate groups

>Unstack

Unstack takes you from a complicated series to a dataframe.

```
s.unstack().fillna()
```

The main idea is that groupby gives you ways to make a dataframe into a series and unstack gives you a dataframe from a series.
This wrangling can come handy to make some comparisons.

>Datetime module
Similar to string
```
s.dt.month
```

>Merge

```
#Combine to dataframes on some reference
pd1.merge(pd2, on = ["col1"])
```

### Plotting

Easiest with a series, so do some groupbys and then plot.

>Exercise 6 handy code


First we get the currency as a column
```
s['currency'] = s.title.str.extract(r'\((.*)\)').fillna(method = "bfill")
```

Simple way to get rid of waste entries
```
s.dropna(inplace = True)
```

Get two dataframes one below the other
```
s = pd.concat([s,s3], axis = 0)
```

Assign function for new column

```
s = s.assign(total=s.price * s.sold)
```

This groupby gives a datframe, because we are specifying a list of columns

```
s.groupby(["title","currency"])[["total"]].sum()
```
The below code would be a series
```
s.groupby(["title","currency"])[["total"]].sum()
```