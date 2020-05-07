# Pandas helpful code

>How to make a mask?

```
mask = np.ones(size, dtype = 'int')
```

>Get a sampled collection

```
df.sample(size, replace = False)
```

>Shuffle data

```
df.sample(frac=1)
```

[Stackoverflow - How to shuffle a dataframe?](https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows)


>Axis to left side

```
from matplotlib import pyplot as plt

f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.tick_right()
plt.plot([2,3,4,5])
plt.show()
```


>How to customize grids in subplots

```
fig3 = plt.figure(constrained_layout=True)
gs = fig3.add_gridspec(3, 3)
f3_ax1 = fig3.add_subplot(gs[0, :])
f3_ax1.set_title('gs[0, :]')
```

[Creating custom gridspaces](https://matplotlib.org/3.2.1/tutorials/intermediate/gridspec.html)

>Make a line chart with bar chart

```
fig = plt.figure()
ax = ts.plot(kind="bar")   # barchart
ax2 = ax.twinx()
ax2.plot(ax.get_xticks(), df.rolling(10).mean()) #linechart
```
[Line chart with bar](https://stackoverflow.com/questions/33239937/python-bar-graph-and-line-graph-in-same-chart-with-pandas-matplotlib)
>Plot bar chart with ax object

```
ax.bar(x,y,*params)
```


>Fix bar chart x labels

```
#set ticks every week
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
#set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
```

[Link to italian blog with solution](https://scentellegher.github.io/programming/2017/05/24/pandas-bar-plot-with-formatted-dates.html)

[Link to blog recommended by italian](https://pbpython.com/effective-matplotlib.html)

>How to combine year, month, date in a single column pandas

```
df['Date']=pd.to_datetime(df.year*10000+df.month*100+df.day,format='%Y%m%d')
```

[Combine year month date pandas](https://stackoverflow.com/questions/48155787/how-to-combine-year-month-and-day-columns-to-single-datetime-column)

>How to nicely print a dictionary using json?

```
import json
print(json.dumps(dictionary, indent=4, sort_keys=True))
```

> How to plot a time series

[Number of births per month](https://jakevdp.github.io/PythonDataScienceHandbook/04.09-text-and-annotation.html)

>Choose a dataframe with non-empty column

```
df = df[df['EPS'].notna()]
```
OR

```
df = df[pd.notnull(df['EPS'])]
```
[Remove null values from a column](https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-a-certain-column-is-nan)

>Make a pandas series as datetime

```
df['date'] = df['date'].astype('datetime64[ns]')
```

>Add a name to the index

```
df.index.name = 'date'
```
[How to add an index name](https://stackoverflow.com/questions/18022845/pandas-index-column-title-or-name)

>Mapping a dictionary value to a pandas series

```
df['continent'] = df.country.map(mapping)
```

[How to map dictionary values to a pandas series](https://stackoverflow.com/questions/20250771/remap-values-in-pandas-column-with-a-dict)

> How to map or apply values to entire dataframe series?

```
def square(x):
    return x ** 2
s.apply(square)
```
[Click here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.apply.html#pandas.Series.apply) for link to documentation

> How to add gaussian noise to a series

```
mu, sigma = 0, 0.1 
# creating a noise with the same dimension as the dataset (2,2) 
noise = np.random.normal(mu, sigma, [2,2]) 
print(noise)
```

[link to](https://stackoverflow.com/questions/46093073/adding-gaussian-noise-to-a-dataset-of-floating-points-and-save-it-python?rq=1) stackoverflow answer

>Change position of a column 
in a pandas dataframe

Change position of the list
df = df[[df.columns[-1]] + list(df.columns[:-1])]

Pandas export to csv

```
df.to_csv(filename, index = False)
```
>Drop a column

[How to drop a column](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html	
)

[Pandas API Quick Reference](https://pandas.pydata.org/pandas-docs/version/0.22.0/api.html)

Quick code snippets

```
df[['col1','col2']].values
```
The above gives a numpy ndarray

**Rename column**
```
df.rename(columns = {'old_col':'new_col'}, inplace = True)

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