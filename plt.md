## Plot snips

Colors:

- darkblue
- #9FC131FF (greenish)
- black
- #FF2F92 (pinkish)

### Homework 1

1. Header

```
# Run this cell for more readable visuals 
large = 22; med = 16; small = 10
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'axes.linewidth': 2,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.style.use('seaborn-whitegrid')
plt.rcParams.update(params)
#sns.set_style("white")
%matplotlib inline
```

2. EDA plots
```
fig.subplots_adjust(hspace =0.5, wspace=0.2)
```

3. Knn

```
fig, axs = plt.subplots(4,2, figsize=(20, 20), facecolor='w', edgecolor='k')
```

```
# To run a for loop
axs = axs.ravel()
```

```
ax.plot(xtrain, ytrain, '.', alpha=0.7, label='Train',markersize=10,color='#9FC131FF')
ax.plot(xtest, ytest, '.', alpha=0.7, label='Test',markersize=10,color='darkblue')
axs.plot(Xline, knn_model.predict(Xline), label='Predicted',color='black', linewidth=2)
```

4. MSE plot

```
ax.plot(K, mse_test, 's-', label='Test',color='darkblue',linewidth=2)
ax.plot(K, mse_train, 's--', label='Train',color='#9FC131FF',linewidth=2)
ax.set_xlabel(r'$K$ values', fontsize=15)
ax.set_ylabel('$MSE$', fontsize=15)
ax.set_title(r'$MSE$')
```

5. Linear regression

Scatter plot
```
plt.scatter(X_train,y_train,color='#FF2F92',label='Train data')
```

6. Residuals 

```
plt.axhline(color='black',linestyle='dashed')
```

7. Barplots

edgecolor='k'
```
ax.hist(df[df.gender==i].income, log=True, label = f'Gender type = {gendermap[i]}',alpha=0.4,color = colors[i], bins = 10,density=True,edgecolor='k' )
```