# Code Snippets

### Transparent plotting

```python
fig.savefig("images/SGD.png",bbox_inches='tight',transparent=True, pad_inches=0)
```

### Research - Function plot + Trace plot

Main functions:

1. `ax.plot()`
2. `ax.set_xlabel()`
3. `ax.legend()`
4. `ax.text()`

```python
fig, ax = plt.subplots(1,2,figsize=(14,6))
ax[0].plot(t_space, x,'-',linewidth=2,alpha=0.7, label = 'True function value', markersize = 10, color = '#9FC131FF')
ax[0].plot(t_space, x_pred0,'-',linewidth=2, alpha=0.7, label='$\eta=0$ predictions',markersize=10,color='darkblue')
ax[0].plot(t_space, x_pred1,'-',linewidth=2, alpha=0.7, label='$\eta=1$ predictions',markersize=10,color='#FF2F92')
ax[0].set_xlabel('$t$',fontsize=24)
ax[0].set_ylabel('$x$',fontsize=24)
ax[0].legend();
ax[0].set_title('$\dot{x} + x = 0$',fontsize=36);
ax[0].text(4, 0.6, f'Number of points = {nt}', bbox=dict(facecolor='blue', alpha=0.2),fontsize=20)


ax[1].plot(np.log10(loss_list0),label='$\eta=0$ loss',markersize=10,color='darkblue',alpha=0.8)
ax[1].plot(np.log10(loss_list1),label='$\eta=1$ loss',markersize=10,color='#FF2F92',alpha=0.8)
ax[1].set_title('Trace plot',fontsize=24);
ax[1].set_xlabel('$epochs$',fontsize=18)
ax[1].set_ylabel('$Log loss$',fontsize=18)
ax[1].legend();
plt.tight_layout()
```

> How to show the x and y axis of a plot?

Full link [here](https://stackoverflow.com/questions/25689238/show-origin-axis-x-y-in-matplotlib-plot)

X,Y axis
```
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
```

> What are the type of subplot options available?

```python
# First create some toy data:
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Create just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(2, 2, subplot_kw=dict(polar=True))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

# Share a X axis with each column of subplots
plt.subplots(2, 2, sharex='col')

# Share a Y axis with each row of subplots
plt.subplots(2, 2, sharey='row')

# Share both X and Y axes with all subplots
plt.subplots(2, 2, sharex='all', sharey='all')

# Note that this is the same as
plt.subplots(2, 2, sharex=True, sharey=True)

# Create figure number 10 with a single subplot
# and clears it if it already exists.
fig, ax = plt.subplots(num=10, clear=True)
Copy to clipboard
Examples us
```

> How to move the `plt.legend()` box around?
`loc='upper right', bbox_to_anchor=(0.5, 0.5)`
full details [here](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.legend.html)

> How to increase distance between title and plot?

[`ax.set_position`](https://stackoverflow.com/questions/16419670/increase-distance-between-title-and-plot-in-matplolib)

> Custom annotation in seaborn heatmap

```
fig, ax = plt.subplots()
ax = sns.heatmap(data, annot = labels, fmt = '')
```

[Stack overflow answer](https://stackoverflow.com/questions/33158075/custom-annotation-seaborn-heatmap)

[Seaborn heatmap doc](https://seaborn.pydata.org/generated/seaborn.heatmap.html)

> Display matrix values along with colormap

```
fig, ax = plt.subplots()

min_val, max_val = 0, 15

intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))

ax.matshow(intersection_matrix, cmap=plt.cm.Blues)

for i in xrange(15):
    for j in xrange(15):
        c = intersection_matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
```
[stackoverflow answer](https://stackoverflow.com/questions/40887753/display-matrix-values-and-colormap)

Also, for custom color map, use this code:

```
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['k', 'w', 'r'])
cax = ax.matshow(x,cmap=cmap)
```

> Plot within a plot

Use inset locator.
Full matplotlib demo code [here](https://matplotlib.org/gallery/axes_grid1/inset_locator_demo.html)

> Creating annotated heatmaps (NOT SEABORN)

[Matplotlib docs](https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html)