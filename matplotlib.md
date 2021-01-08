# Code Snippets

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