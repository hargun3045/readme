## AI3

### Lab 1

1. `df.series.apply(lambda x: ...)`
2. `re.sub('[^A-Za-z0-9 ]+','',harstring)`
3. `re.compile('<[^>]+>').sub('',harstring)`

Smart dictionary
`from collections import defaultdict`

`hardict = defaultdict(lambda: 0)`

Sort a dictionary

`dict(sorted(hardict.items(),key=lambda x: -x[1]))`

In the above: `lambda x: -x[1]` does two things. `-x[1]` highest to lowest, and values


# AI2

### Lab 7

Link is [here](https://colab.research.google.com/drive/1a0yqSBz1Jbaa76JAyLoogjS3WGBoQJMI?usp=sharing)

### HW 2 solutions

Link is [here](https://colab.research.google.com/drive/1H6C3-ENR-OTPaoradtljMpMRxQ3Ujn4Q?authuser=1)