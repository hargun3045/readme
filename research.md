# Research code chunks


# 

```python
# baseloss but with linear algebra - Pavlos implementation

def matrixloss(batch):
    batch = tf.convert_to_tensor(batch.reshape(-1,1),dtype='float32')
    batch = tf.Variable(batch)  
    with tf.GradientTape(persistent=True) as tape:
        dx = tape.gradient(bundle(batch),batch)
    d2x = tape.gradient(dx,batch)
    def custom_loss(x_true,x_pred):
        x = bundle(batch)
        loss = tf.reduce_mean((d2x+x)**2)
        return loss

    return custom_loss   
```


### Interpolation

```python
from scipy.interpolate import interp1d
t_interval = [0,0.5,1]
f_values = [1.5,2.2,3.3]
f = interpolate(t_interval,f_values)
```

### Boosting

```python
def boosted_sample(ts,loss,prev, num_points = 16,knob=0,memory=0,bins=10,t_min=0,t_max=2*np.pi):
    
    
    if loss is None:
        return torch.from_numpy(np.random.uniform(t_min,t_max,num_points)).type(torch.Tensor).view(-1,1), bins*[int((num_points*knob)/(bins))]
    else:
        
        
        ### interpolation part - Approximation of loss
        
        f = interp1d(ts.numpy().reshape(-1,), loss.detach().numpy().reshape(-1,))
        
        tinterp = np.linspace(ts.min().numpy(),ts.max().numpy(),bins)
        
        
        ### Find approximate number of points in each bin
    
        ## truncation #1
        samplesizes = [int(i*knob)  for i in num_points*((f(tinterp)/np.sum(f(tinterp))))]  
        
        samplesizes[samplesizes.index(max(samplesizes))] = max(samplesizes)+ int(num_points*knob) - sum(samplesizes)
        
        ## truncation #2 
        
        samplesizes = [int(i*(1-memory) + memory*j) for i,j in zip(samplesizes,prev)] 
        
        ### Add a masking point
        
        
        ### go to each bin, and sample the number of points which are supposed to be in each bin, and make a master t_tensor
        
        ### t_min = 0, t_max = 2pi 
        
        ### (0,pi/2), (pi/2,pi), (pi, 3pi/2), (3pi/2,2pi)
        
        intervals = [((i*(t_max - t_min)/bins) + t_min) for i in range(bins+1)]   (# bins + 1)
        
        t_sample = []
        for i,lowerlimit in enumerate(intervals[:-1]):
###            val = lower limit, intervals[i+1] = upper limit            

            t_sample.extend(np.random.uniform(lowerlimit,intervals[i+1],samplesizes[i]))
                        
        if len(t_sample) < num_points:
            t_sample.extend(np.random.uniform(t_min,t_max,num_points-len(t_sample)))               
        t_sample = np.array(t_sample)  
        t_tensor= torch.from_numpy(t_sample).view(-1,1).type(torch.Tensor)
        return t_tensor.view(-1,1),samplesizes
    
    
```

### Evaluation based estimate

```python
def eve(lossval=None,warmstart = 2,threshold=0):
## Starting value 
    
    if lossval == None:
        batch_size = warmstart
        
###        

### Threshold 

###        
    elif 2**(-int(np.log(lossval)) - threshold) <= 1:
        batch_size = warmstart
        
###        

    else:
        batch_size = warmstart*2**(-int(np.log(lossval)) - threshold)
    
    return batch_size
```

### Boosting implementation

```python
losses = []
sample_size = 128
bins = 8
t_tensor = np.random.uniform(0,2*np.pi, 128)
boostloss = None
num_epochs = 100
  
for epoch in range(num_epochs):
    # Function call

#             t_tensor,prev_sample  = boosted_sample(t_tensor,boostloss,prev_sample, num_points=sample_size,knob=1,memory=0.9)
        bs = eve(boostloss)
        batchlist = batches(t_tensor,batch_size=bs)
        for batch in batchlist:
            loss = loss_fn(mlp, batch)
            loss.backward()
            optimer.step()
            optimer.zero_grad()

            # Sanity addition - metric
        t_tensor = torch.from_numpy(np.random.uniform(0,2*np.pi,sample_size)).type(torch.Tensor).view(-1,1)
        losses.append(loss_fn(mlp,t_tensor))

            
            
######            

# Get the 'loss over t' part - loss at the points in the domain 
    t_tensor = np.linspace(t_min, t_max, 100)
    t_tensor.requires_grad_(True)
    x = repar(t_tensor,mlp) # reparametrization 
    d2x = nth_derivative(x,t_tensor,n=2)
    boostloss = (d2x+x)**2
    t_tensor.requires_grad_(False);
    
    
#######     

```