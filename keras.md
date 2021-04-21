# Code snippets 

> What is an RNN Cell?

An ability to make custom rnn cells.

More [here](https://becominghuman.ai/understanding-tensorflow-source-code-rnn-cells-55464036fc07)

> How to plot intermediate layer activations?
```
def plot_activation(model, x_train):
  inp = model.input
  outputs = [layer.output for layer in model.layers]
  functors = [K.function([inp],[out]) for out in outputs]

  layer_outs = [func([x_train[0:1]]) for func in functors]
  last = layer_outs[0]

  f, axs = plt.subplots(1,2,figsize=(15,15))

  axs[0].imshow(x_train[0].squeeze(), cmap="bone")
  axs[0].set_title("Train data", fontsize=20)

  axs[1].imshow(last[0][0][:,:,-1].squeeze(), cmap="bone")
  axs[1].set_title("Activation", fontsize=20);
```

> Keras Sequential model

```
# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)
```

[Sequential API doc](https://www.tensorflow.org/guide/keras/sequential_model)


> How to get activations of the hidden layers? 


> How to get activations of hidden layers using functional API