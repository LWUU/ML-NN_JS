# Neural-Network_JS

This is a neural network (fully connected) library including the fundamental hyper-parameter settings as well as optimizer (mini-batch, momentum, Adam) settings. 

The activation chain: linear-> RELU -> linear-> RELU -> ....-> linear-> sigmoid

The matrix library is in https://github.com/LWUU/ML-Matrix_JS, feel free to have a look on it.

You can use it by copying the entire function to the source code or create your own library by modifying it.

As this is the first neural network library I wrote, although I have tested all the functionalities,  the bugs can still exist. Also, I expected there is a lot of room to be improved for the algorithm. In case you (lucky enough to) find any of the bugs or have any feedback positive or negative. I am really looking forward to knowing it, and will try my best to make this work better!

### Get started by training the first NN
You need to load the data sets fisrt, which are avaliable in . / src / Dataset.js
```js
//Load the training sets
var train_X = train_X();
var train_Y = train_Y();
```
Then initialize the NN by defining the NN architecture.
```js
//Transfer the properties to a defined matrix
var NN = new NN.init({
    layer_dims: [train_X[0].length, 20, 5, 1]
});
```
A 4-layer neural network is created! Now we can train this model :) 
```js
//Train the neural network with the given data sets
NN.train(train_X, train_Y);
//You are expected to get the following result:
  //Mini-batch size is set as default (the number of the training sets).
  //Optimization method is set as default (gradient descent).
  //Initialization method is set as default (random inintialization).
  //=====Training Started=====
  //=====Training Finished=====
  //The accuracy is + "your result"
```
The accuracy is calculated by sending all the train_X to the trained model, then compare with the correct Y label train_Y. 

Probably your accuracy is not good enough, that is because several hyper-parameters are set as default, tuning these value and you may get better accuracy.

### Parameters setting
The parameters are defined in function NN.init({}). As is discussed above, the layer_dims must be specified to define the neural network architecture. The definition of the other is optional as the default values have been specified already, but you can play with it to get a better result. The default values are used as input.
```js
var NN = new NN.init({
    layer_dims: [train_X[0].length, 20, 5, 1],
    init_method: null, //"He", 
    iterations: 1000,
    learning_rate: 0.001,
    lambda: 0, //for L2 regularization
    keep_prob: 1, // for drop-out
    mini_batch_size: train_X[1].length,
    opti_method: null, //"momentum"/"adam"
    beta1: 0.9, //for momentum optimizer
    beta2: 0.999, //for adam optimizer (together with beta1)
    print_cost: false,
    gradient_check: false
});
```

