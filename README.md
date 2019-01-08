# Neural-Network_JS

This is a neural network (fully connected) library including the fundamantal hyper-parameter settings as well as optimizer (mini-batch, momentun, adam) settings. 

The activation chain: linear-> RELU -> linear-> RELU -> ....-> linear-> sigmoid

You can use it by coping the entire function to the source code, or create your own library by modifing it.

### Get started by training the first NN
You need to load the data sets fisrt, which is avaliable in . / src / Dataset.js
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
A 4-layer neural network is created, then we can train this model. 
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
The accuracy is calculated by sending all the train_X to the trained model, then compare with the correct Y lable train_Y. 

Probably your accuracy is not good enough, that is because several hyper-parameters are set as default, tuning these value and you may get better accuracy.

### Parameters setting
The parameters are defined in function NN.init({}). As is discussed above, the layer_dims must be specified to define the neural network architecture. The defination of the other is optional as the default values have been specified already, but you can play with it to get better result. The default values are used as input.
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

