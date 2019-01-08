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
The parameters are defined in function NN.init({}). As is discussed above, the layer_dims must be specified to define the neural network architecture. The defination of the other is optional as the default values have been specified already, but you can play with it to get better result. 
```js
var NN = new NN.init({
    init_method: "He",
    layer_dims: [train_X[0].length, 20, 5, 1],
    iterations: 100,
    learning_rate: 0.0007,
    lambda: 0.19,
    //keep_prob: 0.5,
    mini_batch_size: 32,
    opti_method: "adam", //"momentum"/"adam"
    beta1: 0.9,
    beta2: 0.999,
    print_cost: true,
    gradient_check: false
});
```

