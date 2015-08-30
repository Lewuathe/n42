n42 [![Build Status](https://travis-ci.org/Lewuathe/n42.png?branch=master)](https://travis-ci.org/Lewuathe/n42)
===

n42 is the deep learning module for nodejs. 

## How to install

    $ npm install n42

## Getting started 

```js
var n42 = require('n42');
    
// input data
// This is made of sylvester matrix
var input = $M([
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.2, 0.0],
    [1.0, 0.9, 0.1, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.8, 1.0],
    [0.0, 0.0, 1.0, 1.0]
]);

// label data
// This is made of sylvester matrix
var label = $M([
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0]
]);

var sda = new n42.SdA(input, label, 4, [3, 3], 2);

// Training all hidden layers
sda.pretrain(0.3, 0.01, 1000);

// Tuning output layer which is composed of logistics regression
sda.finetune(0.3, 50);

// Test data
var data = $M([
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0]
]);

console.log(sda.predict(data));

/**
 *   Predict answers
 *   [0.9999998973561728, 1.0264382721184357e-7] ~ [1.0, 0.0]
 *   [4.672230837774381e-28, 1]                  ~ [0.0, 1.0]  
 */
 
```

## Algorithms

| Class | Implemented algorithm |
|:-------|:----------|
| NN | Neural Network |
| LogisticsRegression | Logistics Regression |
| SdA | Stacked denoised Autoencoder |
| DBN | Deep Belief Nets |

## API Docs

[n42 API Doc](http://www.lewuathe.com/n42/apidocs/index.html)

## LICENSE

MIT License. Please see the LICENSE file for details.
