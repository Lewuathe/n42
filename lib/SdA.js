/**
 *  Stacked denoised auto encoder
 *
 *  @module n42
 *  @class  SdA
 *  @author Kai Sasaki
 *  @since  2014/01/27
 *
 */

var Mattix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var utils  = require('./utils');

var HiddenLayer = require('./HiddenLayer.js');
var dA          = require('./dA.js');
var LogisticsRegression = require('./LogisticsRegression.js');

/**
 *   Stacked autoencoder class
 *   
 *   @class SdA
 *   @constructor
 *
 */

function SdA(input, label, nIns, hiddenLayerSizes, nOuts) {
    var self = this;
    
    /**
     *  input data. This type is defined in sylvester library
     *
     *  @property x
     *  @type     Matrix
     */
    self.x = input;

    /**
     *  label data. This type is defined in sylvester library
     *
     *  @property y
     *  @type     Matrix
     */
    self.y = label;

    /**
     *  hidden layers which activations are sigmoid function
     *
     *  @property sigmoidLayers
     *  @type     Array
     */
    self.sigmoidLayers = [];
    // Denoised autoencoder layer is used for only training
    // These layer shares own weight parameter with sigmoidLayers
    self.dALayers      = [];
    
    /**
     *  number of hidden layers
     *
     *  @property nLayers
     *  @type     int
     */
    self.nLayers       = hiddenLayerSizes.length;

    // Construct each layers
    for (var layerIndex = 0; layerIndex < self.nLayers; layerIndex++) {

        // Select input size of this layer
        var inputSize;
        if (layerIndex == 0) {
            inputSize = nIns;
        } else {
            inputSize = hiddenLayerSizes[layerIndex-1];
        }

        var layerInput;
        // Select input object of this layer
        if (layerIndex == 0) {
            layerInput = self.x;
        } else {
            layerInput = self.sigmoidLayers[layerIndex-1].sampleHGivenV();
        }
        // Construct sigmoid layer which is used for prediction
        var sigmoidLayer = new HiddenLayer(layerInput,
                                           inputSize,
                                           hiddenLayerSizes[layerIndex],
                                           utils.sigmoid);
        self.sigmoidLayers.push(sigmoidLayer);
        // Construct denoised autoencoder layer which is used for training
        var dALayer = new dA(layerInput,
                             inputSize,
                             hiddenLayerSizes[layerIndex],
                             sigmoidLayer.W,
                             sigmoidLayer.b);
        self.dALayers.push(dALayer);
    }

    var lastIndex = self.sigmoidLayers.length-1;
    self.logLayer = new LogisticsRegression(self.sigmoidLayers[lastIndex].sampleHGivenV,
                                            self.y,
                                            hiddenLayerSizes[lastIndex],
                                            nOuts);
}


/**
 *   Training hidden layers with unsupervised learning
 *   
 *   @method pretrain
 *   @param  lr {float} learning rate
 *   @param  corruptionLevel {float} the standard deviation which is used by denoised autoencoder
 *   @param  epochs {int} the number of times of running gradient decent
 */
SdA.prototype.pretrain = function(lr, corruptionLevel, epochs, input) {
    var self = this;
    // Training each layer by using denoised autoencoder with unsupervised learning

    for (var layerIndex = 0; layerIndex < self.nLayers; layerIndex++) {
        var layerInput;
        if (layerIndex == 0) {
            layerInput = (input != undefined)? input: self.x;
        } else {
            layerInput = self.sigmoidLayers[layerIndex-1].sampleHGivenV(layerIndex);
        }

        var da = self.dALayers[layerIndex];

        for (var i = 0; i < epochs; i++) {
            da.train(lr, corruptionLevel, layerInput);
        }

        self.sigmoidLayers[layerIndex].W = da.W;
        self.sigmoidLayers[layerIndex].b = da.hBias;
    }

}

/**
 *   Training logistics regression algorithm which is on output layer
 *   @method train
 *   @param  lr {float} learning rate
 *   @param  epochs {int} the number of times of running gradient decent
 */
SdA.prototype.finetune = function(lr, epochs) {
    var self = this;
    var lastIndex = self.sigmoidLayers.length-1;
    var layerInput = self.sigmoidLayers[lastIndex].sampleHGivenV();
    for (var i = 0; i < epochs; i++) {
        self.logLayer.train(lr, 0.01, layerInput);
        lr *= 0.95;
    }
}

/**
 *   Predict label with training data
 *
 *   @method predict
 *   @param  x {Matrix} input data
 */
SdA.prototype.predict = function(x) {
    var self = this;
    var layerInput = x;
    
    for (var layerIndex = 0; layerIndex < self.nLayers; layerIndex++) {
        var sigmoidLayer = self.sigmoidLayers[layerIndex];
        layerInput = sigmoidLayer.output(layerInput);
    }

    return self.logLayer.predict(layerInput);
}


module.exports = SdA;