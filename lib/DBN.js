/**
 *  Deep Belief Nets
 *
 *  @module n42
 *  @class  DBN
 *  @author Kai Sasaki
 *  @since  2014/02/03
 *
 */

var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var utils  = require('./utils.js');

var HiddenLayer = require('./HiddenLayer.js');
var RBM         = require('./RBM.js');
var LogisticsRegression = require('./LogisticsRegression.js');

/**
 *  Deep Belief Nets
 *
 *  @class DBN
 *  @constructor
 */

function DBN(input, label, nIns, hiddenLayerSizes, nOuts) {
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
    self.sigmoidLayers = []

    // Restricted Boltzmann machine layer
    self.rbmLayers = [];
    /**
     *  number of hidden layers
     *
     *  @property nLayers
     *  @type     int
     */
    self.nLayers = hiddenLayerSizes.length;
    for (var layerIndex = 0; layerIndex < self.nLayers; layerIndex++) {
        
        // Select input size
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

        var sigmoidLayer = new HiddenLayer(layerInput,
                                           inputSize,
                                           hiddenLayerSizes[layerIndex],
                                           utils.sigmoid);

        self.sigmoidLayers.push(sigmoidLayer);
        
        // Constrcut Restricted Boltzmann machine layer
        var rbmLayer = new RBM(layerIndex,
                               inputSize,
                               hiddenLayerSizes[layerIndex],
                               sigmoidLayer.W,
                               sigmoidLayer.b);
        self.rbmLayers.push(rbmLayer);
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
 *   @param  k  {int} the number of phase
 *   @param  epochs {int} the number of times of running gradient decent
 */

DBN.prototype.pretrain = function(lr, k, epochs, input) {
    var self = this;
    // Training each layer by using Restricted boltzmann machine with unsupervised learning

    for (var layerIndex = 0; layerIndex < self.nLayers; layerIndex++) {
        var layerInput;
        if (layerIndex == 0) {
            layerInput = self.x;
            layerInput = (input != undefined)? input: self.x;
        } else {
            layerInput = self.sigmoidLayers[layerIndex-1].sampleHGivenV(layerIndex);
        }

        var rbm = self.rbmLayers[layerIndex];
        for (var i = 0; i < epochs; i++) {
            rbm.contrastiveDivergence(lr, k, layerInput);
        }
    }
}

/**
 *   Training logistics regression algorithm which is on output layer
 *   @method train
 *   @param  lr {float} learning rate
 *   @param  epochs {int} the number of times of running gradient decent
 */

DBN.prototype.finetune = function(lr, epochs) {
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

DBN.prototype.predict = function(x) {
    var self = this;
    var layerInput = x;

    for (var layerIndex = 0; layerIndex < self.nLayers; layerIndex++) {
        var sigmoidLayer = self.sigmoidLayers[layerIndex];
        layerInput = sigmoidLayer.output(layerInput);
    }

    return self.logLayer.predict(layerInput);
}


module.exports = DBN;