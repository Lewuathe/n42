/*
 *  Hidden Layer
 *
 *  @module n42
 *  @class  HiddenLayer
 *  @author Kai Sasaki
 *  @since  2014/01/26
 *
 */

var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var utils  = require('./utils.js');

function HiddenLayer(input, nIn, nOut, W, b, activation) {
    var self = this;

    // Initilize weight parameter
    self.W = (W != undefined)? W: Matrix.Random(nIn, nOut);

    // Initilize bias parameter
    self.b = (b != undefined)? b: Vector.Zero(nOut);

    self.input = input;
    self.activation = activation;
}

HiddenLayer.prototype.output = function(input) {
    var self = this;
    var linearOutput = utils.plusBias(input.x(self.W), self.b);
    return (self.activation == undefined) ? linearOutput : self.activation(linearOutput);
}

HiddenLayer.prototype.sampleHGivenV = function(input) {
    return 0;
}

module.exports = HiddenLayer;
