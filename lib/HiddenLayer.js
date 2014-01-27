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

function HiddenLayer(input, nIn, nOut, activation, W, b) {
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
    var x = (input != undefined)? input : self.input;
    var linearOutput = utils.plusBias(x.x(self.W), self.b);
    return (self.activation == undefined) ? linearOutput : self.activation(linearOutput);
}

HiddenLayer.prototype.sampleHGivenV = function(input) {
    var self = this;
    var x = (input != undefined)? input : self.input;
    return self.output();
}

module.exports = HiddenLayer;
