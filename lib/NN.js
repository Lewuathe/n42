/*
 *   Newral network
 *
 *   @module n42
 *   @class  NN
 *   @author Kai Sasaki
 *   @since  2014/02/12
 *
 */

var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var utils  = require('./utils.js');
var assert = require('assert');
var generator = require('box-muller');

function NN(input, nVisible, nHidden, nOutput, label, W1, W2) {
    var self = this;
    self.input    = input;
    self.nVisible = nVisible;
    self.nHidden  = nHidden;
    self.label    = label;

    // Initialize weight1 parameter
    self.W1     = (W1 != undefined)? W1 : Matrix.Random(nVisible, nHidden);

    // Initialize weight2 parameter
    self.W2     = (W2 != undefined)? W2 : Matrix.Random(nHidden, nOutput);

    // Initialize hidden bias parameters
    self.hBias = Vector.Zero(nHidden);

    // Initialize visual bias parameters
    self.vBias = Vector.Zero(nOutput);

}

NN.prototype.getHiddenValues = function(input) {
    var self = this;
    // Calculate plus weight
    var rowValues = input.x(self.W1);
    return utils.sigmoid(utils.plusBias(rowValues, self.hBias));
}

NN.prototype.getOutput = function(hidden) {
    var self = this;
    var rowValues = hidden.x(self.W2);
    return utils.sigmoid(utils.plusBias(rowValues, self.vBias));
}

NN.prototype.train = function(lr, input, label) {
    var self = this;
    self.x     = (input != undefined)? input : self.input;
    self.label = (label != undefined)? label : self.label;
    
    var x = self.x;
    var y = self.getHiddenValues(x);
    var z = self.getOutput(y);

    var lH2 = self.label.subtract(z);
    var sigma = lH2.x(self.W2.transpose());
    var lH1 = [];
    for (var i = 0; i < sigma.rows(); i++) {
        lH1.push([]);
        for (var j = 0; j < sigma.cols(); j++) {
            lH1[i].push(sigma.e(i+1, j+1) * y.e(i+1, j+1) * (1 - y.e(i+1, j+1)));
        }
    }
    lH1 = $M(lH1);
    var lW1 = x.transpose().x(lH1);
    var lW2 = y.transpose().x(lH2);

    self.W1 = self.W1.add(lW1.x(lr));
    self.W2 = self.W2.add(lW2.x(lr));

    self.vBias = self.vBias.add(utils.mean(lH2, 0).x(lr));
    self.hBias = self.hBias.add(utils.mean(lH1, 0).x(lr));
}


NN.prototype.predict = function(x) {
    var self = this;
    var y = self.getHiddenValues(x);
    var z = self.getOutput(y);
    return z
}


module.exports = NN;