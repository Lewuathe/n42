/*
 *   Denoised Auto encoder
 *
 *   @module n42
 *   @class  dA
 *   @author Kai Sasaki
 *   @since  2014/01/25
 *
 */

var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var utils  = require('./utils.js');
var assert = require('assert');
var generator = require('box-muller');

function dA(input, nVisible, nHidden, W, hBias, vBias) {
    var self = this;
    self.input    = input;
    self.nVisible = nVisible;
    self.nHidden  = nHidden;
    // Initialize weight parameter
    self.W     = (W != undefined)? W : Matrix.Random(nVisible, nHidden);

    // Initialize hidden bias parameters
    self.hBias = (hBias != undefined)? hBias : Vector.Zero(nHidden);

    // Initialize visual bias parameters
    self.vBias = (vBias != undefined)? vBias : Vector.Zero(nVisible);

    self.wPrime = self.W.transpose();
}

dA.prototype.getCorruptedInput = function(input, corruptionLevel) {
    assert.ok(corruptionLevel < 1);
    noised = [];
    for (var i = 0; i < input.rows(); i++) {
        noised.push([]);
        for (var j = 0; j < input.cols(); j++) {
            noised[i].push((generator() * corruptionLevel + 1.0) * input.e(i+1, j+1));;
        }
    }
    return $M(noised);
}

dA.prototype.getHiddenValues = function(input) {
    var self = this;
    // Calculate plus weight
    var rowValues = input.x(self.W);
    return utils.sigmoid(utils.plusBias(rowValues, self.hBias));
}

dA.prototype.getReconstructedInput = function(hidden) {
    var self = this;
    var rowValues = hidden.x(self.W.transpose());
    return utils.sigmoid(utils.plusBias(rowValues, self.vBias));
}

dA.prototype.train = function(lr, corruptionLevel, input) {
    var self = this;
    self.x = (input != undefined)? input : self.input;
    
    var x = self.x;
    var tildeX = self.getCorruptedInput(x, corruptionLevel);
    var y = self.getHiddenValues(tildeX);
    var z = self.getReconstructedInput(y);
    
    var lH2 = x.subtract(z);
    var sigma = lH2.x(self.W);
    var lH1 = [];
    for (var i = 0; i < sigma.rows(); i++) {
        lH1.push([]);
        for (var j = 0; j < sigma.cols(); j++) {
            lH1[i].push(sigma.e(i+1, j+1) * y.e(i+1, j+1) * (1 - y.e(i+1, j+1)));
        }
    }
    lH1 = $M(lH1);


    var lW = tildeX.transpose().x(lH1).add(lH2.transpose().x(y));

    self.W = self.W.add(lW.x(lr));


    self.vBias = self.vBias.add(utils.mean(lH2, 0).x(lr));
    self.hBias = self.hBias.add(utils.mean(lH1, 0).x(lr));
}


dA.prototype.reconstruct = function(x) {
    var self = this;
    var y = self.getHiddenValues(x);
    var z = self.getReconstructedInput(y);
    return z
}


module.exports = dA;