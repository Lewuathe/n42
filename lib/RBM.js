/*
 *  Restricted Boltzmann machine
 *
 *  @module n42
 *  @class  RBM
 *  @author Kai Sasaki
 *  @since 2014/01/28
 *
 */

var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var utils  = require('./utils.js');
var binomial = require('binomial-sampling');
var generator = require('box-muller');

function RBM(input, nVisible, nHidden, W, hBias, vBias) {
    var self = this;

    self.input = input;
    self.nVisible = nVisible;
    self.nHidden  = nHidden;

    // Initialize weight parameter
    self.W     = (W != undefined)? W : Matrix.Random(nVisible, nHidden);

    // Initialize hidden bias parameters
    self.hBias = (hBias != undefined)? hBias : Vector.Zero(nHidden);

    // Initialize visual bias parameters
    self.vBias = (vBias != undefined)? vBias : Vector.Zero(nVisible);
}

RBM.prototype.contrastiveDivergence = function(lr, k, input) {
    var self = this;
    self.input = (input != undefined)? input : self.input
    var ph = self.sampleHGivenV(self.input);
    
    // Select phSample
    var chainStart = ph[1];

    // stepRet[0] v1Means
    // stepRet[1] v1Samples
    // stepRet[2] h1Means
    // stepRet[3] h1Sample

    for (var step = 0; step < k; step++) {
        var stepRet;
        if (step == 0) {
            stepRet = self.gibbsHvh(chainStart);
        } else {
            // Select a nhSample
            stepRet = self.gibbsHvh(stepRet[3]);
        }
    }

    // W += lr * (W.T * phSample - nvSample.T * nhMeans)
    var witem1 = self.input.transpose().x(ph[1]).x(lr);
    var witem2 = stepRet[1].transpose().x(stepRet[2]).x(lr)
    self.W = self.W.add(witem1.subtract(witem2));

    var vitem1 = self.input.x(lr);
    var vitem2 = stepRet[1].x(lr);
    self.vBias = self.vBias.add(utils.mean(vitem1.subtract(vitem2), 0));

    var hitem1 = ph[1].x(lr);
    var hitem2 = stepRet[2].x(lr);
    self.hBias = self.hBias.add(utils.mean(hitem1.subtract(hitem2), 0));
    
}

RBM.prototype.sampleHGivenV = function(v0Sample) {
    var self = this;
    var h1Means = self.propup(v0Sample);
    var h1Sample = [];
    for (var i = 0; i < h1Means.rows(); i++) {
        h1Sample.push([]);
        for (var j = 0; j < h1Means.cols(); j++) {
            h1Sample[i].push(binomial(1, h1Means.e(i+1, j+1)));
            //h1Sample[i].push(generator() + h1Means.e(i+1, j+1));
        }
    }
    h1Sample = $M(h1Sample);
    return [h1Means, h1Sample];
}

RBM.prototype.sampleVGivenH = function(h0Sample) {
    var self = this;
    var v1Means = self.propdown(h0Sample);
    var v1Sample = [];
    for (var i = 0; i < v1Means.rows(); i++) {
        v1Sample.push([]);
        for (var j = 0; j < v1Means.cols(); j++) {
            v1Sample[i].push(binomial(1, v1Means.e(i+1, j+1)));
            //v1Sample[i].push(generator()*0.01 + v1Means.e(i+1, j+1))
        }
    }
    v1Sample = $M(v1Sample);
    return [v1Means, v1Sample];
}

RBM.prototype.gibbsHvh = function(h0Sample) {
    var self = this;
    var retV1 = self.sampleVGivenH(h0Sample);
    // Given h1Sample == retV1[1]
    var retH1 = self.sampleHGivenV(retV1[1]);
    return [retV1[0], retV1[1], retH1[0], retH1[1]];
}

RBM.prototype.propup = function(v) {
    var self = this;
    var preSigmoidActivation = utils.plusBias(v.x(self.W), self.hBias);
    return utils.sigmoid(preSigmoidActivation);
}

RBM.prototype.propdown = function(h) {
    var self = this;
    var preSigmoidActivation = utils.plusBias(h.x(self.W.transpose()), self.vBias);
    return utils.sigmoid(preSigmoidActivation);
}

RBM.prototype.reconstruct = function(v) {
    var self = this;
    var h = utils.sigmoid(utils.plusBias(v.x(self.W), self.hBias));
    var reconstructedV = utils.sigmoid(utils.plusBias(h.x(self.W.transpose()), self.vBias));
    return reconstructedV;
}

module.exports = RBM;