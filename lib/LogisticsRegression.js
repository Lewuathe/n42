/*
 *  Logistics Regression module
 *
 *  @module n42
 *  @class LogisticsRegression
 *  @author Kai Sasaki
 *  @since  2014/01/26  
 *
 */

var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var utils  = require('./utils.js');

function LogisticsRegression(input, label, nIn, nOut) {
    var self = this;
    self.input = input;
    self.y = label;

    self.W = Matrix.Zero(nIn, nOut);
    self.b = Vector.Zero(nOut);
}

LogisticsRegression.prototype.train = function(lr, l2Reg, input) {
    var self = this;
    self.x = (input != undefined)? input : self.input
    var rowValues = self.x.x(self.W);
    var probYGivenX = utils.softmax(utils.plusBias(rowValues, self.b));
    var dY = self.y.subtract(probYGivenX);

    self.W = self.W.add(self.x.transpose().x(dY).x(lr).subtract(self.W.x(l2Reg).x(lr)));
    self.b = self.b.add(utils.mean(dY, 0));

    return true;
}

LogisticsRegression.prototype.predict = function(x) {
    var self = this;
    return utils.softmax(utils.plusBias(x.x(self.W), self.b));
}

module.exports = LogisticsRegression;