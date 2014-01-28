/**
 *  Logistics Regression module. 
 *
 *  @module n42
 *  @constructor
 *  @author Kai Sasaki
 *  @since  2014/01/26  
 *
 */

var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var utils  = require('./utils.js');

/**
 *   LogisticsRegression module. 
 *   This module is used output layer.After trainging all hidden layers, 
 *   this module is tuned for fitting to training data
 *   
 *   @class LogisticsRegression
 *   @constructor
 *
 */
function LogisticsRegression(input, label, nIn, nOut) {
    var self = this;

    /**
     *  input data. This type is defined in sylvester library
     *
     *  @property input
     *  @type     Matrix
     */
    self.input = input;

    /**
     *  label data. This type is defined in sylvester library
     *  
     *  @property y
     *  @type     Matrix
     */
    self.y = label;

    /**
     *   weight parameter which is obtained after training
     *
     *   @property W
     *   @type     Matrix
     */
    self.W = Matrix.Zero(nIn, nOut);

    /**
     *   bias parameter which is obtained after training
     *
     *   @property b
     *   @type     Vector
     */
    self.b = Vector.Zero(nOut);
}

/**
 *   Training logistics regression algorithm
 *   @method train
 *   @param  lr {float} learning rate
 *   @param  l2Reg {float} regularization parameter
 *   @param  input {Matrix} input data (option)
 */
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

/**
 *   Predict label with training data
 *
 *   @method predict
 *   @param  x {Matrix} input data
 */
LogisticsRegression.prototype.predict = function(x) {
    var self = this;
    return utils.softmax(utils.plusBias(x.x(self.W), self.b));
}

module.exports = LogisticsRegression;