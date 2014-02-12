/*
 *  Module dependencies
 */ 
var LogisticsRegression = require('./LogisticsRegression.js');
var SdA                 = require('./SdA.js');
var NN                  = require('./NN.js');
var DBN                 = require('./DBN.js');

/*
 *  Framework version
 */
require('pkginfo')(module, 'version');


/*
 *  Expose constructors
 */
module.exports.LogisticsRegression = LogisticsRegression;
module.exports.SdA = SdA;
module.exports.NN  = NN;
module.exports.DBN = DBN;