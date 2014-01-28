/*
 *  Module dependencies
 */ 
var LogisticsRegression = require('./LogisticsRegression.js');
var SdA                 = require('./SdA.js');

/*
 *  Framework version
 */
require('pkginfo')(module, 'version');


/*
 *  Expose constructors
 */
module.exports.LogisticsRegression = LogisticsRegression;
module.exports.SdA = SdA;