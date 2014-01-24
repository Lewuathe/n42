var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;

function sigmoid(x) {
    return x.map(function(val) { return 1.0 / (1.0 + Math.exp(-val)); });
}

function softmax(x) {
    var max = x.max();
    var e = x.map(function(val) { 
        return Math.exp(val - max);
    });
    // Return value
    var ret = [];

    if (e.distanceFrom != undefined) {
        // In case of x is Vector
        var total = 0.0;
        e.map(function(val) {
            total += Math.exp(val);
        });
        var a = e.map(function(val) {
            return Math.exp(val) / total;
        });
        return a;
    } else {
        // In case of x is Matrix
        // Each rows
        for (var i = 1; i <= e.rows(); i++) {
            ret.push([]);
            var row = e.row(i);
            var total = 0.0;
            row.map(function(val) {
                total += Math.exp(val);
            });
            var a = row.map(function(val) {
                return Math.exp(val) / total;
            });
            for (var j = 1; j <= e.cols(); j++) {
                ret[i-1].push(a.e(j));
            }
        }
        return $M(ret);
    }

}

module.exports.sigmoid = sigmoid;
module.exports.softmax = softmax;