var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;

/*
 *  plugBias returns the matrix which is sum of 
 *  all data and bias vector
 */
function plusBias(x, b) {
    // x is n(data number) * dimension matrix
    // b is dimension vector
    var ret = [];
    for (var i = 0; i < x.rows(); i++) {
        ret.push([]);
        for (var j = 0; j < x.cols(); j++) {
            ret[i].push(x.e(i+1, j+1) + b.e(j+1));
        }
    }
    return $M(ret);
}

function mean(x, axis) {
    var sums = [];
    if (axis == 0) {
        for (var i = 0; i < x.cols(); i++) {
            sums.push(0);
        }
    
        for (var i = 0; i < x.rows(); i++) {
            for (var j = 0; j < x.cols(); j++) {
                sums[j] += x.e(i+1, j+1);
            }
        }
        var ret = $V(sums);
        return ret.x(1. / x.rows());
    } else if(axis == 1) {
        for (var i = 0; i < x.rows(); i++) {
            sums.push(0);
        }
        for (var i = 0; i < x.rows(); i++) {
            for (var j = 0; j < x.cols(); j++) {
                sums[i] += x.e(i+1, j+1);
            }
        }
        var ret = $V(sums);
        return ret.x(1. / x.cols());
    }
}

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


module.exports.plusBias = plusBias;
module.exports.mean     = mean;
module.exports.sigmoid  = sigmoid;
module.exports.softmax  = softmax;