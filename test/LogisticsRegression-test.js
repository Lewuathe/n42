var vows = require('vows'),
    assert = require('assert');

var utils = require('../lib/utils.js');
var LogisticsRegression = require('../lib/LogisticsRegression.js');
var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var _ = require('underscore');

vows.describe('n42 LogisticsRegression test').addBatch({
    'LogisticsRegression training': {
        'get train result': {
            topic: function() {
                return new LogisticsRegression($M([[1.0,0.0,0.0],[0.0,0.0,1.0]]),
                                               $M([[1.0,0.0], [0.0,1.0]]),
                                               3, 2);
            },
            
            'should not throw exception': function (lr) {
                assert.isTrue(lr.train(0.1, 0.01));
            }
        },

        'get predict result': {
            topic: function() {
                return  new LogisticsRegression($M([[1.0,0.0,0.0],[0.0,0.0,1.0]]),
                                                $M([[1.0,0.0], [0.0,1.0]]),
                                                3, 2);
            },
            
            'should not throw exception': function(lr) {
                for (var i = 0; i < 100; i++) {
                    lr.train(0.1, 0.01);
                }
                console.log(lr.predict($M([[0.0,0.1,0.8]])));
            }
        }
    }

}).export(module); // Export the Suite