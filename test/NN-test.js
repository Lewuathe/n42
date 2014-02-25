vows = require('vows');
assert = require('assert');

var NN = require('../lib/NN.js');
var utils = require('../lib/utils.js');
var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;

vows.describe('n42 Newral network module').addBatch({
    'Newral Network': {
        'get correct answer': {
            topic: function() {
                var input = $M([
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.2, 0.0],
                    [1.0, 0.9, 0.1, 0.0],
                    [1.0, 0.98, 0.02, 0.0],
                    [0.98, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 0.8, 1.0],
                    [0.0, 0.0, 0.9, 1.0],
                    [0.0, 0.0, 1.0, 0.9],
                    [0.0, 0.0, 0.98, 1.0]
                ]);

                var label = $M([
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0]
                ]);

                return new NN(input, 4, 10, 2, label);
            },

            'should predict correctly': function(nn) {
                for (var i = 0; i < 10000; i++) {
                    nn.train(0.1);
                }
                var data = $M([
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0]
                ]);
                console.log(nn.predict(data));
            }
        }
    }
}).export(module);