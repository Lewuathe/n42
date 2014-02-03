vows = require('vows');
assert = require('assert');

var DBN    = require('../lib/DBN.js');
var utils  = require('../lib/utils.js');
var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;

vows.describe('n42 Deep Belief Network module').addBatch({
    'Deep Belief Network': {
        'get correct answer': {
            topic: function() {
                var input = $M([
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.2, 0.0],
                    [1.0, 0.9, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.8, 1.0],
                    [0.0, 0.0, 1.0, 1.0]
                ]);
                var label = $M([
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0]
                ]);
                return new DBN(input, label, 4, [3, 3], 2);
            },
            
            'should predict correctly': function(dbn) {
                dbn.pretrain(0.3, 1, 1000);
                dbn.finetune(0.3, 200);
                var data = $M([
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0]
                ]);
                console.log(dbn.predict(data));
            }
        }
    }
}).export(module);