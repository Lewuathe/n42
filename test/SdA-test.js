 vows = require('vows'),
    assert = require('assert');

var SdA    = require('../lib/SdA.js');
var utils  = require('../lib/utils.js');
var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var _ = require('underscore');

vows.describe('n42 Stacked autoencoder module').addBatch({
    'Stacked autoencoder layer': {
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

                return new SdA(input, label, 4, [3, 3], 2);
            },
            
            'should receive 0.5': function (sda) {
                sda.pretrain(0.3, 0.01, 1000);
                sda.finetune(0.3, 50);
                var data = $M([
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0]
                ]);
                console.log(sda.predict(data));
            }
        }
    }

}).export(module); // Export the Suite