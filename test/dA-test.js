 vows = require('vows'),
    assert = require('assert');

var dA     = require('../lib/dA.js');
var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var _ = require('underscore');

vows.describe('n42 Denoised autoencoder test').addBatch({
    'Denoised autoencoder': {
        'get correct answer': {
            topic: function() {
                return new dA($M([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), 2, 1);
            },
            
            'should receive 0.5': function (autoEncoder) {
                assert.isNotNull(autoEncoder);
            }
        },

        'get corrupt data': {
            topic: function() {
                return new dA($M([[1.0, 1.0], [1.0,1.0]]), 2, 1);
            },
            
            'should receive corrupted data': function(autoEncoder) {
                assert.isNotNull(autoEncoder);
            }
        },

        'get hidden value': {
            topic: function() {
                return new dA($M([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]), 3, 2);
            },

            'should get receive hidden value': function(autoEncoder) {
                var hidden = autoEncoder.getHiddenValues($M([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]));
                assert.isNotNull(hidden);
            }
        },

        'train input': {
            topic: function() {
                var data = [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ];
                return new dA($M(data), 3, 2);
            },

            'should get receive hidden value': function(autoEncoder) {
                for (var i = 0; i < 1000; i++) {
                    autoEncoder.train(0.1, 0.02);
                }
                console.log(autoEncoder.reconstruct($M([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])));
            }
        },
        
    }

}).export(module); // Export the Suite