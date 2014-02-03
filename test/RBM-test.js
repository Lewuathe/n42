 vows = require('vows'),
    assert = require('assert');

var RBM = require('../lib/RBM.js');
var utils  = require('../lib/utils.js');
var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var _ = require('underscore');

vows.describe('n42 RBM  module').addBatch({
    'Restricted Boltzmann machine': {
        'get correct answer': {
            topic: function() {

                var input = $M([[1,1,1,0,0,0],
                                [1,0,1,0,0,0],
                                [1,1,1,0,0,0],
                                [0,0,1,1,1,0],
                                [0,0,1,1,0,0],
                                [0,0,1,1,1,0]]);
                            
                
                return new RBM(input, 6, 2);
            },
            
            'should correct propup output': function (rbm) {
                var input = $M([[1,1,1,0,0,0],
                                [1,0,1,0,0,0],
                                [1,1,1,0,0,0],
                                [0,0,1,1,1,0],
                                [0,0,1,1,0,0],
                                [0,0,1,1,1,0]]);
                
                assert.isNotNull(rbm.propup(input));
            },
            'should correct propdown output': function(rbm) {
                var input = $M([[1.0, 0.0],
                                [1.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 1.0],
                                [0.0, 1.0],
                                [0.0, 1.0]]);

                assert.isNotNull(rbm.propdown(input));
            },
            'should correct sample h given v': function(rbm) {
                var input = $M([[1,1,1,0,0,0],
                                [1,0,1,0,0,0],
                                [1,1,1,0,0,0],
                                [0,0,1,1,1,0],
                                [0,0,1,1,0,0],
                                [0,0,1,1,1,0]]);
                
                var ret = rbm.sampleHGivenV(input)
                assert.isNotNull(ret);
            },
            'should correct sample v given h': function(rbm) {
                var input = $M([[1.0, 0.0],
                                [1.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 1.0],
                                [0.0, 1.0],
                                [0.0, 1.0]]);
                var ret = rbm.sampleVGivenH(input);
                assert.isNotNull(ret);
            },

            'should train propery': function(rbm) {
                for (var i = 0; i < 1000; i++) {
                    rbm.contrastiveDivergence(0.3, 1);
                }
                var v = $M([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]);
                assert.isNotNull(rbm.reconstruct(v));
            }
            
        }
    }

}).export(module); // Export the Suite