var vows = require('vows'),
    assert = require('assert');

var utils = require('../lib/utils.js');
var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var _ = require('underscore');

vows.describe('n42 Utils test').addBatch({
    'sigmoid function': {
        'get correct answer': {
            topic: function() {
                return utils.sigmoid($M([0.0, 0.0, 0.0]));
            },
            
            'should receive 0.5': function (topic) {
                assert.deepEqual(topic, $M([0.5,0.5,0.5]));
            }
        }
    },

    'softmax function': {
        'get matrix which has 2 dimensions': {
            topic: function() {
                return utils.softmax($M([[1.0, 1.0], [1.0, 1.0]]));
            },
            
            'should receive 0.5': function (topic) {
                assert.deepEqual(topic, $M([[0.5, 0.5], [0.5, 0.5]]));
            }
        },
        
        'get matrix which has 1 dimensions': {
            topic: function() {
                return utils.softmax($M([[1.0, 1.0]]));
            },

            'should receive 0.5': function(topic) {
                assert.deepEqual(topic, $M([[0.5, 0.5]]));
            }
        },

        'get vector': {
            topic: function() {
                return utils.softmax($V([1.0, 1.0]));
            },

            'should receive 0.5': function(topic) {
                assert.deepEqual(topic, $V([0.5, 0.5]));
            }
        }
    },

    'mean function': {
        'get mean value of axis = 0': {
            topic: function() {
                return utils.mean($M([[1.0, 2.0], [3.0, 4.0]]), 0);
            },

            'shold receive row oriented mean values': function(topic) {
                assert.deepEqual(topic, $V([2.0, 3.0]));
            }
        },

        'get mean value of axis = 1': {
            topic: function() {
                return utils.mean($M([[1.0, 2.0], [3.0, 4.0]]), 1);
            },

            'should receive col oriented mean values': function(topic) {
                assert.deepEqual(topic, $V([1.5, 3.5]));
            }
        }
    },

    'plusBias function': {
        'get plusBias value': {
            topic: function() {
                return utils.plusBias($M([[1.0, 1.0], [2.0,2.0]]), $V([1.0, 1.0]));
            },

            'should receive 2.0 and 3.0 matrix': function(topic) {
                assert.deepEqual(topic, $V([[2.0, 2.0], [3.0, 3.0]]));
            }
        }
    }

}).export(module); // Export the Suite