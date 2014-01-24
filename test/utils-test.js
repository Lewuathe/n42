var vows = require('vows'),
    assert = require('assert');

var utils = require('../lib/utils.js');
var Matrix = require('sylvester').Matrix;
var _ = require('underscore');

vows.describe('n42 Utils test').addBatch({
    'sigmoid function': {
        topic: function() {
            return utils.sigmoid($M([0.0, 0.0, 0.0]));
        },

        'should receive 0.5': function (topic) {
            assert.deepEqual(topic, $M([0.5,0.5,0.5]));
        }
    },

    'softmax function': {
        topic: function() {
            return utils.softmax($M([[1.0, 1.0], [1.0, 1.0]]));
        },

        'should receive 0.5': function (topic) {
            assert.deepEqual(topic, $M([[0.5, 0.5], [0.5, 0.5]]));
        }
    }

}).export(module); // Export the Suite