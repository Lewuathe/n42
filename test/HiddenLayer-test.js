 vows = require('vows'),
    assert = require('assert');

var HiddenLayer = require('../lib/HiddenLayer.js');
var utils  = require('../lib/utils.js');
var Matrix = require('sylvester').Matrix;
var Vector = require('sylvester').Vector;
var _ = require('underscore');

vows.describe('n42 HiddenLayer module').addBatch({
    'Simple hidden layer': {
        'get correct answer': {
            topic: function() {
                return new HiddenLayer($M([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
                                       3, 2, utils.sigmoid);
            },
            
            'should receive not null': function (hiddenLayer) {
                assert.isNotNull((hiddenLayer.output($M([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))));
            }
        }
    }

}).export(module); // Export the Suite