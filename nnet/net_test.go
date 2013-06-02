package nnet

import (
	"testing"
)

func TestNewNeuralNetWithDefaults(t *testing.T) {
	nn := NewNeuralNetWithDefaults()
	if nn.NumInputs != TotalInputs {
		t.Error("Number of inputs should be set correctly for Net")
	}
	if nn.NumOutputs != TotalOutputs {
		t.Error("Number of outputs should be set correctly for Net")
	}
	if nn.NumHiddenLayers != TotalHiddenLayers {
		t.Error("Number of hidden layers should be set correctly for Net")
	}
	if nn.NeuronsPerHiddenLayer != TotalNeuronsPerHiddenLayer {
		t.Error("Number of neurons per hidden layer should be set correctly for Net")
	}
	if len(nn.Layers) != 0 {
		t.Error("Layers should be defaulted to a zero length")
	}
}
