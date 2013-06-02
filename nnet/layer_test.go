package nnet

import (
	"testing"
)

func TestSetupInputLayer(t *testing.T) {
	nl := &NeuronLayer{}
	numInputs, numNeurons := 2, 2
	nl.SetupInputLayer(numNeurons, numInputs)
	if nl.NumNeurons != numNeurons {
		t.Error("Number of neurons on layer should be set correctly")
	}
	if len(nl.Neurons) != numNeurons {
		t.Error("Total neurons should match length of neurons for layer")
	}
	for _, neuron := range nl.Neurons {
		if neuron.TotalInputs != numInputs {
			t.Error("Neuron total inputs should be set correctly")
		}

		if len(neuron.Weights) > 0 {
			t.Error("Weights should not be set for input layer neurons")
		}
	}
}

func TestSetupLayer(t *testing.T) {
	nl := &NeuronLayer{}
	numInputs, numNeurons := 2, 2
	nl.SetupNeuronLayer(numNeurons, numInputs)
	if nl.NumNeurons != numNeurons {
		t.Error("Number of neurons on layer should be set correctly")
	}
	if len(nl.Neurons) != numNeurons {
		t.Error("Total neurons should match length of neurons for layer")
	}
	for _, neuron := range nl.Neurons {
		if neuron.TotalInputs != numInputs {
			t.Error("Neuron total inputs should be set correctly")
		}

		if len(neuron.Weights) == 0 {
			t.Error("Weights should be set for neurons in a non input layer")
		}
	}
}

func TestLayerOutputs(t *testing.T) {
	nl := &NeuronLayer{}
	numInputs, numNeurons := 2, 2
	nl.SetupNeuronLayer(numNeurons, numInputs)
	outputs := nl.Outputs()
	expected := []float64{0.0, 0.0}
	errored := false
	for i, val := range outputs {
		if val != expected[i] {
			errored = true
			break
		}
	}
	if errored {
		t.Error("Layer outputs should equal all of it's neurons outputs")
	}
}
