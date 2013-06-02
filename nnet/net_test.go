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

func TestSetupWithNoHiddenLayers(t *testing.T) {
	nn := NewNeuralNetWithDefaults()
	nn.NumHiddenLayers = 0
	nn.SetupNeuralNet()
	if len(nn.Layers) != 2 {
		t.Error("Net should only have an input and output layer if no hidden layers")
	}
}

func TestSetupWithHiddenLayers(t *testing.T) {
	nn := NewNeuralNetWithDefaults()
	nn.SetupNeuralNet()
	if len(nn.Layers) != 2+nn.NumHiddenLayers {
		t.Errorf("Net should have %d hidden layers\n", nn.NumHiddenLayers)
	}
}

func TestGettingWeights(t *testing.T) {
	nn := NewNeuralNetWithDefaults()
	nn.SetupNeuralNet()
	weights := nn.GetWeights()
	expectedTotal := (nn.NumInputs * nn.NeuronsPerHiddenLayer * nn.NumHiddenLayers) + (nn.NeuronsPerHiddenLayer * nn.NumOutputs)
	if len(weights) != expectedTotal {
		t.Errorf("Total weights should be %d but got %d\n", len(weights), expectedTotal)
	}
}

func TestTotalWeights(t *testing.T) {
	nn := NewNeuralNetWithDefaults()
	nn.SetupNeuralNet()
	total := nn.TotalWeights()
	expectedTotal := (nn.NumInputs * nn.NeuronsPerHiddenLayer * nn.NumHiddenLayers) + (nn.NeuronsPerHiddenLayer * nn.NumOutputs)
	if total != expectedTotal {
		t.Errorf("Total weights should be %d but got %d\n", total, expectedTotal)
	}
}

func TestSettingNewWeights(t *testing.T) {
	nn := NewNeuralNetWithDefaults()
	nn.SetupNeuralNet()
	newWeights := make([]float64, 0)
	for _, _ = range nn.GetWeights() {
		newWeights = append(newWeights, 0.0)
	}
	nn.setWeights(newWeights)
	for _, weight := range nn.GetWeights() {
		if weight != 0.0 {
			t.Error("New weight was unable to be set")
		}
	}
}

func TestNetOutputs(t *testing.T) {
	nn := NewNeuralNetWithDefaults()
	nn.SetupNeuralNet()
	newWeights := make([]float64, 0)
	for _, _ = range nn.GetWeights() {
		newWeights = append(newWeights, 0.0)
	}
	nn.setWeights(newWeights)
	outputs := nn.Outputs()
	if len(outputs) != nn.NeuronsPerHiddenLayer {
		t.Error("Total outputs should equal the neurons per hidden layer")
	}
	nn.Predict([]float64{0.0, 0.0})
	outputs = nn.Outputs()
	for _, out := range outputs {
		if out != 0.5 {
			t.Error("Incorrect output based on weights of net")
		}
	}
}

func TestClearingDeltas(t *testing.T) {
	nn := NewNeuralNetWithDefaults()
	nn.SetupNeuralNet()
	for _, layer := range nn.Layers {
		for _, neuron := range layer.Neurons {
			neuron.delta = 100.00
		}
	}
	nn.ClearDeltas()
	for _, layer := range nn.Layers {
		for _, neuron := range layer.Neurons {
			if neuron.delta != 0.0 {
				t.Error("Failed to clear neuron delta")
			}
		}
	}
}

func TestPredictZeroWeights(t *testing.T) {
	nn := NewNeuralNetWithDefaults()
	nn.SetupNeuralNet()
	newWeights := make([]float64, 0)
	for _, _ = range nn.GetWeights() {
		newWeights = append(newWeights, 0.0)
	}
	nn.setWeights(newWeights)
	nn.Predict([]float64{0.0, 0.0})
	for _, out := range nn.Outputs() {
		if out != 0.5 {
			t.Error("Incorrect predict output based on weights of net")
		}
	}
}
