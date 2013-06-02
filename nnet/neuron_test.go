package nnet

import (
	"testing"
)

func TestSetupNeuron(t *testing.T) {
	n := &Neuron{}
	inputs := 4
	n.SetupNeuron(inputs)
	if len(n.Weights) != inputs {
		t.Error("Number of inputs does not match number of weights")
	}

	if n.TotalInputs != inputs {
		t.Error("Total inputs on neuron should be set to inputs")
	}
}

func TestCalcOutput(t *testing.T) {
	n := &Neuron{}
	n.SetupNeuron(4)
	n.Weights = []float64{0.5, 1.0, -1.0, -0.5}
	n.Inputs = []float64{0.0, 0.0, 0.0, 0.0}
	n.CalcOutput()

	if n.Output != 0.5 {
		t.Error("If all inputs are zero the output of neuron should be 0.5 due to Sigmoid")
	}

	n.Inputs = []float64{-5.0, -1.0, 0.0, 8.0}
	// Sum of inputs*weights is -7.5
	n.CalcOutput()
	expected := 0.0005527786369235996
	if n.Output != expected {
		t.Errorf("Expected output for neuron is %f\n", expected)
	}
}

func TestSigmoid(t *testing.T) {
	fx := Sigmoid(0.0)
	expected := 0.5
	if fx != expected {
		t.Errorf("Sigmoid of 0.0 should be %f but got %f\n", expected, fx)
	}
	fx = Sigmoid(10.0)
	expected = 0.9999546021312976
	if fx != expected {
		t.Errorf("Sigmoid of 10.0 should be %f but got %f\n", expected, fx)
	}
}
