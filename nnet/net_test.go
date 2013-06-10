package nnet

import (
	"fmt"
	"testing"
)

func TestSigmoid(t *testing.T) {
	result := Sigmoid(1.0)
	expected := 0.7310585786300049
	if result != expected {
		t.Errorf("Expected Sigmoid(1.0) to equal %f but got %f", expected, result)
	}

	result = Sigmoid(0.0)
	expected = 0.5
	if result != expected {
		t.Errorf("Expected Sigmoid(0.0) to equal %f but got %f", expected, result)
	}

	result = Sigmoid(-1.0)
	expected = 0.2689414213699951
	if result != expected {
		t.Errorf("Expected Sigmoid(-1.0) to equal %f but got %f", expected, result)
	}
}

func TestDSigmoid(t *testing.T) {
	result := DSigmoid(1.0)
	expected := 0.0
	if result != expected {
		t.Errorf("Expected DSigmoid(1.0) to equal %f but got %f", expected, result)
	}

	result = DSigmoid(0.0)
	expected = 0.0
	if result != expected {
		t.Errorf("Expected DSigmoid(0.0) to equal %f but got %f", expected, result)
	}

	result = DSigmoid(-1.0)
	expected = -2.0
	if result != expected {
		t.Errorf("Expected DSigmoid(-1.0) to equal %f but got %f", expected, result)
	}
}

func TestSetupNeuralNetTotals(t *testing.T) {
	nn := &NeuralNet{}
	totalInputs := 4
	totalHidden := 3
	totalOutputs := 2
	nn.SetupNeuralNet(totalInputs, totalHidden, totalOutputs)

	if nn.totalInputs != totalInputs+1 {
		t.Error("Total inputs not set correctly")
	}

	if nn.totalHidden != totalHidden {
		t.Error("Total hidden nodes not set correctly")
	}

	if nn.totalOutputs != totalOutputs {
		t.Error("Total outputs not set correctly")
	}
}

func TestSetupNeuralNetActivations(t *testing.T) {
	nn := &NeuralNet{}
	totalInputs := 4
	totalHidden := 3
	totalOutputs := 2
	nn.SetupNeuralNet(totalInputs, totalHidden, totalOutputs)

	if len(nn.inputActivations) != nn.totalInputs {
		t.Error("Input activations don't match total inputs")
	}

	if len(nn.hiddenActivations) != nn.totalHidden {
		t.Error("Hidden activations don't match total hidden nodes")
	}

	if len(nn.outputActivations) != nn.totalOutputs {
		t.Error("Output activations don't match total outputs")
	}
}

func TestSetupNeuralNetWeights(t *testing.T) {
	nn := &NeuralNet{}
	totalInputs := 4
	totalHidden := 3
	totalOutputs := 2
	nn.SetupNeuralNet(totalInputs, totalHidden, totalOutputs)

	if len(nn.inputWeights) != nn.totalInputs {
		t.Error("Input weights don't match total inputs")
	}

	for i, _ := range nn.inputWeights {
		if len(nn.inputWeights[i]) != nn.totalHidden {
			t.Error("Hidden weights don't match total inputs")
		}
	}

	if len(nn.outputWeights) != nn.totalHidden {
		t.Error("Output weights don't match total hidden")
	}

	for i, _ := range nn.outputWeights {
		if len(nn.outputWeights[i]) != nn.totalOutputs {
			t.Error("Output weights don't match total outputs")
		}
	}
}

func ExampleTrain() {
	nn := &NeuralNet{}
	nn.SetupNeuralNet(3, 5, 1)
	nn.Train(TrainingSet2)

	for _, ex := range TrainingSet2 {
		input := ex[:3]
		expected := ex[3:]
		output := nn.Predict(input)
		fmt.Printf("For %+v neural net predicts %+v and we expect %+v\n", input, output, expected)
	}
	// Output:
	// For [1 0 0] neural net predicts [0.9999971598586201] and we expect [1]
	// For [1 0 1] neural net predicts [0.9993285002784352] and we expect [1]
	// For [1 1 0] neural net predicts [0.9992881939774748] and we expect [1]
	// For [1 1 1] neural net predicts [0.0010209810300121532] and we expect [0]
}
