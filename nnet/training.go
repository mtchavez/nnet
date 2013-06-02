package nnet

import (
	// "log"
	"math"
)

func (nn *NeuralNet) Train() {
	for _, ex := range TrainingSet2 {
		// input := make([]float64, 0)
		// for _, val := range ex[:len(ex)-1] {
		//  // input = append(input, Lookup[val])
		//  input = append(input, val)
		// }
		// answer := Lookup[ex[len(ex)-1]]
		// answer := ex[len(ex)-1]
		// expected := []float64{answer, answer, answer, answer}
		exMid := len(ex) / 2
		input := ex[:exMid]
		expected := ex[exMid:]
		nn.guess = expected
		newWeights := nn.GradientDescent(input, expected)
		nn.setWeights(newWeights)
	}
}

func (nn *NeuralNet) f(weights, inputs []float64) (sum float64) {
	nn.setWeights(weights)
	// log.Println("nn.guess = ", nn.guess)
	nn.BackPropagate(inputs, nn.guess)
	// log.Println("Outputs from f()", nn.Outputs())
	outputLyr := nn.Layers[len(nn.Layers)-1]
	for _, neuron := range outputLyr.Neurons {
		// log.Println("Delta ->", neuron.delta)
		sum += math.Pow(neuron.delta, 2.0)
	}
	return
}

func (nn *NeuralNet) diffofF(inputs []float64, expectedOutputs []float64) (gradient []float64) {
	nn.BackPropagate(inputs, expectedOutputs)
	// for all X->Y in the network:
	for i, layer := range nn.Layers[1:] {
		prevLyr := nn.Layers[i]
		for _, currNeuron := range layer.Neurons {
			for _, prevNeuron := range prevLyr.Neurons {
				gradient = append(gradient, prevNeuron.Output*currNeuron.delta)
			}
		}
	}
	return
}

func (nn *NeuralNet) isConverged(inputWeights, newWeights, inputs []float64) bool {
	// log.Println("inputWeights:", inputWeights)
	// log.Println("newWeights:", newWeights)
	// log.Println("f(inputWeights) = ", nn.f(inputWeights, inputs))
	// log.Println("f(newWeights) = ", nn.f(newWeights, inputs))
	// log.Println("Weight Diff For Convergence", math.Abs(nn.f(inputWeights, inputs)-nn.f(newWeights, inputs)))
	return math.Abs(nn.f(inputWeights, inputs)-nn.f(newWeights, inputs)) < ConvergenceThreshold
}

func (nn *NeuralNet) calcWeights(inputWeights []float64, gradients []float64) (scaled []float64) {
	for i, gradient := range gradients {
		scaled = append(scaled, gradient*LearningRate+inputWeights[i])
	}
	// log.Println("scaled weights", scaled)
	return
}

func (nn *NeuralNet) GradientDescent(inputs []float64, expectedOutputs []float64) (newWeights []float64) {
	inputWeights := nn.GetWeights()
	newWeights = nn.calcWeights(inputWeights, nn.diffofF(inputs, expectedOutputs))
	for {
		if nn.isConverged(inputWeights, newWeights, inputs) {
			break
		}
		// log.Println("Not Converged Yet")
		inputWeights = nn.GetWeights()
		newWeights = nn.calcWeights(inputWeights, nn.diffofF(inputs, expectedOutputs))
	}
	return
}

func (nn *NeuralNet) BackPropagate(inputs, expectedOutputs []float64) {
	nn.Predict(inputs)
	//PRECONDITION: neuron.delta = 0.0 for all neurons in the network
	nn.ClearDeltas()
	// log.Println("expected outputs", expectedOutputs)
	// loop over output neurons, and set their deltas
	outputLayer := nn.Layers[len(nn.Layers)-1]
	for i, outNeuron := range outputLayer.Neurons {
		outNeuron.delta = outNeuron.Output - expectedOutputs[i]

		if nn.NumHiddenLayers > 0 {
			// Set delta on previous hidden layer neurons
			firstHidden := nn.Layers[nn.NumHiddenLayers]
			for j, neuron := range firstHidden.Neurons {
				neuron.delta += outNeuron.Weights[j] * outNeuron.delta
			}
		}
	}

	if nn.NumHiddenLayers > 1 {
		// loop over hidden layers (from output to input)
		//   in each hidden layer, loop over neurons, and set the deltas of their input neurons
		// don't do it for the first hidden layer (index 0), because the inputLayer shouldn't have a delta (the delta of input layer doesn't affect training)
		for i := nn.NumHiddenLayers; i > 0; i-- {
			layer := nn.Layers[i]
			prevLyr := nn.Layers[i-1]
			for _, currNeuron := range layer.Neurons { // <<<<< neuron.delta is set. you need to set the deltas of the inputs to
				for j, prevNeuron := range prevLyr.Neurons {
					prevNeuron.delta += currNeuron.Weights[j] * currNeuron.delta
				}
			}
		}
	}
}
