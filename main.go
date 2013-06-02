package main

import (
	"github.com/mtchavez/nnet/nnet"
	"log"
	"math/rand"
	"time"
)

// const (
// 	TotalInputs                = 2
// 	TotalHiddenLayers          = 2
// 	TotalNeuronsPerHiddenLayer = 2
// 	TotalOutputs               = 2
// 	ActivationResponse         = 1
// 	// Bias                       = -1
// 	LearningRate         = float64(1e-4)
// 	ConvergenceThreshold = float64(1e-10)
// )

// type Neuron struct {
// 	TotalInputs int
// 	Inputs      []float64
// 	Weights     []float64
// 	Output      float64
// 	delta       float64
// }

// type NeuronLayer struct {
// 	NumNeurons int
// 	Neurons    []*Neuron
// }

// type NeuralNet struct {
// 	NumInputs             int
// 	NumOutputs            int
// 	NumHiddenLayers       int
// 	NeuronsPerHiddenLayer int
// 	Layers                []*NeuronLayer
// 	guess                 []float64
// }

// var Lookup = map[string]float64{
// 	"false":       0.0,
// 	"true":        1.0,
// 	"< 18":        -1.0,
// 	"18 - 35":     -0.75,
// 	"36 - 55":     -0.50,
// 	"> 55":        -0.25,
// 	"high school": 0.25,
// 	"bachelors":   0.50,
// 	"masters":     0.75,
// 	"high":        0.4,
// 	"low":         0.3,
// 	"single":      0.8,
// 	"married":     0.9,
// }

// var TrainingSet = [][]string{
// 	{"36 - 55", "masters", "high", "single", "true"},
// 	{"18 - 35", "high school", "low", "single", "false"},
// 	{"36 - 55", "masters", "low", "single", "true"},
// 	{"18 - 35", "bachelors", "high", "single", "false"},
// 	{"< 18", "high school", "low", "single", "true"},
// 	{"18 - 35", "bachelors", "high", "married", "false"},
// 	{"36 - 55", "bachelors", "low", "married", "false"},
// 	{"> 55", "bachelors", "high", "single", "true"},
// 	{"36 - 55", "masters", "low", "married", "false"},
// 	{"> 55", "masters", "low", "married", "true"},
// 	{"36 - 55", "masters", "high", "single", "true"},
// 	{"> 55", "masters", "high", "single", "true"},
// 	{"< 18", "high school", "high", "single", "false"},
// 	{"36 - 55", "masters", "low", "single", "true"},
// 	{"36 - 55", "high school", "low", "single", "true"},
// 	{"< 18", "high school", "low", "married", "true"},
// 	{"18 - 35", "bachelors", "high", "married", "false"},
// 	{"> 55", "high school", "high", "married", "true"},
// 	{"> 55", "bachelors", "low", "single", "true"},
// 	{"36 - 55", "high school", "high", "married", "false"},
// }

// var TrainingSet2 = [][]float64{
// 	{0, 0, 1, 1},
// 	{0, 1, 1, 0},
// 	{1, 1, 0, 0},
// 	{-1, 0, 1, 1},
// 	{0, -1, 1, 1},
// 	{-0.5, -0.5, 0.5, 0.5},
// 	{0.3, 0.3, -0.3, -0.3},
// 	{0.25, -0.25, -0.25, 0.25},
// }

// func (n *Neuron) SetupNeuron(inputs int) {
// 	n.TotalInputs = inputs
// 	// Add 1 weight to carry bias
// 	for i := 0; i < inputs; i++ {
// 		n.Weights = append(n.Weights, rand.Float64())
// 	}
// }

// func (nl *NeuronLayer) SetupInputLayer(numNeurons, numInputs int) {
// 	nl.NumNeurons = numNeurons
// 	for i := 0; i < numInputs; i++ {
// 		n := &Neuron{}
// 		n.TotalInputs = numInputs
// 		nl.Neurons = append(nl.Neurons, n)
// 	}
// }

// func (nl *NeuronLayer) SetupNeuronLayer(numNeurons, numInputs int) {
// 	nl.NumNeurons = numNeurons
// 	for i := 0; i < numInputs; i++ {
// 		n := &Neuron{}
// 		n.SetupNeuron(numInputs)
// 		nl.Neurons = append(nl.Neurons, n)
// 	}
// }

//
//  Neural Net
//
// func (nn *NeuralNet) SetupNeuralNet() {
// 	// Create input layer
// 	nl := &NeuronLayer{}
// 	nl.SetupInputLayer(nn.NumOutputs, nn.NumInputs)
// 	nn.Layers = append(nn.Layers, nl)

// 	if nn.NumHiddenLayers <= 0 {
// 		// Create output layer
// 		nl := &NeuronLayer{}
// 		nl.SetupNeuronLayer(nn.NumOutputs, nn.NumInputs)
// 		nn.Layers = append(nn.Layers, nl)
// 		return
// 	}

// 	for i := 0; i < nn.NumHiddenLayers; i++ {
// 		nl := &NeuronLayer{}
// 		nl.SetupNeuronLayer(nn.NeuronsPerHiddenLayer, nn.NeuronsPerHiddenLayer)
// 		nn.Layers = append(nn.Layers, nl)
// 	}

// 	// Create output layer
// 	output := &NeuronLayer{}
// 	output.SetupNeuronLayer(nn.NumOutputs, nn.NeuronsPerHiddenLayer)
// 	nn.Layers = append(nn.Layers, output)
// }

// func (nn *NeuralNet) GetWeights() (weights []float64) {
// 	// Start at 1 because input layer has no weights
// 	for _, layer := range nn.Layers[1:] {
// 		for _, neuron := range layer.Neurons {
// 			weights = append(weights, neuron.Weights...)
// 		}
// 	}
// 	return
// }

// func (nn *NeuralNet) TotalWeights() (total int) {
// 	// Start at 1 because input layer has no weights
// 	for _, layer := range nn.Layers[1:] {
// 		for _, neuron := range layer.Neurons {
// 			total += len(neuron.Weights)
// 		}
// 	}
// 	return
// }

// func (nn *NeuralNet) setWeights(weights []float64) {
// 	// Start at 1 because input layer has no weights
// 	for _, layer := range nn.Layers[1:] {
// 		for _, neuron := range layer.Neurons {
// 			weightLength := len(neuron.Weights)
// 			toAdd := weights[:weightLength]
// 			weights = weights[weightLength:]
// 			neuron.Weights = toAdd //append(toAdd, Bias)
// 		}
// 	}
// }

// func Sigmoid(sum float64) float64 {
// 	return (1 / (1 + math.Exp(-sum)))
// }

// func (n *Neuron) CalcOutput() {
// 	sum := 0.0
// 	for i, input := range n.Inputs {
// 		sum += input * n.Weights[i]
// 	}

// 	// Add bias
// 	// sum += n.Weights[n.TotalInputs] * Bias
// 	n.Output = Sigmoid(sum)
// }

// func (nl *NeuronLayer) Outputs() (outputs []float64) {
// 	for _, neuron := range nl.Neurons {
// 		outputs = append(outputs, neuron.Output)
// 	}
// 	return
// }

// func (nn *NeuralNet) Predict(inputs []float64) {
// 	if len(inputs) != nn.NumInputs {
// 		return
// 	}

// 	// Set inputs/outputs of input layer
// 	inputLayer := nn.Layers[0]
// 	for i, neuron := range inputLayer.Neurons {
// 		neuron.Output = inputs[i]
// 	}

// 	for i, layer := range nn.Layers[1:] {
// 		prevLyr := nn.Layers[i]
// 		for _, neuron := range layer.Neurons {
// 			neuron.Inputs = prevLyr.Outputs()
// 			neuron.CalcOutput()
// 		}
// 	}
// }

// func (nn *NeuralNet) Outputs() (outputs []float64) {
// 	outputLyr := nn.Layers[len(nn.Layers)-1]
// 	for _, neuron := range outputLyr.Neurons {
// 		outputs = append(outputs, neuron.Output)
// 	}
// 	return
// }

// func (nn *NeuralNet) ClearDeltas() {
// 	for _, layer := range nn.Layers {
// 		for _, neuron := range layer.Neurons {
// 			neuron.delta = 0.0
// 		}
// 	}
// }

// func (nn *NeuralNet) Train() {
// 	for _, ex := range TrainingSet2 {
// 		// input := make([]float64, 0)
// 		// for _, val := range ex[:len(ex)-1] {
// 		// 	// input = append(input, Lookup[val])
// 		// 	input = append(input, val)
// 		// }
// 		// answer := Lookup[ex[len(ex)-1]]
// 		// answer := ex[len(ex)-1]
// 		// expected := []float64{answer, answer, answer, answer}
// 		exMid := len(ex) / 2
// 		input := ex[:exMid]
// 		expected := ex[exMid:]
// 		nn.guess = expected
// 		newWeights := nn.GradientDescent(input, expected)
// 		nn.setWeights(newWeights)
// 	}
// }

// func (nn *NeuralNet) f(weights, inputs []float64) (sum float64) {
// 	nn.setWeights(weights)
// 	// log.Println("nn.guess = ", nn.guess)
// 	nn.BackPropagate(inputs, nn.guess)
// 	// log.Println("Outputs from f()", nn.Outputs())
// 	outputLyr := nn.Layers[len(nn.Layers)-1]
// 	for _, neuron := range outputLyr.Neurons {
// 		// log.Println("Delta ->", neuron.delta)
// 		sum += math.Pow(neuron.delta, 2.0)
// 	}
// 	return
// }

// func (nn *NeuralNet) diffofF(inputs []float64, expectedOutputs []float64) (gradient []float64) {
// 	nn.BackPropagate(inputs, expectedOutputs)
// 	// for all X->Y in the network:
// 	for i, layer := range nn.Layers[1:] {
// 		prevLyr := nn.Layers[i]
// 		for _, currNeuron := range layer.Neurons {
// 			for _, prevNeuron := range prevLyr.Neurons {
// 				gradient = append(gradient, prevNeuron.Output*currNeuron.delta)
// 			}
// 		}
// 	}
// 	return
// }

// func (nn *NeuralNet) isConverged(inputWeights, newWeights, inputs []float64) bool {
// 	// log.Println("inputWeights:", inputWeights)
// 	// log.Println("newWeights:", newWeights)
// 	// log.Println("f(inputWeights) = ", nn.f(inputWeights, inputs))
// 	// log.Println("f(newWeights) = ", nn.f(newWeights, inputs))
// 	// log.Println("Weight Diff For Convergence", math.Abs(nn.f(inputWeights, inputs)-nn.f(newWeights, inputs)))
// 	return math.Abs(nn.f(inputWeights, inputs)-nn.f(newWeights, inputs)) < ConvergenceThreshold
// }

// func (nn *NeuralNet) calcWeights(inputWeights []float64, gradients []float64) (scaled []float64) {
// 	for i, gradient := range gradients {
// 		scaled = append(scaled, gradient*LearningRate+inputWeights[i])
// 	}
// 	// log.Println("scaled weights", scaled)
// 	return
// }

// func (nn *NeuralNet) GradientDescent(inputs []float64, expectedOutputs []float64) (newWeights []float64) {
// 	inputWeights := nn.GetWeights()
// 	newWeights = nn.calcWeights(inputWeights, nn.diffofF(inputs, expectedOutputs))
// 	for {
// 		if nn.isConverged(inputWeights, newWeights, inputs) {
// 			break
// 		}
// 		// log.Println("Not Converged Yet")
// 		inputWeights = nn.GetWeights()
// 		newWeights = nn.calcWeights(inputWeights, nn.diffofF(inputs, expectedOutputs))
// 	}
// 	return
// }

// func (nn *NeuralNet) BackPropagate(inputs, expectedOutputs []float64) {
// 	nn.Predict(inputs)
// 	//PRECONDITION: neuron.delta = 0.0 for all neurons in the network
// 	nn.ClearDeltas()
// 	// log.Println("expected outputs", expectedOutputs)
// 	// loop over output neurons, and set their deltas
// 	outputLayer := nn.Layers[len(nn.Layers)-1]
// 	for i, outNeuron := range outputLayer.Neurons {
// 		outNeuron.delta = outNeuron.Output - expectedOutputs[i]

// 		if nn.NumHiddenLayers > 0 {
// 			// Set delta on previous hidden layer neurons
// 			firstHidden := nn.Layers[nn.NumHiddenLayers]
// 			for j, neuron := range firstHidden.Neurons {
// 				neuron.delta += outNeuron.Weights[j] * outNeuron.delta
// 			}
// 		}
// 	}

// 	if nn.NumHiddenLayers > 1 {
// 		// loop over hidden layers (from output to input)
// 		//   in each hidden layer, loop over neurons, and set the deltas of their input neurons
// 		// don't do it for the first hidden layer (index 0), because the inputLayer shouldn't have a delta (the delta of input layer doesn't affect training)
// 		for i := nn.NumHiddenLayers; i > 0; i-- {
// 			layer := nn.Layers[i]
// 			prevLyr := nn.Layers[i-1]
// 			for _, currNeuron := range layer.Neurons { // <<<<< neuron.delta is set. you need to set the deltas of the inputs to
// 				for j, prevNeuron := range prevLyr.Neurons {
// 					prevNeuron.delta += currNeuron.Weights[j] * currNeuron.delta
// 				}
// 			}
// 		}
// 	}
// }

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func main() {
	nn := &nnet.NeuralNet{TotalInputs, TotalOutputs, TotalHiddenLayers, TotalNeuronsPerHiddenLayer, make([]*NeuronLayer, 0), make([]float64, 0)}
	nn.SetupNeuralNet()
	log.Printf("Neural Net Layers: %d\n", len(nn.Layers))
	// log.Printf("Neural Net:\n\n%+v", nn)

	log.Printf("Neural Net Total Weights: %+v\n", nn.TotalWeights())
	// log.Printf("Neural Net Weights: \n\n%+v", nn.GetWeights())

	log.Printf("Initial Outputs: %+v\n", nn.Outputs())
	inputs := []float64{0.0, -0.32456}
	log.Println("Predicting with ", inputs)
	nn.Predict(inputs)
	log.Printf("New Outputs: %+v\n", nn.Outputs())

	log.Println("Weights Before Train")
	log.Println(nn.GetWeights())
	log.Println("Train")
	nn.Train()
	log.Println("Finished Training")
	log.Println("Weights After Train")
	log.Println(nn.GetWeights())
	log.Println("Predicting again with ", inputs)
	nn.Predict(inputs)
	log.Printf("New Outputs: %+v\n", nn.Outputs())
}
