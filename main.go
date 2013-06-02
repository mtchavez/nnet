package main

import (
	"log"
	"math"
	"math/rand"
	"time"
)

const (
	TotalInputs                = 4
	TotalHiddenLayers          = 1
	TotalNeuronsPerHiddenLayer = 4
	TotalOutputs               = 1
	ActivationResponse         = 1
	Bias                       = -1
)

type Neuron struct {
	TotalInputs int
	Inputs      []float64
	Weights     []float64
	Output      float64
	delta       float64
}

type NeuronLayer struct {
	NumNeurons int
	Neurons    []*Neuron
}

type NeuralNet struct {
	NumInputs             int
	NumOutputs            int
	NumHiddenLayers       int
	NeuronsPerHiddenLayer int
	Layers                []*NeuronLayer
}

var Lookup = map[string]float64{
	"false":       0.0,
	"true":        1.0,
	"< 18":        -1.0,
	"18 - 35":     -0.75,
	"36 - 55":     -0.50,
	"> 55":        -0.25,
	"high school": 0.25,
	"bachelors":   0.50,
	"masters":     0.75,
	"high":        0.4,
	"low":         0.3,
	"single":      0.8,
	"married":     0.9,
}

var TrainingSet = [][]string{
	{"36 - 55", "masters", "high", "single", "true"},
	{"18 - 35", "high school", "low", "single", "false"},
	{"36 - 55", "masters", "low", "single", "true"},
	{"18 - 35", "bachelors", "high", "single", "false"},
	{"< 18", "high school", "low", "single", "true"},
	{"18 - 35", "bachelors", "high", "married", "false"},
	{"36 - 55", "bachelors", "low", "married", "false"},
	{"> 55", "bachelors", "high", "single", "true"},
	{"36 - 55", "masters", "low", "married", "false"},
	{"> 55", "masters", "low", "married", "true"},
	{"36 - 55", "masters", "high", "single", "true"},
	{"> 55", "masters", "high", "single", "true"},
	{"< 18", "high school", "high", "single", "false"},
	{"36 - 55", "masters", "low", "single", "true"},
	{"36 - 55", "high school", "low", "single", "true"},
	{"< 18", "high school", "low", "married", "true"},
	{"18 - 35", "bachelors", "high", "married", "false"},
	{"> 55", "high school", "high", "married", "true"},
	{"> 55", "bachelors", "low", "single", "true"},
	{"36 - 55", "high school", "high", "married", "false"},
}

func (n *Neuron) SetupNeuron(inputs int) {
	n.TotalInputs = inputs
	// Add 1 weight to carry bias
	for i := 0; i < inputs+1; i++ {
		n.Weights = append(n.Weights, rand.Float64()-0.5)
	}
}

func (nl *NeuronLayer) SetupInputLayer(numNeurons, numInputs int) {
	nl.NumNeurons = numNeurons
	for i := 0; i < numInputs; i++ {
		n := &Neuron{}
		n.TotalInputs = numInputs
		nl.Neurons = append(nl.Neurons, n)
	}
}

func (nl *NeuronLayer) SetupNeuronLayer(numNeurons, numInputs int) {
	nl.NumNeurons = numNeurons
	for i := 0; i < numInputs; i++ {
		n := &Neuron{}
		n.SetupNeuron(numInputs)
		nl.Neurons = append(nl.Neurons, n)
	}
}

//
//  Neural Net
//
func (nn *NeuralNet) SetupNeuralNet() {
	// Create input layer
	nl := &NeuronLayer{}
	nl.SetupInputLayer(nn.NumOutputs, nn.NumInputs)
	nn.Layers = append(nn.Layers, nl)

	if nn.NumHiddenLayers <= 0 {
		// Create output layer
		nl := &NeuronLayer{}
		nl.SetupNeuronLayer(nn.NumOutputs, nn.NumInputs)
		nn.Layers = append(nn.Layers, nl)
		return
	}

	for i := 0; i < nn.NumHiddenLayers; i++ {
		nl := &NeuronLayer{}
		nl.SetupNeuronLayer(nn.NeuronsPerHiddenLayer, nn.NeuronsPerHiddenLayer)
		nn.Layers = append(nn.Layers, nl)
	}

	// Create output layer
	output := &NeuronLayer{}
	output.SetupNeuronLayer(nn.NumOutputs, nn.NeuronsPerHiddenLayer)
	nn.Layers = append(nn.Layers, output)
}

func (nn *NeuralNet) GetWeights() (weights []float64) {
	// Start at 1 because input layer has no weights
	for i := 1; i < nn.NumHiddenLayers; i++ {
		layer := nn.Layers[i]
		for _, neuron := range layer.Neurons {
			weights = append(weights, neuron.Weights...)
		}
	}
	return
}

func (nn *NeuralNet) TotalWeights() (total int) {
	// Start at 1 because input layer has no weights
	for _, layer := range nn.Layers[1:] {
		for _, neuron := range layer.Neurons {
			total += len(neuron.Weights) - 1
		}
	}
	return
}

func Sigmoid(sum, response float64) float64 {
	return (1 / (1 + math.Exp(-sum/response)))
}

func (n *Neuron) CalcOutput() {
	sum := 0.0
	for i, input := range n.Inputs {
		sum += input * n.Weights[i]
	}

	// Add bias
	sum += n.Weights[n.TotalInputs] * Bias
	n.Output = Sigmoid(sum, ActivationResponse)
}

func (nl *NeuronLayer) Outputs() (outputs []float64) {
	for _, neuron := range nl.Neurons {
		outputs = append(outputs, neuron.Output)
	}
	return
}

func (nn *NeuralNet) Predict(inputs []float64) {
	if len(inputs) != nn.NumInputs {
		return
	}

	// Set inputs/outputs of input layer
	inputLayer := nn.Layers[0]
	for i, neuron := range inputLayer.Neurons {
		neuron.Output = inputs[i]
	}

	for i, layer := range nn.Layers[1:] {
		prevLyr := nn.Layers[i]
		for _, neuron := range layer.Neurons {
			neuron.Inputs = prevLyr.Outputs()
			neuron.CalcOutput()
		}
	}
}

func (nn *NeuralNet) Outputs() (outputs []float64) {
	outputLyr := nn.Layers[len(nn.Layers)-1]
	for _, neuron := range outputLyr.Neurons {
		outputs = append(outputs, neuron.Output)
	}
	return
}

func (nn *NeuralNet) Train() {
	for _, ex := range TrainingSet {
		input := make([]float64, 0)
		for _, val := range ex[:len(ex)-1] {
			input = append(input, Lookup[val])
		}
		nn.Predict(input)
		sum := 0.0
		for _, num := range nn.Outputs() {
			sum += num * num
		}
		out := Sigmoid(sum, ActivationResponse)

		expected := ex[len(ex)-1]
		log.Println("INPUT: ", input)
		log.Println("Predicted: ", out > 0.5)
		log.Println("Expected: ", expected)
		// compute_error := 1
		// if predicted && expected {
		// 	compute_error = 0
		// }

	}
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func main() {
	nn := &NeuralNet{TotalInputs, TotalOutputs, TotalHiddenLayers, TotalNeuronsPerHiddenLayer, make([]*NeuronLayer, 0)}
	nn.SetupNeuralNet()
	log.Printf("Neural Net Layers: %d\n", len(nn.Layers))
	// log.Printf("Neural Net:\n\n%+v", nn)

	log.Printf("Neural Net Total Weights: %+v\n", nn.TotalWeights())
	// log.Printf("Neural Net Weights: \n\n%+v", nn.GetWeights())

	log.Printf("Initial Outputs: %+v\n", nn.Outputs())
	inputs := []float64{0.0, -1.0, 0.0, 1.0}
	log.Println("Updating with ", inputs)
	nn.Predict(inputs)
	log.Printf("New Outputs: %+v\n", nn.Outputs())
	sum := 0.0
	for _, num := range nn.Outputs() {
		sum += num * num
	}
	sum += Bias
	out := Sigmoid(sum, ActivationResponse)
	log.Println("Output through Sigmoid: ", out)

	log.Println("Train")
	nn.Train()
}
