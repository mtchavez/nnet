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
	NumInputs int
	Weights   []float64
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

func (sn *Neuron) SetupNeuron(numInputs int) {
	sn.NumInputs = numInputs
	for i := 0; i < numInputs+1; i++ {
		sn.Weights = append(sn.Weights, rand.Float64())
	}
}

func (nl *NeuronLayer) SetupNeuronLayer(numNeurons, numInputs int) {
	nl.NumNeurons = numNeurons
	for i := 0; i < numInputs; i++ {
		sn := &Neuron{}
		sn.SetupNeuron(numInputs)
		nl.Neurons = append(nl.Neurons, sn)
	}
}

//
//  Neural Net
//
func (nn *NeuralNet) SetupNeuralNet() {
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
		nn.Layers = append([]*NeuronLayer{nl}, nn.Layers...)
	}

	// Create output layer
	output := &NeuronLayer{}
	output.SetupNeuronLayer(nn.NumOutputs, nn.NeuronsPerHiddenLayer)
	nn.Layers = append(nn.Layers, output)
}

func (nn *NeuralNet) GetWeights() (weights []float64) {
	// Each layer
	for i := 0; i < nn.NumHiddenLayers; i++ {
		layer := nn.Layers[i]
		// Each neuron
		for j := 0; j < layer.NumNeurons; j++ {
			neuron := layer.Neurons[j]
			for k := 0; k < neuron.NumInputs; k++ {
				weight := neuron.Weights[k]
				weights = append([]float64{weight}, weights...)
			}
		}
	}
	return
}

func (nn *NeuralNet) TotalWeights() (total int) {
	for i := 0; i < nn.NumHiddenLayers; i++ {
		layer := nn.Layers[i]
		for j := 0; j < layer.NumNeurons; j++ {
			neuron := layer.Neurons[j]
			for k := 0; k < neuron.NumInputs; k++ {
				total++
			}
		}
	}
	return
}

func (nn *NeuralNet) Update(inputs []float64) (outputs []float64) {
	cWeight := 0
	if len(inputs) != nn.NumInputs {
		return
	}

	for i := 0; i < len(nn.Layers); i++ {
		if i > 0 {
			inputs = outputs
		}
		outputs = make([]float64, 0)
		cWeight = 0
		layer := nn.Layers[i]
		for j := 0; j < layer.NumNeurons; j++ {
			var netinput float64 = 0.0
			neuron := layer.Neurons[j]
			numInputs := neuron.NumInputs

			for k := 0; k < numInputs; k++ {
				w := 0.0
				if cWeight < len(inputs) {
					w = inputs[cWeight]
				}
				netinput += neuron.Weights[k] * w
				cWeight++
			}

			// Add Bias
			netinput += neuron.Weights[numInputs-1] * Bias

			outputs = append([]float64{Sigmoid(netinput, ActivationResponse)}, outputs...)
			cWeight = 0
		}
	}
	return
}

func Sigmoid(netinput, response float64) float64 {
	return (1 / (1 + math.Exp(-netinput/response)))
}

func (nn *NeuralNet) Train() {
	for {
		for _, ex := range TrainingSet {
			input := make([]float64, 0)
			for _, val := range ex[:len(ex)-1] {
				input = append(input, Lookup[val])
			}
			predicted := nn.Update(input)[0] > 0.5
			expected := ex[len(ex)-1]
			log.Println("INPUT: ", input)
			log.Println("Predicted: ", predicted)
			log.Println("Expected: ", expected)
			// compute_error := 1
			// if predicted && expected {
			// 	compute_error = 0
			// }

		}
		break
	}
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func main() {
	nn := &NeuralNet{TotalInputs, TotalOutputs, TotalHiddenLayers, TotalNeuronsPerHiddenLayer, make([]*NeuronLayer, 0)}
	nn.SetupNeuralNet()
	// log.Printf("Neural Net Layers = ", len(nn.Layers))
	// log.Printf("Neural Net:\n\n%+v", nn)

	log.Printf("Neural Net Total Weights: %+v\n", nn.TotalWeights())
	// log.Printf("Neural Net Weights: \n\n%+v", nn.GetWeights())

	// inputs := []float64{0.0, 0.0, 0.0, 1.0}
	// log.Println("Updating with ", inputs)
	// log.Printf("%+v\n", nn.Update(inputs))

	nn.Train()
}
