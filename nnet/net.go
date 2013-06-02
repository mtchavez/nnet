package nnet

const (
	TotalInputs                = 2
	TotalHiddenLayers          = 2
	TotalNeuronsPerHiddenLayer = 2
	TotalOutputs               = 2
	ActivationResponse         = 1
	// Bias                       = -1
	LearningRate         = float64(1e-4)
	ConvergenceThreshold = float64(1e-10)
)

type NeuralNet struct {
	NumInputs             int
	NumOutputs            int
	NumHiddenLayers       int
	NeuronsPerHiddenLayer int
	Layers                []*NeuronLayer
	guess                 []float64
}

func NewNeuralNetWithDefaults() *NeuralNet {
	return &NeuralNet{
		NumInputs:             TotalInputs,
		NumOutputs:            TotalOutputs,
		NumHiddenLayers:       TotalHiddenLayers,
		NeuronsPerHiddenLayer: TotalNeuronsPerHiddenLayer,
		Layers:                make([]*NeuronLayer, 0),
	}
}

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
	for _, layer := range nn.Layers[1:] {
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
			total += len(neuron.Weights)
		}
	}
	return
}

func (nn *NeuralNet) setWeights(weights []float64) {
	// Start at 1 because input layer has no weights
	for _, layer := range nn.Layers[1:] {
		for _, neuron := range layer.Neurons {
			weightLength := len(neuron.Weights)
			toAdd := weights[:weightLength]
			weights = weights[weightLength:]
			neuron.Weights = toAdd //append(toAdd, Bias)
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

func (nn *NeuralNet) ClearDeltas() {
	for _, layer := range nn.Layers {
		for _, neuron := range layer.Neurons {
			neuron.delta = 0.0
		}
	}
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
