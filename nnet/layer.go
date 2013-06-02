package nnet

type NeuronLayer struct {
	NumNeurons int
	Neurons    []*Neuron
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

func (nl *NeuronLayer) Outputs() (outputs []float64) {
	for _, neuron := range nl.Neurons {
		outputs = append(outputs, neuron.Output)
	}
	return
}
