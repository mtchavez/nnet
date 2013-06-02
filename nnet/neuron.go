package nnet

import (
	"math"
	"math/rand"
)

type Neuron struct {
	TotalInputs int
	Inputs      []float64
	Weights     []float64
	Output      float64
	delta       float64
}

func (n *Neuron) SetupNeuron(inputs int) {
	n.TotalInputs = inputs
	for i := 0; i < inputs; i++ {
		n.Weights = append(n.Weights, rand.Float64())
	}
}

func (n *Neuron) CalcOutput() {
	sum := 0.0
	for i, input := range n.Inputs {
		sum += input * n.Weights[i]
	}
	n.Output = Sigmoid(sum)
}

func Sigmoid(sum float64) float64 {
	return (1 / (1 + math.Exp(-sum)))
}
