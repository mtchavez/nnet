package main

import (
	"github.com/mtchavez/nnet/nnet"
	"log"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func main() {
	nn := &nnet.NeuralNet{
		NumInputs:             nnet.TotalInputs,
		NumOutputs:            nnet.TotalOutputs,
		NumHiddenLayers:       nnet.TotalHiddenLayers,
		NeuronsPerHiddenLayer: nnet.TotalNeuronsPerHiddenLayer,
		Layers:                make([]*nnet.NeuronLayer, 0),
	}
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
