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
	nn := &nnet.NeuralNet{}
	nn.SetupNeuralNet(4, 5, 1)
	nn.Train(nnet.TrainingSet)

	for _, ex := range nnet.TrainingSet {
		input := ex[:4]
		expected := ex[4:]
		output := nn.Predict(input)
		log.Printf("For %+v neural net predicts %+v and we expect %+v\n", input, output, expected)
	}
}
