package nnet

import (
	"fmt"
)

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
