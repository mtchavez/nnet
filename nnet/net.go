package nnet

import (
	"log"
	"math"
	"math/rand"
)

const (
	TotalInputs                = 3
	TotalHiddenLayers          = 2
	TotalNeuronsPerHiddenLayer = 2
	TotalOutputs               = 1
	ActivationResponse         = 1
	Bias                       = -1
	LearningRate               = float64(0.5)
	ConvergenceThreshold       = float64(1e-6)
)

type NeuralNet struct {
	inputWeights      [][]float64
	outputWeights     [][]float64
	inputChanges      [][]float64
	outputChanges     [][]float64
	totalInputs       int
	totalHidden       int
	totalOutputs      int
	inputActivations  []float64
	hiddenActivations []float64
	outputActivations []float64
}

func Sigmoid(sum float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -float64(sum)))
}

func DSigmoid(y float64) float64 {
	return y * (1.0 - y)
}

func (nn *NeuralNet) SetupNeuralNet(totalInputs, totalHidden, totalOutputs int) {
	totalInputs++
	nn.totalInputs = totalInputs
	nn.totalHidden = totalHidden
	nn.totalOutputs = totalOutputs
	nn.inputActivations = make([]float64, totalInputs)
	nn.hiddenActivations = make([]float64, totalHidden)
	nn.outputActivations = make([]float64, totalOutputs)

	nn.inputWeights = make([][]float64, totalInputs)
	for i, _ := range nn.inputWeights {
		nn.inputWeights[i] = make([]float64, totalHidden)
		for j := 0; j < totalHidden; j++ {
			nn.inputWeights[i][j] = rand.Float64()
		}
	}
	nn.outputWeights = make([][]float64, totalHidden)
	for i, _ := range nn.outputWeights {
		nn.outputWeights[i] = make([]float64, totalOutputs)
		for j := 0; j < totalOutputs; j++ {
			nn.outputWeights[i][j] = rand.Float64()
		}
	}

	nn.inputChanges = make([][]float64, totalInputs)
	for i, _ := range nn.inputWeights {
		nn.inputChanges[i] = make([]float64, totalHidden)
	}

	nn.outputChanges = make([][]float64, totalHidden)
	for i, _ := range nn.outputWeights {
		nn.outputChanges[i] = make([]float64, totalOutputs)
	}
}

func (nn *NeuralNet) Train(examples [][]float64) {
	i := 0
	for {
		i++
		err := 0.0
		for _, example := range examples {
			inputs := example[:nn.totalInputs-1]
			expected := example[nn.totalInputs-1:]
			nn.Predict(inputs)
			err += nn.BackPropLearn(expected)
		}
		if i%1000000 == 0 {
			log.Println("Err: ", err)
		}
		if err <= ConvergenceThreshold {
			break
		}
	}
}

func (nn *NeuralNet) BackPropLearn(expected []float64) (totalErr float64) {
	outputDeltas := make([]float64, nn.totalOutputs)
	for j := 0; j < nn.totalOutputs; j++ {
		actual := nn.outputActivations[j]
		err := expected[j] - actual
		outputDeltas[j] = err * DSigmoid(actual)
	}

	for j := 0; j < nn.totalHidden; j++ {
		for k := 0; k < nn.totalOutputs; k++ {
			change := outputDeltas[k] * nn.hiddenActivations[j]
			nn.outputWeights[j][k] += LearningRate*change + 0.1*nn.outputChanges[j][k]
			nn.outputChanges[j][k] = change
		}
	}

	hiddenDeltas := make([]float64, nn.totalHidden)
	for j := 0; j < nn.totalHidden; j++ {
		err := 0.0
		for k := 0; k < nn.totalOutputs; k++ {
			err += outputDeltas[k] * nn.outputWeights[j][k]
		}
		hiddenDeltas[j] = err * DSigmoid(nn.hiddenActivations[j])
	}

	for i := 0; i < nn.totalInputs; i++ {
		for j := 0; j < nn.totalHidden; j++ {
			change := hiddenDeltas[j] * nn.inputActivations[i]
			nn.inputWeights[i][j] += LearningRate*change + 0.1*nn.inputChanges[i][j]
			nn.inputChanges[i][j] = change
		}
	}

	for i, output := range expected {
		totalErr += 0.5 * math.Pow(output-nn.outputActivations[i], 2)
	}
	return
}

func (nn *NeuralNet) Predict(inputs []float64) []float64 {
	if len(inputs) != nn.totalInputs-1 {
		return []float64{}
	}

	for i, input := range inputs {
		nn.inputActivations[i] = input
	}

	for j := 0; j < nn.totalHidden; j++ {
		sum := 0.0
		for i := 0; i < nn.totalInputs; i++ {
			sum += nn.inputActivations[i] * nn.inputWeights[i][j]
		}
		nn.hiddenActivations[j] = Sigmoid(sum)
	}

	for j := 0; j < nn.totalOutputs; j++ {
		sum := 0.0
		for i := 0; i < nn.totalHidden; i++ {
			sum += nn.hiddenActivations[i] * nn.outputWeights[i][j]
		}
		nn.outputActivations[j] = Sigmoid(sum)
	}
	return nn.outputActivations
}
