nnet
====

Neural Net in Go

## Installation

Use ```go get``` to install the package.

```
go get -u github.com/mtchavez/nnet/nnet
```

## Usage

There are some training examples in ```./nnet/training_sets.go``` to use.

An example using ```TrainingSet2``` (NAND) would look like this:

```go
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
```

The set named ```TrainingSet``` will take a while to run but will finish eventually
and using the same script above should give you similar output:

```
2013/06/09 19:08:49 Err:  3.874987320725568e-05
2013/06/09 19:09:26 Err:  1.6621031811903178e-05
2013/06/09 19:10:02 Err:  1.0078854035870953e-05
...
2013/06/09 19:19:31 Err:  1.1254870026927736e-06
2013/06/09 19:20:06 Err:  1.0652633938055432e-06
2013/06/09 19:20:41 Err:  1.0111589283784186e-06
2013/06/09 19:20:48 For [-0.5 0.75 0.4 0.8] neural net predicts [0.999999715590078] and we expect [1]
2013/06/09 19:20:48 For [-0.75 0.25 0.3 0.8] neural net predicts [0.0006310811073551809] and we expect [0]
2013/06/09 19:20:48 For [-0.5 0.75 0.3 0.8] neural net predicts [0.9997530594821283] and we expect [1]
2013/06/09 19:20:48 For [-0.75 0.5 0.4 0.8] neural net predicts [7.72391221796645e-10] and we expect [0]
2013/06/09 19:20:48 For [-1 0.25 0.3 0.8] neural net predicts [0.9995155065107615] and we expect [1]
2013/06/09 19:20:48 For [-0.75 0.5 0.4 0.9] neural net predicts [2.110466190106359e-11] and we expect [0]
2013/06/09 19:20:48 For [-0.5 0.5 0.3 0.9] neural net predicts [0.00037037185571059725] and we expect [0]
2013/06/09 19:20:48 For [-0.25 0.5 0.4 0.8] neural net predicts [0.9998935743831123] and we expect [1]
2013/06/09 19:20:48 For [-0.5 0.75 0.3 0.9] neural net predicts [0.0003226877510435841] and we expect [0]
2013/06/09 19:20:48 For [-0.25 0.75 0.3 0.9] neural net predicts [0.9999999953574958] and we expect [1]
2013/06/09 19:20:48 For [-0.5 0.75 0.4 0.8] neural net predicts [0.999999715590078] and we expect [1]
2013/06/09 19:20:48 For [-0.25 0.75 0.4 0.8] neural net predicts [0.9999998573667516] and we expect [1]
2013/06/09 19:20:48 For [-1 0.25 0.4 0.8] neural net predicts [1.2870298645547013e-09] and we expect [0]
2013/06/09 19:20:48 For [-0.5 0.75 0.3 0.8] neural net predicts [0.9997530594821283] and we expect [1]
2013/06/09 19:20:48 For [-0.5 0.25 0.3 0.8] neural net predicts [0.9992302273315177] and we expect [1]
2013/06/09 19:20:48 For [-1 0.25 0.3 0.9] neural net predicts [0.9999997050099974] and we expect [1]
2013/06/09 19:20:48 For [-0.75 0.5 0.4 0.9] neural net predicts [2.110466190106359e-11] and we expect [0]
2013/06/09 19:20:48 For [-0.25 0.25 0.4 0.9] neural net predicts [0.9999999919841681] and we expect [1]
2013/06/09 19:20:48 For [-0.25 0.5 0.3 0.8] neural net predicts [0.9997764205767077] and we expect [1]
2013/06/09 19:20:48 For [-0.5 0.25 0.4 0.9] neural net predicts [0.0005914493975230156] and we expect [0]
```

Which shows the NeuralNet has learned the example training set of detemining some outcome given
basic information of on a group of people.

## Documentation

Docs are on [Godoc](http://godoc.org/github.com/mtchavez/nnet/nnet)

## Test

You can run the tests using ```go test```

## TODO

* Allow multiple hidden layers
* Write function to export weights of neural net
* Write function to import previously known weights
* Allow Activation functions to be set to anything

## License
Written by Chavez

Released under the MIT License: http://www.opensource.org/licenses/mit-license.php
