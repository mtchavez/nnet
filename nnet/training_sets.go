package nnet

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

var TrainingSet2 = [][]float64{
	{0, 0, 1, 1},
	{0, 1, 1, 0},
	{1, 1, 0, 0},
	{-1, 0, 1, 1},
	{0, -1, 1, 1},
	{-0.5, -0.5, 0.5, 0.5},
	{0.3, 0.3, -0.3, -0.3},
	{0.25, -0.25, -0.25, 0.25},
}
