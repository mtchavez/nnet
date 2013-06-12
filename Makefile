ROOT := $(CURDIR)

test:
	cd ${ROOT}/nnet && go test
