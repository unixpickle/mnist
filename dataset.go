package mnist

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// A Sample is one instance of a handwritten digit.
type Sample struct {
	// Intensities is a bitmap of white-and-black
	// values, where 0xff is black and 0 is white.
	//
	// This is ordered first horizontally, then
	// vertically, like a traditional bitmap.
	Intensities []uint8

	// Label is a number between 0 and 9 (inclusive)
	// indicating what digit this is.
	Label int
}

// Floats returns a floating-point representation
// of the vector of intensities, where the max
// intensity is 1 and the min is 0.
func (s *Sample) Floats() []float64 {
	res := make([]float64, len(s.Intensities))
	for i, x := range s.Intensities {
		res[i] = float64(x) / 255.0
	}
	return res
}

// A DataSet is a collection of samples.
type DataSet struct {
	Samples []Sample

	// These fields indicate the dimensions of
	// the sample bitmaps.
	Width  int
	Height int
}

func LoadTrainingDataSet() DataSet {
	return loadDataSet("train")
}

func LoadTestingDataSet() DataSet {
	return loadDataSet("t10k")
}

func loadDataSet(prefix string) DataSet {
	labelFilename := prefix + "-labels-idx1-ubyte.gz"
	imageFilename := prefix + "-images-idx3-ubyte.gz"
	intensities, w, h, err := readIntensities(assetReader(imageFilename))
	if err != nil {
		panic("failed to read images: " + err.Error())
	}
	labels, err := readLabels(assetReader(labelFilename), len(intensities))
	if err != nil {
		panic("failed to read labels: " + err.Error())
	}
	var dataSet DataSet
	dataSet.Width = w
	dataSet.Height = h
	dataSet.Samples = make([]Sample, len(intensities))
	for i := range dataSet.Samples {
		dataSet.Samples[i].Intensities = intensities[i]
		dataSet.Samples[i].Label = labels[i]
	}
	return dataSet
}

// FloatVectors returns a slice of intensity float
// vectors, one per sample.
func (d DataSet) FloatVectors() [][]float64 {
	res := make([][]float64, len(d.Samples))
	for i, sample := range d.Samples {
		res[i] = sample.Floats()
	}
	return res
}

// LabelVectors returns a slice of output vectors,
// where the first value of an output vector is 1
// for samples labeled 0, the second value is
// 1 for samples labeled 1, etc.
//
// This is useful for classifiers such as neural
// networks where the output of the network is a
// vector of probabilities.
func (d DataSet) LabelVectors() [][]float64 {
	res := make([][]float64, len(d.Samples))
	for i, sample := range d.Samples {
		res[i] = make([]float64, 10)
		res[i][sample.Label] = 1
	}
	return res
}

func assetReader(name string) io.Reader {
	data, err := Asset("data/" + name)
	if err != nil {
		panic("could not load asset: " + name)
	}
	reader, err := gzip.NewReader(bytes.NewBuffer(data))
	if err != nil {
		panic(fmt.Sprintf("could not decompress %s: %s", name, err.Error()))
	}
	return reader
}

func readIntensities(reader io.Reader) (results [][]uint8, width, height int, err error) {
	r := bufio.NewReader(reader)
	if _, err := r.Discard(4); err != nil {
		return nil, 0, 0, err
	}

	var params [3]uint32

	for i := 0; i < 3; i++ {
		if err := binary.Read(r, binary.BigEndian, &params[i]); err != nil {
			return nil, 0, 0, err
		}
	}

	count := int(params[0])
	width = int(params[1])
	height = int(params[2])

	results = make([][]uint8, count)
	for j := range results {
		var buffer bytes.Buffer
		limited := io.LimitedReader{R: r, N: int64(width * height)}
		if n, err := io.Copy(&buffer, &limited); err != nil {
			return nil, 0, 0, err
		} else if n < int64(width*height) {
			return nil, 0, 0, errors.New("not enough data for image")
		}

		vec := make([]uint8, width*height)
		for i := range buffer.Bytes() {
			vec[i] = uint8(i)
		}
		results[j] = vec
	}

	return
}

func readLabels(reader io.Reader, count int) ([]int, error) {
	r := bufio.NewReader(reader)

	if _, err := r.Discard(8); err != nil {
		return nil, err
	}

	res := make([]int, count)
	for i := range res {
		label, err := r.ReadByte()
		if err != nil {
			return nil, err
		}
		res[i] = int(label)
	}

	return res, nil
}
