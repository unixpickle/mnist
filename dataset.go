package mnist

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// A Classifier classifies an image (data) as a
// digit between 0 and 9 (inclusive).
type Classifier func(data []float64) int

// A Sample is one instance of a handwritten digit.
type Sample struct {
	// Intensities is a bitmap of white-and-black
	// values, where 1 is black and 0 is white.
	Intensities []float64

	// Label is a number between 0 and 9 (inclusive)
	// indicating what digit this is.
	Label int
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
		floats := make([]float64, len(intensities[i]))
		for i, x := range intensities[i] {
			floats[i] = float64(x) / 255.0
		}
		dataSet.Samples[i].Intensities = floats
		dataSet.Samples[i].Label = labels[i]
	}
	return dataSet
}

// IntensityVectors returns a slice of intensity
// vectors, one per sample.
func (d DataSet) IntensityVectors() [][]float64 {
	res := make([][]float64, len(d.Samples))
	for i, sample := range d.Samples {
		res[i] = sample.Intensities
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

// NumCorrect reports the number of samples a
// Classifier correctly classifies.
func (d DataSet) NumCorrect(classifier Classifier) int {
	var count int
	for _, sample := range d.Samples {
		c := classifier(sample.Intensities)
		if c == sample.Label {
			count++
		}
	}
	return count
}

// CorrectnessHistogram returns a human-readable
// string indicating how many of each digit a
// classifier gets correct.
// For example, its output might start like
// "0: 50.25%, 1: 90.32%, 2: 30.15%".
func (d DataSet) CorrectnessHistogram(classifier Classifier) string {
	var correct [10]int
	var total [10]int
	for _, sample := range d.Samples {
		c := classifier(sample.Intensities)
		if c == sample.Label {
			correct[sample.Label]++
		}
		total[sample.Label]++
	}

	histogramParts := make([]string, 10)
	for i := range histogramParts {
		histogramParts[i] = fmt.Sprintf("%d: %0.2f%%", i,
			100*float64(correct[i])/float64(total[i]))
	}
	return strings.Join(histogramParts, ", ")
}

// SGDSampleSet creates an sgd.SampleSet full of
// neuralnet.VectorSample entries.
// Each entry contains the intensity vector and
// label vector for a digit.
func (d DataSet) SGDSampleSet() sgd.SampleSet {
	labelVecs := d.LabelVectors()
	inputVecs := d.IntensityVectors()
	return neuralnet.VectorSampleSet(vecVec(inputVecs), vecVec(labelVecs))
}

// AnyNetSamples creates an anyff.SampleList.
// The output vector for each sample is a one-hot encoding
// of the correct digit.
func (d DataSet) AnyNetSamples(c anyvec.Creator) anyff.SampleList {
	var res anyff.SliceSampleList
	labVec := d.LabelVectors()
	for i, x := range d.IntensityVectors() {
		res = append(res, &anyff.Sample{
			Input:  c.MakeVectorData(c.MakeNumericList(x)),
			Output: c.MakeVectorData(c.MakeNumericList(labVec[i])),
		})
	}
	return res
}

func vecVec(f [][]float64) []linalg.Vector {
	res := make([]linalg.Vector, len(f))
	for i, x := range f {
		res[i] = x
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
		for i, b := range buffer.Bytes() {
			vec[i] = uint8(b)
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
