package mnist

import "testing"

func TestDataSets(t *testing.T) {
	dataSet := LoadTrainingDataSet()
	if len(dataSet.Samples) != 60000 {
		t.Errorf("invalid sample count: %d (expected 60,000)", len(dataSet.Samples))
	}
	if dataSet.Width != 28 || dataSet.Height != 28 {
		t.Errorf("invalid dimensions: %dx%d (expected 28x28)", dataSet.Width, dataSet.Height)
	}

	dataSet = LoadTestingDataSet()
	if len(dataSet.Samples) != 10000 {
		t.Errorf("invalid sample count: %d (expected 10,000)", len(dataSet.Samples))
	}
	if dataSet.Width != 28 || dataSet.Height != 28 {
		t.Errorf("invalid dimensions: %dx%d (expected 28x28)", dataSet.Width, dataSet.Height)
	}
}
