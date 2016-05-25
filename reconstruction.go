package mnist

import (
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
)

const gridSpacing = 3

// A Reconstructor produces output images for
// input images.
// For example, an autoencoder could be used
// as a Reconstructor.
type Reconstructor func(img []float64) []float64

// ReconstructionGrid produces an image which
// is a grid of reconstructed samples chosen
// randomly from d.
// This is useful for visualizing the quality of
// an autoencoder or other Reconstructor.
func ReconstructionGrid(r Reconstructor, d DataSet, rows, cols int) image.Image {
	imageWidth := gridSpacing*(cols+1) + cols*d.Width
	imageHeight := gridSpacing*(rows+1) + rows*d.Height
	img := image.NewGray(image.Rect(0, 0, imageWidth, imageHeight))

	for row := 0; row < rows; row++ {
		y := gridSpacing*(row+1) + row*d.Height
		for col := 0; col < cols; col++ {
			x := gridSpacing*(col+1) + col*d.Width

			sample := d.Samples[rand.Intn(len(d.Samples))]
			reconstruction := r(sample.Intensities)
			pixelIdx := 0
			for ry := 0; ry < d.Height; ry++ {
				for rx := 0; rx < d.Width; rx++ {
					pixel := reconstruction[pixelIdx]
					img.SetGray(x+rx, y+ry, color.Gray{Y: 0xff - uint8(pixel*0xff)})
					pixelIdx++
				}
			}
		}
	}

	return img
}

// SaveReconstructionGrid is like ReconstructionGrid,
// but it saves the result to a PNG file.
func SaveReconstructionGrid(path string, r Reconstructor, d DataSet, rows, cols int) error {
	img := ReconstructionGrid(r, d, rows, cols)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
