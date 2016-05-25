package mnist

import (
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
)

const gridSpacing = 4
const smallGridSpace = 2

var gridBackground = color.RGBA{R: 0x56, G: 0xbc, B: 0xd4, A: 0xff}

// A Reconstructor produces output images for
// input images.
// For example, an autoencoder could be used
// as a Reconstructor.
type Reconstructor func(img []float64) []float64

// ReconstructionGrid produces an image which
// is a grid of randomly-chosen samples alongside
// their reconstructions.
//
// This is useful for visualizing the quality of
// an autoencoder or another kind of Reconstructor.
func ReconstructionGrid(r Reconstructor, d DataSet, rows, cols int) image.Image {
	imageWidth := gridSpacing*(cols+1) + cols*(d.Width*2+smallGridSpace)
	imageHeight := gridSpacing*(rows+1) + rows*d.Height
	img := image.NewRGBA(image.Rect(0, 0, imageWidth, imageHeight))

	// Set a non-gray background color so that it can be
	// distinguished from the actual samples.
	for y := 0; y < imageHeight; y++ {
		for x := 0; x < imageWidth; x++ {
			img.Set(x, y, gridBackground)
		}
	}

	for row := 0; row < rows; row++ {
		y := gridSpacing*(row+1) + row*d.Height
		for col := 0; col < cols; col++ {
			x := gridSpacing*(col+1) + col*(d.Width*2+smallGridSpace)

			sample := d.Samples[rand.Intn(len(d.Samples))]
			pixelIdx := 0
			for ry := 0; ry < d.Height; ry++ {
				for rx := 0; rx < d.Width; rx++ {
					pixel := sample.Intensities[pixelIdx]
					img.Set(x+rx, y+ry, color.Gray{Y: 0xff - uint8(pixel*0xff)})
					pixelIdx++
				}
			}

			x += d.Width + smallGridSpace
			reconstruction := r(sample.Intensities)
			pixelIdx = 0
			for ry := 0; ry < d.Height; ry++ {
				for rx := 0; rx < d.Width; rx++ {
					pixel := reconstruction[pixelIdx]
					img.Set(x+rx, y+ry, color.Gray{Y: 0xff - uint8(pixel*0xff)})
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
