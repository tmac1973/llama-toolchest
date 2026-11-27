package api

// sd.cpp's DGL upscale op takes only 3-channel pictures (issue #737).
// image::load keeps what the PNG has: a gray picture gives 1 channel, and
// a picture whose alpha is not 255 everywhere gives 4. Toolchest therefore
// writes the input as an RGB PNG. When the picture has alpha, toolchest
// scales the alpha to the result's size and puts it back on the result,
// so a picture with a clear background keeps it.

import (
	"image"
	"image/color"
	"math"
)

// splitAlpha gives img's colors as a fully opaque picture, which Go's PNG
// encoder writes as RGB, and img's alpha, or nil when every pixel is
// opaque. The colors are not premultiplied: a pixel that is half clear
// keeps its own color, not a darker one.
func splitAlpha(img image.Image) (*image.NRGBA, *image.Gray) {
	b := img.Bounds()
	rgb := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	alpha := image.NewGray(rgb.Rect)
	opaque := true
	for y := range b.Dy() {
		for x := range b.Dx() {
			c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			if c.A != 255 {
				opaque = false
			}
			alpha.Pix[y*alpha.Stride+x] = c.A
			c.A = 255
			rgb.SetNRGBA(x, y, c)
		}
	}
	if opaque {
		return rgb, nil
	}
	return rgb, alpha
}

// withAlpha gives img with alpha, scaled to img's size, as its alpha.
func withAlpha(img image.Image, alpha *image.Gray) *image.NRGBA {
	b := img.Bounds()
	a := scaleGray(alpha, b.Dx(), b.Dy())
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := range b.Dy() {
		for x := range b.Dx() {
			c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			c.A = a.Pix[y*a.Stride+x]
			out.SetNRGBA(x, y, c)
		}
	}
	return out
}

// scaleGray scales g to w×h with bilinear filtering. Alpha is mostly flat
// areas and soft edges, so it needs no ESRGAN pass of its own.
func scaleGray(g *image.Gray, w, h int) *image.Gray {
	sw, sh := g.Rect.Dx(), g.Rect.Dy()
	out := image.NewGray(image.Rect(0, 0, w, h))
	if sw == 0 || sh == 0 {
		return out
	}
	at := func(x, y int) float64 {
		x = min(max(x, 0), sw-1)
		y = min(max(y, 0), sh-1)
		return float64(g.Pix[y*g.Stride+x])
	}
	for y := range h {
		fy := (float64(y)+0.5)*float64(sh)/float64(h) - 0.5
		y0 := int(math.Floor(fy))
		ty := fy - float64(y0)
		for x := range w {
			fx := (float64(x)+0.5)*float64(sw)/float64(w) - 0.5
			x0 := int(math.Floor(fx))
			tx := fx - float64(x0)
			top := at(x0, y0)*(1-tx) + at(x0+1, y0)*tx
			bottom := at(x0, y0+1)*(1-tx) + at(x0+1, y0+1)*tx
			out.Pix[y*out.Stride+x] = uint8(math.Round(top*(1-ty) + bottom*ty))
		}
	}
	return out
}
