package videofile

// probeWebP reads a WebP's canvas size and frame count. A still WebP is one
// frame with no rate: video::load then needs an explicit fps. An animated
// WebP's rate comes from its frame durations, which must all be equal, as
// sd.cpp infers a fixed rate from them.
func probeWebP(s *src) (Info, error) {
	head, err := s.at(0, 12)
	if err != nil {
		return Info{}, err
	}
	end := int64(le32(head[4:])) + 8
	if end > s.size {
		return Info{}, codecErr("the WebP file is truncated or damaged")
	}
	info := Info{Format: "webp"}
	animated, first := false, true
	duration := -1
	for pos := int64(12); pos+8 <= end; {
		if err := s.tick(); err != nil {
			return Info{}, err
		}
		h, err := s.at(pos, 8)
		if err != nil {
			return Info{}, err
		}
		id, size := string(h[:4]), int64(le32(h[4:]))
		data := pos + 8
		if data+size > end {
			return Info{}, codecErr("the WebP file is truncated or damaged")
		}
		switch {
		case first && id == "VP8 ":
			b, err := s.at(data, min(10, int(size)))
			if err != nil || len(b) < 10 || b[3] != 0x9d || b[4] != 0x01 || b[5] != 0x2a {
				return Info{}, codecErr("the WebP file has a damaged VP8 frame")
			}
			info.Width, info.Height, info.Frames = le16(b[6:])&0x3fff, le16(b[8:])&0x3fff, 1
		case first && id == "VP8L":
			b, err := s.at(data, min(5, int(size)))
			if err != nil || len(b) < 5 || b[0] != 0x2f {
				return Info{}, codecErr("the WebP file has a damaged lossless frame")
			}
			bits := le32(b[1:])
			info.Width, info.Height, info.Frames = int(bits&0x3fff)+1, int(bits>>14&0x3fff)+1, 1
		case first && id == "VP8X":
			b, err := s.at(data, min(10, int(size)))
			if err != nil || len(b) < 10 {
				return Info{}, codecErr("the WebP file has a damaged header")
			}
			animated = b[0]&0x02 != 0
			info.Width, info.Height = le24(b[4:])+1, le24(b[7:])+1
			if !animated {
				info.Frames = 1
			}
		case animated && id == "ANMF":
			b, err := s.at(data, min(16, int(size)))
			if err != nil || len(b) < 16 {
				return Info{}, codecErr("the WebP file has a damaged animation frame")
			}
			d := le24(b[12:])
			switch {
			case duration < 0:
				duration = d
			case d != duration:
				return Info{}, codecErr("the animated WebP frames have different durations (%d ms and %d ms); video upscaling needs a fixed frame rate", duration, d)
			}
			info.Frames++
		case first:
			return Info{}, codecErr("the WebP file has no image")
		}
		first = false
		pos = data + size + size&1
	}
	if info.Width < 1 || info.Width > 16384 || info.Height < 1 || info.Height > 16384 {
		return Info{}, codecErr("the WebP size %d×%d is not supported", info.Width, info.Height)
	}
	if animated && duration > 0 {
		info.FPSNum, info.FPSDen = reduce(1000, duration)
	}
	return info, nil
}

func le24(b []byte) int { return int(b[0]) | int(b[1])<<8 | int(b[2])<<16 }
