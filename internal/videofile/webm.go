package videofile

import (
	"encoding/binary"
	"math"
)

// EBML element IDs (with their length marker, as written).
const (
	idEBML        = 0x1A45DFA3
	idDocType     = 0x4282
	idSegment     = 0x18538067
	idSeekHead    = 0x114D9B74
	idInfo        = 0x1549A966
	idTimecodeSc  = 0x2AD7B1
	idTracks      = 0x1654AE6B
	idTrackEntry  = 0xAE
	idTrackNumber = 0xD7
	idTrackType   = 0x83
	idCodecID     = 0x86
	idDefaultDur  = 0x23E383
	idCodecDelay  = 0x56AA
	idVideo       = 0xE0
	idPixelWidth  = 0xB0
	idPixelHeight = 0xBA
	idAudio       = 0xE1
	idSampleFreq  = 0xB5
	idChannels    = 0x9F
	idBitDepth    = 0x6264
	idCluster     = 0x1F43B675
	idTimecode    = 0xE7
	idSimpleBlock = 0xA3
	idBlockGroup  = 0xA0
	idBlock       = 0xA1
	idRefBlock    = 0xFB
	idCues        = 0x1C53BB6B
	idChapters    = 0x1043A770
	idTags        = 0x1254C367
	idAttachments = 0x1941A469
)

// level1 are the IDs that end an unknown-size Cluster.
var level1 = map[uint64]bool{
	idSeekHead: true, idInfo: true, idTracks: true, idCluster: true,
	idCues: true, idChapters: true, idTags: true, idAttachments: true,
}

type ebml struct {
	*src
}

// element reads an element header at off: its ID, the data offset, and
// its size (-1 for unknown).
func (e ebml) element(off int64) (id uint64, data, size int64, err error) {
	if err = e.tick(); err != nil {
		return
	}
	b, err := e.at(off, int(min(12, e.size-off)))
	if err != nil {
		return
	}
	idLen := vintLen(b[0])
	if idLen == 0 || idLen > 4 || idLen >= len(b) {
		return 0, 0, 0, codecErr("the WebM file has a damaged element")
	}
	for _, c := range b[:idLen] {
		id = id<<8 | uint64(c)
	}
	sizeLen := vintLen(b[idLen])
	if sizeLen == 0 || idLen+sizeLen > len(b) {
		return 0, 0, 0, codecErr("the WebM file has a damaged element")
	}
	v := uint64(b[idLen]) & (0xFF >> sizeLen)
	allOnes := v == 0xFF>>sizeLen
	for _, c := range b[idLen+1 : idLen+sizeLen] {
		v = v<<8 | uint64(c)
		allOnes = allOnes && c == 0xFF
	}
	data = off + int64(idLen+sizeLen)
	if allOnes {
		return id, data, -1, nil
	}
	if v > uint64(e.size) {
		return 0, 0, 0, codecErr("the WebM file is truncated or damaged")
	}
	return id, data, int64(v), nil
}

// vintLen is the length of a variable-size integer from its first byte, or
// 0 when the byte is 0.
func vintLen(b byte) int {
	for i := range 8 {
		if b&(0x80>>i) != 0 {
			return i + 1
		}
	}
	return 0
}

func (e ebml) uint(data, size int64) (uint64, error) {
	if size < 0 || size > 8 {
		return 0, codecErr("the WebM file has a damaged number")
	}
	b, err := e.at(data, int(size))
	if err != nil {
		return 0, err
	}
	var v uint64
	for _, c := range b {
		v = v<<8 | uint64(c)
	}
	return v, nil
}

func (e ebml) float(data, size int64) (float64, error) {
	b, err := e.at(data, int(size))
	if err != nil {
		return 0, err
	}
	switch size {
	case 4:
		return float64(math.Float32frombits(binary.BigEndian.Uint32(b))), nil
	case 8:
		return math.Float64frombits(binary.BigEndian.Uint64(b)), nil
	}
	return 0, codecErr("the WebM file has a damaged number")
}

func (e ebml) str(data, size int64) (string, error) {
	if size > 256 {
		return "", codecErr("the WebM file has a damaged text field")
	}
	b, err := e.at(data, int(size))
	if err != nil {
		return "", err
	}
	for len(b) > 0 && b[len(b)-1] == 0 {
		b = b[:len(b)-1]
	}
	return string(b), nil
}

// children calls fn for each child of the element whose data is
// [start, end).
func (e ebml) children(start, end int64, fn func(id uint64, data, size int64) error) error {
	for pos := start; pos < end; {
		id, data, size, err := e.element(pos)
		if err != nil {
			return err
		}
		if size < 0 || data+size > end {
			return codecErr("the WebM file has an element of unknown or wrong size; it is not supported")
		}
		if err := fn(id, data, size); err != nil {
			return err
		}
		pos = data + size
	}
	return nil
}

type webmTrack struct {
	number    uint64
	typ       uint64
	codec     string
	defDur    uint64
	delay     uint64
	width     uint64
	height    uint64
	freq      float64
	channels  uint64
	bitDepth  uint64
	blocks    int
	notKey    bool
	laced     bool
	first     int64 // first and last block times, in timecode units
	last      int64
	haveFirst bool
}

type webmProbe struct {
	ebml
	scale  uint64
	tracks []*webmTrack
	blocks map[uint64]*webmTrack // by track number; filled as blocks come
}

func (w *webmProbe) track(n uint64) *webmTrack {
	if t := w.blocks[n]; t != nil {
		return t
	}
	t := &webmTrack{number: n}
	w.blocks[n] = t
	return t
}

func probeWebM(s *src) (Info, error) {
	w := &webmProbe{ebml: ebml{s}, scale: 1_000_000, blocks: map[uint64]*webmTrack{}}
	id, data, size, err := w.element(0)
	if err != nil {
		return Info{}, err
	}
	if id != idEBML || size < 0 {
		return Info{}, codecErr("the WebM file has a damaged header")
	}
	doc := ""
	if err := w.children(data, data+size, func(id uint64, d, n int64) error {
		if id == idDocType {
			doc, err = w.str(d, n)
		}
		return err
	}); err != nil {
		return Info{}, err
	}
	if doc != "webm" && doc != "matroska" {
		return Info{}, ErrUnknownFormat
	}
	pos := data + size
	id, data, size, err = w.element(pos)
	if err != nil {
		return Info{}, err
	}
	if id != idSegment {
		return Info{}, codecErr("the WebM file has no segment")
	}
	end := data + size
	if size < 0 || end > s.size {
		end = s.size
	}
	if err := w.segment(data, end); err != nil {
		return Info{}, err
	}
	return w.info()
}

func (w *webmProbe) segment(start, end int64) error {
	for pos := start; pos < end; {
		id, data, size, err := w.element(pos)
		if err != nil {
			return err
		}
		if size < 0 && id != idCluster {
			return codecErr("the WebM file has an element of unknown size; it is not supported")
		}
		if size >= 0 && data+size > end {
			return codecErr("the WebM file is truncated or damaged")
		}
		switch id {
		case idInfo:
			err = w.children(data, data+size, func(id uint64, d, n int64) error {
				if id == idTimecodeSc {
					v, err := w.uint(d, n)
					if err == nil && v > 0 {
						w.scale = v
					}
					return err
				}
				return nil
			})
		case idTracks:
			err = w.children(data, data+size, func(id uint64, d, n int64) error {
				if id == idTrackEntry {
					return w.trackEntry(d, d+n)
				}
				return nil
			})
		case idCluster:
			var next int64
			next, err = w.cluster(data, size, end)
			if err == nil {
				pos = next
				continue
			}
		}
		if err != nil {
			return err
		}
		pos = data + size
	}
	return nil
}

func (w *webmProbe) trackEntry(start, end int64) error {
	t := &webmTrack{}
	err := w.children(start, end, func(id uint64, d, n int64) error {
		var err error
		switch id {
		case idTrackNumber:
			t.number, err = w.uint(d, n)
		case idTrackType:
			t.typ, err = w.uint(d, n)
		case idCodecID:
			t.codec, err = w.str(d, n)
		case idDefaultDur:
			t.defDur, err = w.uint(d, n)
		case idCodecDelay:
			t.delay, err = w.uint(d, n)
		case idVideo:
			err = w.children(d, d+n, func(id uint64, d, n int64) error {
				var err error
				switch id {
				case idPixelWidth:
					t.width, err = w.uint(d, n)
				case idPixelHeight:
					t.height, err = w.uint(d, n)
				}
				return err
			})
		case idAudio:
			t.freq = 8000 // Matroska's default
			err = w.children(d, d+n, func(id uint64, d, n int64) error {
				var err error
				switch id {
				case idSampleFreq:
					t.freq, err = w.float(d, n)
				case idChannels:
					t.channels, err = w.uint(d, n)
				case idBitDepth:
					t.bitDepth, err = w.uint(d, n)
				}
				return err
			})
		}
		return err
	})
	w.tracks = append(w.tracks, t)
	return err
}

// cluster reads a Cluster's blocks and returns where the next element
// starts. An unknown-size cluster ends at the next level-1 element.
func (w *webmProbe) cluster(data, size, segEnd int64) (int64, error) {
	end := segEnd
	if size >= 0 {
		end = data + size
	}
	var base int64
	for pos := data; pos < end; {
		id, d, n, err := w.element(pos)
		if err != nil {
			return 0, err
		}
		if size < 0 && level1[id] {
			return pos, nil
		}
		if n < 0 || d+n > end {
			return 0, codecErr("the WebM file has a block of unknown or wrong size; it is not supported")
		}
		switch id {
		case idTimecode:
			v, err := w.uint(d, n)
			if err != nil {
				return 0, err
			}
			base = int64(v)
		case idSimpleBlock:
			if err := w.block(d, n, base, true, false); err != nil {
				return 0, err
			}
		case idBlockGroup:
			var blockData, blockSize int64 = -1, 0
			ref := false
			if err := w.children(d, d+n, func(id uint64, d, n int64) error {
				switch id {
				case idBlock:
					blockData, blockSize = d, n
				case idRefBlock:
					ref = true
				}
				return nil
			}); err != nil {
				return 0, err
			}
			if blockData >= 0 {
				if err := w.block(blockData, blockSize, base, false, !ref); err != nil {
					return 0, err
				}
			}
		}
		pos = d + n
	}
	return end, nil
}

// block reads a (Simple)Block header: track number, relative time and
// flags. For a SimpleBlock the keyframe flag is in the flags; for a Block
// in a BlockGroup it is key when the group has no ReferenceBlock.
func (w *webmProbe) block(data, size, base int64, simple, groupKey bool) error {
	b, err := w.at(data, int(min(size, 11)))
	if err != nil {
		return err
	}
	l := vintLen(b[0])
	if l == 0 || l+3 > len(b) {
		return codecErr("the WebM file has a damaged block")
	}
	track := uint64(b[0]) & (0xFF >> l)
	for _, c := range b[1:l] {
		track = track<<8 | uint64(c)
	}
	rel := int64(int16(binary.BigEndian.Uint16(b[l:])))
	flags := b[l+2]
	t := w.track(track)
	t.blocks++
	key := groupKey
	if simple {
		key = flags&0x80 != 0
	}
	if !key || flags&0x08 != 0 {
		t.notKey = true
	}
	if flags&0x06 != 0 {
		t.laced = true
	}
	at := base + rel
	if !t.haveFirst {
		t.first, t.haveFirst = at, true
	}
	t.last = at
	return nil
}

func (w *webmProbe) info() (Info, error) {
	var video, audio *webmTrack
	for _, t := range w.tracks {
		switch {
		case t.typ == 1 && video == nil:
			video = t
		case t.typ == 2 && audio == nil:
			audio = t
		}
	}
	if video == nil {
		return Info{}, codecErr("the WebM file has no video track")
	}
	if video.codec != "V_VP8" {
		return Info{}, codecErr("the WebM video is %s; send WebM with independent VP8 frames, as written by stable-diffusion.cpp", printable(video.codec))
	}
	if video.width < 1 || video.width > 16384 || video.height < 1 || video.height > 16384 {
		return Info{}, codecErr("the WebM video size %d×%d is not supported", video.width, video.height)
	}
	if audio != nil && (audio.codec != "A_PCM/INT/LIT" || audio.bitDepth != 16 || audio.channels < 1 || audio.channels > 32 ||
		audio.freq < 1000 || audio.freq > 384000 || audio.freq != math.Floor(audio.freq) || audio.delay != 0) {
		return Info{}, codecErr("the WebM audio is %s; WebM audio must be PCM16 (or send the clip with no audio)", printable(audio.codec))
	}
	vb := w.blocks[video.number]
	if vb == nil || vb.blocks == 0 {
		return Info{}, codecErr("the WebM file has no video frames")
	}
	if vb.notKey || vb.laced {
		return Info{}, codecErr("the WebM video must use independent VP8 frames (every frame a keyframe, no lacing), as written by stable-diffusion.cpp")
	}
	if audio != nil {
		if ab := w.blocks[audio.number]; ab != nil && ab.laced {
			return Info{}, codecErr("the WebM audio uses laced blocks, which are not supported")
		}
	}
	info := Info{Format: "webm", Width: int(video.width), Height: int(video.height), Frames: vb.blocks, HasAudio: audio != nil}
	switch {
	case video.defDur > 0:
		info.FPSNum, info.FPSDen = rate(1e9 / float64(video.defDur))
	case vb.blocks > 1 && vb.last > vb.first:
		ns := float64(vb.last-vb.first) * float64(w.scale) / float64(vb.blocks-1)
		info.FPSNum, info.FPSDen = rate(1e9 / ns)
	}
	return info, nil
}

// rate turns a frame rate into a fraction: n/1 or n×1000/1001 when it is
// that close, else rounded to 1/1000.
func rate(fps float64) (int, int) {
	if fps <= 0 || math.IsInf(fps, 0) || math.IsNaN(fps) {
		return 0, 0
	}
	if n := math.Round(fps); math.Abs(fps-n) < 0.01 {
		return reduce(int(n), 1)
	}
	if n := math.Round(fps * 1.001); math.Abs(fps-n*1000/1001) < 0.01 {
		return reduce(int(n)*1000, 1001)
	}
	return reduce(int(math.Round(fps*1000)), 1000)
}
