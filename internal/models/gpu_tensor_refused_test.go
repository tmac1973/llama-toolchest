package models

import "testing"

// A model whose architecture the build cannot split by tensor: the
// tensor options are greyed out with the reason, except the one already
// chosen (the form must still send it back), and none is recommended.
func TestDisableTensorOptions(t *testing.T) {
	opts := GPUAssignOptions(3, nil)
	DisableTensorOptions(opts, "tensor-3", "lfm3")
	sawTensor := false
	for _, o := range opts {
		switch {
		case o.Value == "tensor-3":
			if o.Disabled || o.Note != "" {
				t.Errorf("the chosen option was disabled: %+v", o)
			}
		case o.IsTensor:
			sawTensor = true
			if !o.Disabled || o.Note == "" {
				t.Errorf("tensor option left usable: %+v", o)
			}
		default:
			if o.Disabled || o.Note != "" {
				t.Errorf("layer option disabled: %+v", o)
			}
		}
	}
	if !sawTensor {
		t.Fatal("no other tensor option to check")
	}

	opts = GPUAssignOptions(2, nil)
	DisableTensorOptions(opts, "all", "lfm3")
	MarkRecommended(opts, 1, 24, nil)
	for _, o := range opts {
		if o.Recommend && o.Disabled {
			t.Errorf("a disabled option was recommended: %+v", o)
		}
	}
}
