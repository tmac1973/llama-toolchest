package models

import (
	"fmt"
	"testing"
)

func TestDeviceIndicesForConfig(t *testing.T) {
	cases := []struct {
		cfg  ModelConfig
		want []int
	}{
		{ModelConfig{GPUAssign: "all"}, nil},
		{ModelConfig{}, nil},
		{ModelConfig{GPUAssign: "2"}, []int{2}},
		{ModelConfig{GPUAssign: "0,2"}, []int{0, 2}},
		{ModelConfig{GPUAssign: "tensor:1,2"}, []int{1, 2}},
		{ModelConfig{GPUAssign: "tensor-3"}, nil},
		{ModelConfig{TensorSplit: "0,1,1"}, []int{1, 2}},
	}
	for _, tc := range cases {
		got := DeviceIndicesForConfig(&tc.cfg, 3)
		if fmt.Sprint(got) != fmt.Sprint(tc.want) {
			t.Errorf("%+v: got %v, want %v", tc.cfg, got, tc.want)
		}
	}
}
