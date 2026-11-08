package config

import (
	"errors"
	"regexp"
	"strings"
)

// DefaultModelConfig sets the virtual model name: a name clients send
// instead of a real model, which resolves on each chat request to the
// pinned model or the chat model used last. It is on unless Off is set,
// so a config from before the setting has it on.
type DefaultModelConfig struct {
	Off bool `yaml:"off,omitempty"`
	// Name is what clients send; "" means DefaultModelNameDefault.
	Name string `yaml:"name,omitempty"`
	// Pin is the model the name always resolves to: a registry ID, or
	// DefaultModelForkPrefix and a fork instance ID. "" means no pin.
	Pin string `yaml:"pin,omitempty"`
}

const (
	// DefaultModelNameDefault is the virtual model name of a config that
	// sets none.
	DefaultModelNameDefault = "default"
	// DefaultModelForkPrefix marks a DefaultModelConfig.Pin that names a
	// fork instance rather than a registry model.
	DefaultModelForkPrefix = "fork:"
)

// DefaultModelName is the virtual model name clients send: the saved
// name, or DefaultModelNameDefault when none is set.
func (c *Config) DefaultModelName() string {
	if name := strings.TrimSpace(c.DefaultModel.Name); name != "" {
		return name
	}
	return DefaultModelNameDefault
}

// DefaultModelOn reports whether the virtual model name is switched on.
func (c *Config) DefaultModelOn() bool {
	return !c.DefaultModel.Off
}

var defaultModelNameRE = regexp.MustCompile(`^[A-Za-z0-9._:/-]{1,64}$`)

// ValidDefaultModelName checks a virtual model name: 1 to 64 letters,
// digits and . _ - : / after trimming. It does not check for a clash with
// a real model name; that needs the registry.
func ValidDefaultModelName(name string) error {
	if !defaultModelNameRE.MatchString(strings.TrimSpace(name)) {
		return errors.New("use 1 to 64 letters (A-Z, a-z), digits and . _ - : /")
	}
	return nil
}
