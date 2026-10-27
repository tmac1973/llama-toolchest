package web

import "embed"

//go:embed templates/*.html templates/partials/*.html
var Templates embed.FS

//go:embed static/*
var Static embed.FS

// OpenAPISpec describes the public API (/v1 and the ACE-Step routes);
// it is served at /openapi.json and /openapi.yaml.
//
//go:embed apidocs/openapi.yaml
var OpenAPISpec []byte

// APIDocsPage is the API reference page at /docs/api, which draws
// OpenAPISpec with the Scalar viewer in static/scalar.
//
//go:embed apidocs/index.html
var APIDocsPage []byte
