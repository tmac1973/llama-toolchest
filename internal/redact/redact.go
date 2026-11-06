// Package redact takes credentials out of URLs before they reach a log or
// an error message.
package redact

import (
	"errors"
	"net/url"
)

// URL returns raw without credentials: a user name or password becomes
// "xxxxx" (a token can be the user name alone: https://TOKEN@host/...),
// and a query string, which can hold a signature (a CDN's presigned
// download URL), becomes "xxxxx". A value that is not a URL is returned
// unchanged.
func URL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || (u.User == nil && u.RawQuery == "") {
		return raw
	}
	if u.User != nil {
		if _, hasPassword := u.User.Password(); !hasPassword {
			u.User = url.User("xxxxx")
		}
	}
	if u.RawQuery != "" {
		u.RawQuery = "xxxxx"
	}
	return u.Redacted()
}

// Error returns err with the URL of a *url.Error in its chain passed
// through URL. Call it on the error as the HTTP client returns it, before
// it is wrapped: fmt.Errorf copies the text it wraps. The error is changed
// in place; errors.Is and errors.As still match it.
func Error(err error) error {
	var ue *url.Error
	if errors.As(err, &ue) {
		ue.URL = URL(ue.URL)
	}
	return err
}
