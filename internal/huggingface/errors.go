package huggingface

import (
	"fmt"
	"net/http"
	"strings"
)

// statusError turns a failed Hugging Face answer into an error a user can
// act on. Hugging Face names the cause in two headers: X-Error-Code (for
// example GatedRepo, RepoNotFound or EntryNotFound) and X-Error-Message,
// a sentence for people. A bare "HTTP 403" tells the user nothing, so the
// message says what went wrong and what to do about it. The HTTP status
// stays at the end for bug reports.
//
// hasToken is whether the request carried a token: the same 401 means
// "add a token" without one and "the token is wrong" with one.
func statusError(resp *http.Response, modelID string, hasToken bool) error {
	code := resp.Header.Get("X-Error-Code")
	hfMsg := strings.TrimSpace(resp.Header.Get("X-Error-Message"))
	status := resp.StatusCode
	page := SiteURL + "/" + modelID

	var msg string
	switch {
	case code == "GatedRepo" && !hasToken:
		msg = fmt.Sprintf("%s is a gated model. Open %s, sign in and agree to its access terms. "+
			"Then save a HuggingFace token in Settings and try again.", modelID, page)
	case code == "GatedRepo" || (status == http.StatusForbidden && hasToken):
		// A 403 with a token: Hugging Face knows who you are but will not
		// let the token read this repository.
		msg = fmt.Sprintf("Hugging Face refused the saved token for %s. Open %s and check that you have access "+
			"(agree to the access terms, or wait for the owner to approve your request). "+
			"A fine-grained token also needs \"Read contents of gated repos you can access\" "+
			"turned on in its permissions.", modelID, page)
	case status == http.StatusUnauthorized && hasToken:
		msg = fmt.Sprintf("Hugging Face did not accept the saved token for %s. "+
			"The token may be wrong, expired or deleted. Make a new one and save it in Settings.", modelID)
	case code == "RepoNotFound" || status == http.StatusUnauthorized:
		// Hugging Face answers 401 for a repository that does not exist,
		// so that it does not tell strangers which private ones do.
		msg = fmt.Sprintf("Hugging Face cannot find %s. Check the name. "+
			"If it is a private repository, save a HuggingFace token that can read it in Settings.", modelID)
		if hasToken {
			msg = fmt.Sprintf("Hugging Face cannot find %s, or the saved token cannot read it. Check the name and the token's permissions.", modelID)
		}
	case code == "EntryNotFound" || status == http.StatusNotFound:
		msg = fmt.Sprintf("The file is not in %s any more. The repository may have been changed; open it in Browse to see its files now.", modelID)
	case code == "RevisionNotFound":
		msg = fmt.Sprintf("The version of %s this download asked for is gone from Hugging Face. Start the download again from Browse.", modelID)
	case status == http.StatusTooManyRequests:
		msg = "Hugging Face is limiting requests from this address. Try again in a few minutes."
	case status >= 500:
		msg = "Hugging Face had a server problem. Try again later."
	default:
		msg = "Hugging Face refused the request."
		if hfMsg != "" {
			msg += " It said: " + hfMsg
		}
	}

	// For the access cases, Hugging Face's own sentence often says the
	// exact cause (not on the approved list, fine-grained token without
	// gated access), so it goes along. It is left out where it adds
	// nothing, such as "Entry not found".
	if hfMsg != "" && status == http.StatusForbidden {
		msg += " Hugging Face said: " + hfMsg
	}
	return fmt.Errorf("%s (HTTP %d)", msg, status)
}
