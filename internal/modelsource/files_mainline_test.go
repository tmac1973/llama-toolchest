package modelsource

import "testing"

// In a repository with renamed mainline copies, the copies are marked as
// such and the root files with the same name are marked as having one.
// A file with no copy is not marked.
func TestMarkMainlineCopies(t *testing.T) {
	files := []File{
		{Filename: "Model-UD-Q4_K_XL.gguf"},
		{Filename: "Model-UD-Q8_K_XL.gguf"},
		{Filename: "mmproj-F16.gguf"},
		{Filename: "Mainline_Support/Model-UD-Q4_K_XL.gguf_file"},
	}
	MarkMainlineCopies(files)
	if !files[0].HasMainlineCopy || files[0].MainlineCopy {
		t.Errorf("root Q4 = %+v, want HasMainlineCopy only", files[0])
	}
	if files[1].HasMainlineCopy || files[2].HasMainlineCopy {
		t.Errorf("files with no copy were marked: %+v %+v", files[1], files[2])
	}
	if !files[3].MainlineCopy || files[3].HasMainlineCopy {
		t.Errorf("mainline copy = %+v, want MainlineCopy only", files[3])
	}
}
