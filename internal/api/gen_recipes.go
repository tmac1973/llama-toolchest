package api

import (
	"log/slog"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// refreshRecipes makes the newest stable-diffusion.cpp build that this
// machine's GPUs can run the source of the active recipe list, or the
// built-in snapshot when there is none. It runs at start and after each
// build finishes or is deleted; it is safe to run at the same time as
// itself.
func (s *Server) refreshRecipes() {
	book := generation.SnapshotRecipes()
	if builds := s.genBuildsFor(generation.KindSD); len(builds) > 0 {
		// A forced rebuild keeps its ID, so read its folder again.
		generation.ForgetBuildRecipes(builds[0].ID)
		book = generation.RecipesForBuild(builds[0])
	}
	generation.SetActiveRecipes(book)
	slog.Info("recipe list", "source", book.Source, "recipes", book.Len())
}
