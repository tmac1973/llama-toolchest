# Can a build from one container image run in the other?

Measured 22 September 2026 on this project's two ROCm images, both llama.cpp
`v0.4.1` with identical cmake flags.

## Where it stands

**Neither direction works as shipped, and neither fails for the reason you would
expect.** The build stamp is a good proxy for "this was built somewhere else",
but it is not a diagnosis — the two failures have nothing in common.

| build made in | run in | result | actual cause |
|---|---|---|---|
| Fedora 43 / ROCm 7.2.4 | Ubuntu 24.04 / ROCm 10 | fails | `libcrypto.so.3: version OPENSSL_3.3.0 not found` — Fedora links OpenSSL 3.3 symbols that Ubuntu 24.04's libcrypto does not provide. **Nothing to do with ROCm.** |
| Ubuntu 24.04 / ROCm 10 | Fedora 43 / ROCm 7.2.4 | fails | `libhipblas.so.3: cannot open shared object file` — but the binary itself is fine. Supplied with `/opt/rocm/lib` it loads and enumerates the GPU correctly. |

The second row is the interesting one. Both images ship the same hipBLAS soname
(`libhipblas.so.3` — 3.2.70204 on Fedora, 3.6 on ROCm 10), so there is no ABI
barrier between them. What breaks is that cmake bakes `/opt/rocm/lib` into
`libggml-hip.so`'s RUNPATH under Fedora and not under ROCm 10, and neither image
has an `ld.so.conf.d` entry for ROCm. `Dockerfile.rocm-next` now adds one, which
is why builds made there work *in that image* — it does not help a build carried
into the other one.

So: a cross-image build may fail for a ROCm reason, an OpenSSL reason, or a
glibc reason, and which one depends on the direction. Any wording that blames
ROCm specifically would be wrong half the time.

## What the UI does about it

Three places, deliberately different in strength:

**Builds page — informs.** A struck-through stamp and a warning mark with a
tooltip. Never blocks: the stamp can be absent, and a build that merely *looks*
wrong may be fine.

**Server page — refuses.** The build picker renders mismatched builds marked
and unselectable, because this is where the confusing failure actually happened:
picking one here produces a loader error that says nothing about images or ROCm
versions, and the user has no way to connect the two. `Auto (newest ref)` says
so too when it would land on an unusable build, since it is otherwise the option
that looks safest.

**The marker is text, not CSS, and that was learned the hard way.** The first
version set `style="text-decoration:line-through"` on the `<option>`. The
attribute was in the HTML and the browser ignored it — an `<option>` takes
almost no styling, because the dropdown is drawn by the platform rather than
laid out by the page. So the build appeared unselectable with no visible reason.
The option now carries a warning sign and the words "cannot run in this image";
the browser greys out a disabled option by itself, so the text only has to
supply the reason.

Three cases stay selectable, each because refusing them would be worse:

- **An unstamped build.** Cannot be judged, so it is never refused.
- **All builds mismatched.** If nothing here could run, refusing everything
  leaves no way to start at all. They stay selectable with a louder hint. The
  test is "is there any candidate", counting unstamped builds as candidates —
  while one exists there is somewhere to fall back to.
- **The build already selected.** A disabled selected `<option>` cannot be
  submitted, so the form would silently post a different value on the next
  change. It is marked but not refused.

The explanation is the select's own tooltip. It started as a line of text under
the control, on the reasoning that an `<option title>` is not shown reliably —
true, but it cluttered a page that has eight other controls, and the marker in
the option text already answers "which one" without hovering. The tooltip
answers "why", and is absent entirely when every build can run.

**The router — reports.** No server-side block on starting a mismatched build.
Deliberate: the stamp is a proxy, the escape hatches above exist for a reason,
and an API caller that has decided to try one should be allowed to. The loader
error it gets is now the last line of defence rather than the first.
