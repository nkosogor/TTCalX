push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# We don't actually `using TTCalX` because it pulls in CUDA / PyCall,
# which are not available in the docs build environment.  Instead we let
# Documenter render the hand-written pages and the tutorial.

using Documenter

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
makedocs(
    sitename = "TTCalX Documentation",
    authors  = "Nikita Kosogorov and contributors",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://nkosogor.github.io/TTCalX/",
        assets     = String[],
    ),
    pages = [
        "Home"            => "index.md",
        "Installation"    => "install.md",
        "Tutorial"        => "tutorial.md",
        "Peeling"         => "peeling.md",
        "GPU CLEAN Imager" => "gpu-clean-imager.md",
        "CLI Reference"   => "cli.md",
        "API Reference"   => "api.md",
        "Internals"       => "internals.md",
    ],
    # No doctests — the module cannot be loaded without CUDA/PyCall
    doctest  = false,
    # Warn on missing docstrings but don't error — we can't introspect the module
    warnonly = true,
)

# ---------------------------------------------------------------------------
# Deploy (only runs on CI when triggered by the workflow)
# ---------------------------------------------------------------------------
deploydocs(
    repo   = "github.com/nkosogor/TTCalX.git",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
