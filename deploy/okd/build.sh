#!/usr/bin/env bash
# Build the app image from the pushed commit and tag it with that commit's
# SHA, so an image always names the source it came from.
#
# BuildConfig writes to :latest because OKD has no way to put the commit in
# the output tag itself (no substitution there, no --to on start-build), so
# the SHA tag is applied here, after the build.
set -euo pipefail

# Same string, two different resources: the BuildConfig is the recipe, the
# ImageStream is where the finished image lands. OKD keeps them apart by kind
# (bc/... vs is/...), so the matching name is convention, not coupling.
BC=elog-semantic-search
IS=elog-semantic-search
SHA=$(git rev-parse --short HEAD)

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree is dirty. The build uses the pushed commit, not these" >&2
    echo "changes, so the $SHA tag would be wrong. Commit and push first." >&2
    exit 1
fi

if ! git merge-base --is-ancestor HEAD "@{upstream}" 2>/dev/null; then
    echo "HEAD is not on the remote yet. Push first, or the build clones an" >&2
    echo "older commit than $SHA." >&2
    exit 1
fi

oc start-build "$BC" --follow --commit="$SHA"
oc tag "$IS:latest" "$IS:$SHA"

echo
echo "Built and tagged $IS:$SHA"
echo "To deploy it, set this in deploy/okd/app-deployment.yaml:"
echo "    image: $IS:$SHA"
