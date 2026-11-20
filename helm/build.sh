#!/usr/bin/env bash
# Build the app image from the pushed commit, tag it with that commit's SHA,
# then print the helm command that deploys it.
#
# BuildConfig writes to :latest because OKD has no way to put the commit in
# the output tag itself (no substitution there, no --to on start-build), so
# the SHA tag is applied here, after the build.
set -euo pipefail

CHART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE="${RELEASE:-elog-search}"

# Same string, two different resources: the BuildConfig is the recipe, the
# ImageStream is where the finished image lands. Both are named from
# image.repository in values.yaml.
NAME=$(helm show values "$CHART_DIR" | awk '/^image:/{f=1} f&&/^  repository:/{print $2; exit}')
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

oc start-build "$NAME" --follow --commit="$SHA"
oc tag "$NAME:latest" "$NAME:$SHA"

echo
echo "Built and tagged $NAME:$SHA"
echo "To deploy it:"
echo
echo "    helm upgrade --install $RELEASE $CHART_DIR \\"
echo "        -f $CHART_DIR/values-okd.yaml \\"
echo "        --set image.tag=$SHA"
