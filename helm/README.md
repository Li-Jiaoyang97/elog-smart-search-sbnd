# Helm chart for the SBND ELog semantic search

An umbrella chart: this chart's own templates are the serving API and its
build objects, and Qdrant comes in as a subchart from upstream.

`deploy/okd/` still holds the hand-written manifests and is untouched. The two
deploy the same thing in different ways, so pick one per namespace. See
[Do not run both](#do-not-run-both).

## What is in here

| Path | What it is |
|---|---|
| `Chart.yaml` | chart metadata; pins the Qdrant subchart to 1.17.1 |
| `values.yaml` | all defaults, with the reasoning for each |
| `values-okd.yaml` | OpenShift overrides, **required on every command** |
| `templates/` | the API Deployment, Service, ConfigMap, models PVC + download Job, ImageStream, BuildConfig |
| `templates/validate.yaml` | fails the render if `values-okd.yaml` was left out |
| `build.sh` | runs a build, tags the image with the commit SHA, prints the deploy command |

The Qdrant StatefulSet, its Services, ConfigMap and ServiceAccount all come
from `qdrant/qdrant` and are not maintained here.

## Why `-f values-okd.yaml` is not optional

The subchart defaults to `runAsUser: 1000`, `runAsGroup: 2000` and
`fsGroup: 3000`. OpenShift allocates a UID range per namespace and
`restricted-v2` rejects any pod that names its own UID, so those three keys
have to be deleted, not changed.

They cannot be deleted from `values.yaml`. Helm ignores a `null` written in a
parent chart's own values file and keeps the subchart's default; a `null` only
deletes when it arrives through `-f` or `--set`. That is the only reason
`values-okd.yaml` exists as a separate file.

`templates/validate.yaml` checks for the keys and fails the render with an
explanatory message, so forgetting the flag is a Helm error rather than a pod
stuck in `CreateContainerConfigError`.

## Install

Prerequisites: `helm` 3.8+ or 4.x and `oc` logged in. The repo is public, so
the in-cluster build clones it with no credential.

```bash
oc login --web https://api.okdprod1.fnal.gov:6443
oc project ai-for-neutrino-detector-ops

helm dependency build helm/          # vendors qdrant-1.17.1.tgz into charts/
helm install elog-search helm/ -f helm/values-okd.yaml --set build.enabled=true --set image.tag=placeholder
```

That first install creates the ImageStream and BuildConfig but cannot start
the API, because no image exists yet. Build one and upgrade:

```bash
./helm/build.sh                      # prints the exact upgrade command
```

`image.tag` has no default and the template calls `required` on it, so an
install can never silently pin `:latest`.

## Check before you apply

```bash
helm lint helm/ --set image.tag=abc1234
helm template elog-search helm/ -f helm/values-okd.yaml --set image.tag=abc1234
oc apply --dry-run=server -f <(helm template elog-search helm/ -f helm/values-okd.yaml --set image.tag=abc1234)
```

The last one is the only check that exercises the cluster's OPA admission
webhook and the SCC. It has not been run yet; everything above it has.

## Do not run both

`deploy/okd/` and this chart create objects with overlapping names, including
the `elog-semantic-search` ImageStream and BuildConfig. Installing the chart
into a namespace that already has the hand-applied manifests will collide.

The Qdrant volume does **not** carry over either. The hand-written StatefulSet
produces the PVC `storage-qdrant-0`; the subchart produces
`qdrant-storage-elog-search-qdrant-0`. Same mount path, different claim, so an
index built under one is invisible to the other. Re-running `run_update.py`
against the new Qdrant is the simple path.

## Pointing at a Qdrant you do not run

```yaml
qdrant:
  enabled: false
qdrantExternalHost: qdrant.some-other-namespace.svc.cluster.local
```

The subchart is skipped and `QDRANT_HOST` takes the external value. The
template fails if `qdrant.enabled` is false and `qdrantExternalHost` is empty.

## Going multi-node

`qdrant.config.cluster.enabled` is `false`, matching the hand-written
StatefulSet and matching how the local index was built. Raising
`qdrant.replicaCount` on its own gives you a second pod holding none of your
data, because `run_update.py:605` creates the collection with no
`replication_factor`.

The real sequence is:

1. Set `qdrant.config.cluster.enabled: true`
2. Pass `replication_factor=2` in `run_update.py`'s `create_collection` call
3. Rebuild the collection, or replicate the existing shard through Qdrant's
   cluster API
4. Raise `qdrant.replicaCount` and add `qdrant.affinity` podAntiAffinity so
   replicas land on different nodes

Steps 2 and 3 are code and data changes, not chart values.

## Not in this chart yet

The index-update CronJob, the ExternalSecret for the ECL password, and a
Route. All three are listed under "Still to do" in `deploy/okd/README.md` and
apply equally here. The CronJob will take `QDRANT_HOST` from the same
`elog-search.qdrantHost` helper the Deployment uses, so the two cannot drift.
