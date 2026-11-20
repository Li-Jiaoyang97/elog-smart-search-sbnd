# Deploying to OKD

Manifests for running the SBND elog semantic search on okdprod1, namespace
`ai-for-neutrino-detector-ops`. This is the in-cluster version of what
`docker-compose.yml` runs locally: the same two containers, the same single
collection, backed by persistent volumes instead of bind mounts.

Everything here is a file you can read and diff. The repo is public, so the
in-cluster build clones it with no credential.

## What is here

| File | What it is |
|---|---|
| `configmap.yaml` | collection name, model paths, rerank settings |
| `models-pvc.yaml` | 5Gi RWX volume for the two models |
| `models-download-job.yaml` | one-shot fetch of both models from HuggingFace |
| `qdrant-statefulset.yaml` | the vector database, with its own 10Gi volume |
| `qdrant-service.yaml` | headless service, gives the pod a stable DNS name |
| `imagestream.yaml` | where built images are stored |
| `buildconfig.yaml` | how to build the app image from this repo |
| `build.sh` | runs a build, then tags the result with the commit SHA |
| `app-deployment.yaml` | the serving API |
| `app-service.yaml` | cluster-internal address for the API |

There is no Route. OpenWebUI currently runs on a laptop and reaches the API
through `oc port-forward`; if it ever moves into this namespace it will reach
the service directly and still not need one.

## Before you start

```bash
oc login --web https://api.okdprod1.fnal.gov:6443
oc project ai-for-neutrino-detector-ops
```

Tokens expire. If a command starts failing with `Unauthorized`, log in again.

## Order to apply things

The order matters in two places: the models volume has to hold the models
before the API starts, and Qdrant has to exist before the API can reach it.

```bash
# 1. Config and the models volume
oc apply -f configmap.yaml -f models-pvc.yaml

# 2. Fetch the models (about 1.7G, a few minutes)
oc apply -f models-download-job.yaml
oc wait --for=condition=complete job/download-models --timeout=30m
oc logs job/download-models | tail

# 3. The database
oc apply -f qdrant-service.yaml -f qdrant-statefulset.yaml
oc rollout status statefulset/qdrant

# 4. Build the app image (see below)
oc apply -f imagestream.yaml -f buildconfig.yaml
./build.sh

# 5. The API, after putting the SHA from build.sh into app-deployment.yaml
oc apply -f app-deployment.yaml -f app-service.yaml
oc rollout status deployment/elog-semantic-search
```

The first rollout is slow. Both models are read off CephFS before uvicorn binds
its port, so the startup probe allows five minutes.

## Building

```bash
./build.sh
```

It refuses to run on a dirty working tree or an unpushed commit, because the
cluster clones from GitHub rather than from your laptop: a build of code you
have not pushed would be tagged with a commit that does not contain it.

Images land on `:latest` and `build.sh` immediately tags them with the short
commit SHA. OKD cannot put the SHA in the BuildConfig output itself, so the tag
is applied afterwards. Put that SHA in `app-deployment.yaml`. Nothing should
ever deploy `:latest`, or a pod restart could quietly change versions.

Old images accumulate, since every SHA tag keeps its image referenced forever.
Layers are shared, so a code-only rebuild costs a few MB while a change to
`requirements.txt` costs about 1.5G. `oc get istag` lists them and
`oc delete istag elog-semantic-search:<sha>` drops one.

## Loading the index

Still undecided, and nothing below is tested yet. The collection
`Elog_collection_1024_METADATA` is about 593M locally. Two ways to get it in:

- Copy the existing index up from the laptop. Qdrant must not be running while
  its files are replaced, which means scaling the StatefulSet to zero and
  mounting `storage-qdrant-0` from a temporary pod, or using Qdrant's snapshot
  API against a port-forward.
- Re-run `run_update.py` against the in-cluster Qdrant through a port-forward.
  No copying, but it re-embeds everything and needs the ECL credentials.

## Reaching the API

```bash
oc port-forward svc/elog-semantic-search 8000:8000
curl -s localhost:8000/health
```

OpenWebUI keeps pointing at `http://localhost:8000`, exactly as it does with
docker compose. The port-forward runs in the foreground and stops when the
terminal closes or the login expires.

## Still to do

- **Index updates on a schedule.** A CronJob running `run_update.py`. It needs
  the ECL password, which should not be a hand-made `oc create secret`: that
  credential belongs to SBND, has no expiry, and will be needed
  indefinitely. This cluster has the External Secrets Operator installed and
  `oc auth can-i create externalsecrets` says yes, so the password can live in
  Vault and be synced into a Secret automatically. What is not known yet is
  the Vault server address, whether a ClusterSecretStore already exists
  (reading them is forbidden at this access level), and who can write the
  value into Vault. Ask the okdprod1 admins.
