{{- define "elog-search.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "elog-search.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "elog-search.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "elog-search.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "elog-search.selectorLabels" -}}
app.kubernetes.io/name: {{ include "elog-search.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Qdrant's in-cluster hostname. This is the one piece of config that is purely
about topology, and it has to be derived in exactly one place so the API and
the index-updater CronJob can never disagree about where the index is.
Mirrors the subchart's own fullname rule; helm/verify.sh asserts the two match.
*/}}
{{- define "elog-search.qdrantHost" -}}
{{- if .Values.qdrant.enabled -}}
{{- $q := .Values.qdrant -}}
{{- if $q.fullnameOverride -}}
{{- $q.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default "qdrant" $q.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- else -}}
{{- required "qdrant.enabled is false, so qdrantExternalHost must be set" .Values.qdrantExternalHost -}}
{{- end -}}
{{- end -}}
