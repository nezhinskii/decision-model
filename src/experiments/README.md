# Experiment records and MLflow

`__main__.py` exposes `upload` and `export` commands.
`tracking.py` owns local MLflow and background/manual remote synchronization.
`export.py` imports and analyzes runs. `provenance.py` describes source data,
feature caches and the code/environment. Both training and evaluation use these
modules; this package imports neither training nor evaluation.

## Commands

```bash
uv run python -m src.experiments upload --run artifacts/runs/RUN_NAME --remote-uri REMOTE_URI
uv run python -m src.experiments export --source-uri REMOTE_URI --run-id RUN_ID
```

`upload` retries synchronization of an existing local run; it does not run a model.
Training/evaluation with `--remote-uri` still upload automatically in the background.
`export` retains its existing behavior: optional import into local MLflow plus an
analysis JSON report. Series selection and all existing export flags remain valid.

## Local UI

```bash
uv run mlflow server --backend-store-uri sqlite:///artifacts/mlflow/tracking.db \
  --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. Metrics, parameters and small reports are in local
MLflow; model/cache files are not uploaded as artifacts. `local_directory` and
`best.json` identify local weights. Local paths shown remotely are references,
not downloadable model files. Keep the database and its adjacent artifacts folder
together when moving the tracking store.

## DagsHub

Create a public repository and use its MLflow endpoint:
`https://dagshub.com/OWNER/REPOSITORY.mlflow`. Set credentials outside the repository:
`MLFLOW_TRACKING_USERNAME` is your username and `MLFLOW_TRACKING_PASSWORD` your token.
You can also copy `.env.example` to `.env` in the project root and fill in these
two values. Tracking loads it automatically on Linux and Windows; existing shell
variables take precedence. `.env` is ignored by Git and is never a run artifact.
Do not put credentials into the URI or saved CLI config.

Pass `--remote-uri URL` to training or evaluation for background synchronization after validation/end,
or retry independently:

```bash
uv run python -m src.experiments upload \
  --run artifacts/runs/pointer_seed42 \
  --remote-uri https://dagshub.com/OWNER/REPOSITORY.mlflow
```

Sync mirrors local parameters, tags, full metric histories and an explicit
allowlist of JSON reports and `predictions.jsonl`. A source-local-run tag links distinct server IDs.
Existing metric points are deduplicated on retry; only one sync process per run
should be active. Network failures warn without discarding local results. Sync is
performed by one background sender. Training requests do not wait for network
responses, and repeated requests coalesce while a transfer is running. Failures
retry after 60 seconds (or when new data is requested). Training requests synchronization at validation intervals; evaluation requests it
at its configured question logging intervals. Shutdown requests a final snapshot and
waits until the sender finishes, including any preceding in-flight transfer.
There is no five-second cutoff. A network failure still warns and leaves results
locally for the retry CLI; shutdown does not retry an unavailable server forever.
The terminal shows connection, metadata, metric history, upload and status phases,
plus a waiting message every 15 seconds during final synchronization. Progress
counts work items, not bytes. Parameters/tags and metric points are batched;
unchanged JSON artifacts are skipped by content hash. Remote acknowledgment tags
record each exact uploaded history prefix, so normal incremental sync does not
download old histories. Interrupted writes and inserted/replayed points trigger
reconciliation before retrying, preserving the full history without duplicates.
Local MLflow is the durable source for any unsent results. The standalone retry
CLI remains synchronous. Do not run it concurrently with the background sender
for the same run.
Model weights and caches are never in the upload allowlist.

Resume can replay steps newer than the last snapshot; their earlier metrics remain
in MLflow history rather than being silently deleted. The checkpoint cursor,
not the last logged metric, determines which training state is restored.

## Import and analysis reports

The following single-line command works in Bash and PowerShell. Replace `RUN_ID`
with the remote MLflow run ID (run details in DagsHub, or `remote_run_id` in the
training directory's `tracking.json`). Credentials load from the project `.env`.

```text
uv run --no-sync python -m src.experiments export --source-uri https://dagshub.com/OWNER/REPOSITORY.mlflow --run-id RUN_ID --output artifacts/exports/run.json
```

By default this imports the selected run into `artifacts/mlflow/tracking.db`
and writes an analysis report. The local copy contains full metric histories,
parameters, source tags, JSON reports and prediction artifacts. Its start/end times and
status match the source snapshot. Repeating the command reuses the imported run
and adds missing points. Use only one importer at a time for the same run.
Imports are tagged `imported_view_only=true`: they are for browsing, not training
resume. Local training directories are not changed. Model weights, feature caches
and arbitrary source artifacts are not downloaded. Source parent-run IDs and
machine-local directory tags are not reused as local MLflow links.

The report contains full overall metric curves as `[step, value]` pairs, endpoint
and extrema summaries, configuration, compact provenance and data/cache identity.
Category/target breakdowns contain their latest values and values at the selected
best step. It omits per-point timestamps and rounds numbers to six significant
digits. Replayed steps use their latest timestamp in the report; MLflow retains
every original point. Nonfinite values are explicit strings. Nothing in training
reads this report: it can always be regenerated from MLflow.

Optional flags:

- `--no-import-local`: write only the analysis report.
- `--max-points 32`: explicitly thin each curve, retaining endpoints and extrema.
- `--local-db PATH`: choose the destination/read-default local MLflow database.

Without `--source-uri`, the command reads the local database and does not clone
the run. Both import and report cover one selected run; separately tracked test
evaluations must be exported by their own run IDs. A linked ID alone does not
include their metrics.

Attach the output JSON for analysis. Open imported runs using the usual local
MLflow server command above. Prediction JSONL files are separate artifacts, not an alternative metric-history store.

To export a whole series, use an inclusive start-time boundary:

```text
uv run --no-sync python -m src.experiments export --source-uri https://dagshub.com/OWNER/REPOSITORY.mlflow --since-run-id FIRST_RUN_ID --output artifacts/exports/series
```

This selects the anchor run and every nondeleted run in its MLflow experiment
whose start time is at least the anchor's, including running/failed runs and
timestamp ties. IDs themselves are not ordered. Other experiments are excluded.
All result pages are read. `--output` is a directory in this mode: each report is
named `<run_id>.json`, and `index.json` maps IDs to names, status, best results and
report paths. Import into local MLflow remains enabled by default; add
`--no-import-local` for JSON reports only. Individual failures are recorded in the
index and remaining runs continue; the command exits nonzero if any failed.

