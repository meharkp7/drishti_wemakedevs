# DRISHTI

AI-assisted geospatial intelligence pipeline: imagery -> semantic segmentation ->
confidence-driven human verification -> temporal change detection -> GIS-ready export.

See docs/DRISHTI_Enhanced_Production_Workflow.md and
docs/DRISHTI_Enhanced_Implementation_Plan.md for the full architecture and build order.

## Repository layout

frontend/       React + map UI, WebGL overlay rendering, review UI, dashboard
backend/        API, services, background workers, schemas, core utilities
adapters/       Interaction Adapter - normalizes map/text/voice/sms/email into one Command
inference/      Model registry, preprocessing/postprocessing, device-aware Inference Router
geospatial/     Imagery abstraction, tiling, vectorization, change detection, validation
crowdsource/    Citizen/field-worker submission intake, dedup, geotag validation
agent/          LLM natural-language command layer (5-tool allowlist)
notifications/  Templated email/SMS dispatch on job-state transitions
storage/        Storage clients/adapters (object store + metadata DB)
tests/          unit / integration / e2e
configs/        Centralized configuration (no magic constants in code)
docs/           Architecture + implementation plan
scripts/        One-off/dev scripts
