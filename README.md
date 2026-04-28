# Peer Effects and Adoption - Brown Pitch

This orphan branch is a clean Brown-pitch version of the ESB peer-effects project. It keeps raw source materials, a minimal data-preparation pipeline, Stata estimation scripts, generated-output folders, and documentation for the evolving paper and slides.

## Project Spine

1. Build cleaned analysis datasets from `1_Data/Raw` into `1_Data/Cleaned`.
2. Run preliminary descriptive checks and figures.
3. Estimate the main hazard-panel specifications in Stata.
4. Write results into `3_Output` and maintain the argument in `Documentation`.

## Stata

The local Stata executable is expected at:

```text
C:\Stata16\StataMP-64.exe
```

Use `2_Scripts/run_stata_estimation.bat` to run the main estimation sequence.
