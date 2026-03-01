# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SFSG AI Tool is a Streamlit web application for forensic DNA sample quality assessment, built for SANE (Science For Social Good CIC). It analyzes qPCR-derived quantity data to classify sample degradation and calculate M:F ratios.

## Running the App

```bash
pip install -r requirements.txt
streamlit run main.py
```

The app runs on port 8501. In the devcontainer, it starts automatically via `postAttachCommand` with CORS and XSRF protection disabled.

## Dataset Format

`SFSG_Dataset.csv` shows the expected CSV schema:

| Column | Description |
|---|---|
| `Sample_ID` | Unique sample identifier |
| `Target_Name` | One of: `Control`, `Autosom 1`, `Autosom 2`, `Male` |
| `Quantity` | Numeric qPCR quantity value |
| `Ct_Value`, `Instrument_ID`, `Run_Date`, `Run`, `Control_Ct_Flag` | Supplementary metadata |

Each sample has multiple rows — one per `Target_Name`. The app pivots this long-format data by `Sample_ID` internally.

## Architecture (`main.py`)

Single-file Streamlit app with these logical layers:

**Data layer**
- `load_csv()` — loads and validates a CSV; checks `Quantity` is numeric
- `check_columns()` — asserts required columns exist, halts on failure
- `prepare_data()` — pivots `Autosom 1` and `Autosom 2` rows by `Sample_ID`, computes `Degradation_Index = Quantity_Autosom_2 / Quantity_Autosom_1`
- `calculate_mf_ratio()` — merges `Male` and `Autosom 2` rows by `Sample_ID`, computes M:F ratio string

**Assessment layer**
- `classify_degradation(index)` — thresholds: `<1` → Ready, `1–10` → Degraded, `>10` → Significant Degradation
- `assess_sample(df, sample_id)` — looks up a sample in prepared data and renders its degradation verdict

**ML layer**
- `train_model(df)` — trains a `RandomForestClassifier` (scikit-learn) on `[Quantity_Autosom_1, Quantity_Autosom_2, Degradation_Index]` features with an 80/20 train/test split; outputs `classification_report` and accuracy in the UI

**UI layer**
- `main()` — sidebar navigation (`streamlit_option_menu`) with two pages:
  - **About Us**: static info page with external link
  - **AI Sample Assessment Tool**: file upload → data preview → model training → M:F ratio → per-sample assessment

## Key Dependencies

- `streamlit` — UI framework
- `pandas` / `numpy` — data manipulation
- `scikit-learn` — `RandomForestClassifier`
- `streamlit_option_menu` — sidebar navigation component
- `Pillow` — loads `SFSG_Logo.png` for sidebar display



## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.