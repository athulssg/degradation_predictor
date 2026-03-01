# SFSG AI Tool

AI Assessment Tool for Forensic Sample Analysis, built for Science For Social Good CIC.

---

## Running with Docker (recommended)

### 1. Build the image

```bash
docker build -t sfsg-ai-tool .
```

### 2. Run the container

```bash
docker run -p 8501:8501 sfsg-ai-tool
```

Open your browser at **http://localhost:8501**

### Persist the trained model between runs

The model (`model.pkl`) and accumulated training data (`training_data.csv`) are written inside the container and lost when it stops. To keep them across runs, mount a local directory:

```bash
docker run -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  -e MODEL_PATH=/app/data/model.pkl \
  -e TRAINING_DATA_PATH=/app/data/training_data.csv \
  sfsg-ai-tool
```

> **Windows (Command Prompt):** replace `$(pwd)` with `%cd%`
> **Windows (PowerShell):** replace `$(pwd)` with `${PWD}`

Or use the simpler bind-mount approach — mount the whole working directory:

```bash
docker run -p 8501:8501 -v "$(pwd):/app" sfsg-ai-tool
```

### Stop the container

```bash
docker ps                        # find the container ID
docker stop <container-id>
```

---

## Running locally (without Docker)

```bash
pip install -r requirements.txt Pillow
python -m streamlit run main.py
```

Open your browser at **http://localhost:8501**

---

## Dataset format

Upload a CSV with these columns:

| Column | Description |
|---|---|
| `Sample_ID` | Unique sample identifier |
| `Target_Name` | One of: `Control`, `Autosom 1`, `Autosom 2`, `Male` |
| `Quantity` | Numeric qPCR quantity value |

A sample file (`SFSG_Dataset.csv`) is included in the repository.
