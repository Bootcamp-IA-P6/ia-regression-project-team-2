# ia-regression-project-team-2


## 🔹 macOS / Linux


#### Create virtual environment
```bash
uv venv
```

#### Activate virtual environment
```bash
source .venv/bin/activate
```

#### Install dependencies from pyproject.toml
```bash
uv sync
```

## 🔹 Windows (PowerShell o CMD)

#### Create virtual environment
```bash
uv venv
```

#### Activate virtual environment
```bash
source .venv/Scripts/activate
```

#### Install dependencies from pyproject.toml
```bash
uv sync
```

## 🔹 To train the model
```bash
uv run model_training.py
```

## 🔹 To run the app
```bash
uv run streamlit run app.py
```