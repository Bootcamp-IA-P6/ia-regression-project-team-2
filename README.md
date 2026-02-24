# ia-regression-project-team-2


## 🔹 Open the app with Docker

Build docker image:
```bash
docker build -t house_price_model:v1 .
```
Run docker image:
```bash
docker run -p 8000:8000 house_price_model:v1 
```

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

## 🔹 Windows (PowerShell or CMD)

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

## To train the model
```bash
uv run model/model_training.py
```

## To run the app
```bash
uv run streamlit run streamlit/app.py

```