# TriIntel — AI-Powered Career Intelligence Platform

> A machine learning system that recommends career roles, identifies skill gaps, generates personalised learning roadmaps, and surfaces live job market trends — built end-to-end from data collection to deployed web application.

🌐 **Live Demo:** [tri-intel-ai.vercel.app](https://tri-intel-ai.vercel.app)

---

## Overview

TriIntel combines three independently trained ML classifiers into a unified career intelligence engine. A user inputs their background — education, experience level, and current skills — and the platform returns a ranked list of suitable tech roles, highlights which skills they're missing, and generates a step-by-step roadmap to bridge those gaps, backed by real 2025–26 job market demand data.

The system covers **27 tech roles** and tracks **192 distinct skills** across the technology landscape, trained on a dataset of **50,000+ records**.

---

## Features

| Feature | Description |
|---|---|
| **Role Recommender** | Predicts best-fit tech careers from user profile inputs |
| **Skill Gap Analyser** | Identifies missing skills relative to target role requirements |
| **Roadmap Generator** | Produces a structured, step-by-step learning path to close skill gaps |
| **Market Trend Dashboard** | Displays role demand scores, salary ranges, and hiring trends for 2025–26 |

---

## Tech Stack

**Machine Learning & Data**
- Python, Scikit-learn — model training and evaluation
- Pandas, NumPy — data processing and feature engineering
- Matplotlib, Seaborn — EDA and visualisation
- Jupyter Notebooks — model development and experimentation

**Backend**
- Python (Flask / FastAPI) — REST API serving model predictions

**Frontend**
- HTML, CSS, JavaScript — UI for user input and results display
- Deployed on **Vercel**
---

## Model Architecture

Three separate classifiers handle distinct prediction tasks:

```
User Profile Input
       │
       ├──► Role Classifier        → Ranked role recommendations
       │
       ├──► Skill Gap Model        → Missing skills per target role
       │
       └──► Roadmap Generator      → Ordered learning milestones
```

Each model was trained, validated, and evaluated independently using cross-validation. Feature engineering was applied to encode education level, experience bands, and multi-label skill vectors.

---

## Dataset

- **Size:** 50,000+ job profile records
- **Scope:** 27 tech roles, 192 tracked skills
- **Features include:** education level, years of experience, current skill set, target role, salary band, and geographic market
- **Source:** Aggregated from publicly available job market datasets and career survey data (2024–25)

---

## Project Structure

```
TriIntel-AI/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD pipeline
├── backend/                # Python API (Flask/FastAPI)
│   ├── app.py
│   └── models/             # Serialised trained models (.pkl)
├── data/                   # Raw and processed datasets
├── frontend/               # HTML/CSS/JS web interface
├── notebooks/              # Jupyter notebooks for EDA + model training
│   ├── 01_eda.ipynb
│   ├── 02_role_classifier.ipynb
│   ├── 03_skill_gap_model.ipynb
│   └── 04_roadmap_generator.ipynb
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Clone the repository
```bash
git clone https://github.com/siri-004/TriIntel-AI.git
cd TriIntel-AI
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the backend
```bash
cd backend
python app.py
```

### Open the frontend
Open `frontend/index.html` in your browser, or run a local server:
```bash
cd frontend
python -m http.server 8080
```

The app will be available at `http://localhost:8080`.

---

## Notebooks

The `notebooks/` directory contains the full ML development pipeline:

1. **EDA** — distribution of roles, skill frequencies, and market demand
2. **Role Classifier** — feature engineering, model selection, cross-validation
3. **Skill Gap Model** — multi-label classification for missing skill prediction
4. **Roadmap Generator** — sequence modelling for learning path output

---
## Results

| Model | Metric | Score |
|---|---|---|
| Role Recommender | Accuracy | 99.7% |
| Skill Gap Classifier | F1 Score (Micro) | 0.905 |
| Skill Gap Classifier | F1 Score (Macro) | 0.907 |
| Demand Score Predictor | R² | 1.00 |
| Demand Score Predictor | RMSE | 0.000 |

> **Note:** Dataset is synthetically generated (50,000 records, 27 roles, 40 skills),
> which accounts for the near-perfect scores. Performance on real-world data
> would be expected in the 80–90% range.

---

## License

This project is open source and available under the [MIT License](LICENSE).
