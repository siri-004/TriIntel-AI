# ─────────────────────────────────────────────────────────────────────────────
# TriIntel — FastAPI Backend
# File location: triintel/backend/main.py
#
# How to run:
#   1. Open terminal inside triintel/backend/
#   2. Activate venv:  venv\Scripts\activate  (Windows)
#                      source venv/bin/activate (Mac/Linux)
#   3. Install deps:   pip install fastapi uvicorn pandas scikit-learn joblib python-dotenv
#   4. Start server:   uvicorn main:app --reload
#   5. Open browser:   http://localhost:8000/docs
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib, json, os, numpy as np, pandas as pd
from pathlib import Path

app = FastAPI(title="TriIntel API", version="1.0.0")

# ── Allow the React frontend (any origin) to call this API ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load all models saved by the Jupyter notebook ────────────────────────────
BASE = Path(__file__).parent / "models"

# The 3 ML models
model_demand = joblib.load(BASE / "demand_prediction_model.pkl")   # Model 1: demand score
model_rec    = joblib.load(BASE / "recommendation_model.pkl")      # Model 2: role recommendation
clf_skills   = joblib.load(BASE / "skill_gap_model.pkl")           # Model 3: skill classification

# Encoders & transformers (saved in notebook Cell 3 & 5)
le_role       = joblib.load(BASE / "le_role.pkl")
le_edu        = joblib.load(BASE / "le_edu.pkl")
le_domain     = joblib.load(BASE / "le_domain.pkl")
mlb_user      = joblib.load(BASE / "mlb_skills.pkl")          # user skill binarizer
mlb_role      = joblib.load(BASE / "mlb_role_skills.pkl")     # role skill binarizer
feat_cols     = joblib.load(BASE / "rec_feature_columns.pkl") # exact column order for model_rec
top_skill_labels = joblib.load(BASE / "top_skill_labels.pkl") # top 40 skill names for Model 3

# Static JSON data files (role info, trending roles, etc.)
with open(BASE / "role_skills_map.json")  as f: role_skills_map  = json.load(f)
with open(BASE / "role_info_map.json")    as f: role_info_map    = json.load(f)
with open(BASE / "trending_roles.json")   as f: trending_roles   = json.load(f)
with open(BASE / "top_skills.json")       as f: top_skills_data  = json.load(f)
with open(BASE / "all_roles.json")        as f: all_roles        = json.load(f)
with open(BASE / "all_skills_list.json")  as f: all_skills_list  = json.load(f)

print("✅ All models loaded successfully.")

# ── Learning resource links for each skill ───────────────────────────────────
LEARNING_RESOURCES = {
    "Python":              "https://realpython.com/",
    "SQL":                 "https://www.sqltutorial.org/",
    "Machine Learning":    "https://www.coursera.org/learn/machine-learning",
    "Deep Learning":       "https://fast.ai/",
    "TensorFlow":          "https://www.tensorflow.org/tutorials",
    "PyTorch":             "https://pytorch.org/tutorials/",
    "AWS":                 "https://aws.amazon.com/training/",
    "Azure":               "https://learn.microsoft.com/en-us/azure/",
    "Google Cloud":        "https://cloud.google.com/training",
    "Docker":              "https://docs.docker.com/get-started/",
    "Kubernetes":          "https://kubernetes.io/docs/tutorials/",
    "Terraform":           "https://developer.hashicorp.com/terraform/tutorials",
    "React":               "https://react.dev/",
    "JavaScript":          "https://javascript.info/",
    "TypeScript":          "https://www.typescriptlang.org/docs/",
    "Java":                "https://dev.java/learn/",
    "Git":                 "https://git-scm.com/doc",
    "Linux":               "https://linuxjourney.com/",
    "Tableau":             "https://www.tableau.com/learn/training",
    "Power BI":            "https://learn.microsoft.com/en-us/power-bi/",
    "Excel":               "https://support.microsoft.com/en-us/excel",
    "NLP":                 "https://huggingface.co/learn/nlp-course/",
    "LLMs":                "https://huggingface.co/learn/",
    "LangChain":           "https://python.langchain.com/docs/get_started/",
    "Spark":               "https://spark.apache.org/docs/latest/",
    "Kafka":               "https://kafka.apache.org/documentation/",
    "Airflow":             "https://airflow.apache.org/docs/",
    "dbt":                 "https://docs.getdbt.com/docs/introduction",
    "Snowflake":           "https://docs.snowflake.com/",
    "Figma":               "https://help.figma.com/",
    "Networking":          "https://www.cisco.com/c/en/us/training-events/training-certifications/certifications/associate/ccna.html",
    "Statistics":          "https://www.khanacademy.org/math/statistics-probability",
    "Data Modeling":       "https://www.datacamp.com/courses/data-modeling",
    "Agile":               "https://www.atlassian.com/agile",
}

# ── Helper: parse pipe-separated skill list ───────────────────────────────────
def parse_pipe_list(s):
    if not s or pd.isna(s):
        return []
    return [x.strip() for x in str(s).split('|') if x.strip()]

# ── Helper: build feature vector for Model 2 (recommendation) ────────────────
def build_rec_feature_vector(user_skill_names: List[str], yoe: float,
                              edu: str, skill_gap: int, readiness: int,
                              demand_score: int = 85, growth_pct: int = 20):
    """
    Builds the exact same feature matrix the recommendation model was trained on.
    Matches the column order saved in rec_feature_columns.pkl
    """
    # Encode education (use same LabelEncoder from notebook)
    try:
        edu_enc_val = le_edu.transform([edu])[0]
    except Exception:
        edu_enc_val = 2  # default to Bachelor's index

    # Binarize user skills using the same MLB from notebook
    known_skills = [s for s in user_skill_names if s in mlb_user.classes_]
    skill_vec    = mlb_user.transform([known_skills])
    skill_df     = pd.DataFrame(skill_vec, columns=mlb_user.classes_)

    # Base numeric features (same 7 columns used in notebook Cell 5)
    base = pd.DataFrame([[
        yoe,
        len(user_skill_names),
        skill_gap,
        readiness,
        edu_enc_val,
        demand_score,
        growth_pct,
    ]], columns=[
        'years_of_experience',
        'num_user_skills',
        'skill_gap_count',
        'readiness_score',
        'edu_enc',
        'role_demand_score',
        'role_growth_pct',
    ])

    # Combine and align to training column order
    X = pd.concat([base.reset_index(drop=True), skill_df.reset_index(drop=True)], axis=1)
    for col in feat_cols:
        if col not in X.columns:
            X[col] = 0
    return X[feat_cols]

# ── Helper: build feature vector for Model 1 (demand prediction) ─────────────
def build_demand_feature_vector(role_name: str) -> pd.DataFrame:
    """
    Builds features for the demand prediction model.
    Matches the 6 columns used in notebook Cell 4.
    """
    info = role_info_map.get(role_name, {})

    risk_map   = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4, 'Very High': 5}
    trend_map  = {'Declining': 0, 'Stable': 1, 'Growing': 2, 'Surging': 3}
    future_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4, 'Extremely High': 5}

    return pd.DataFrame([[
        info.get('growth_pct', 0),
        risk_map.get(info.get('ai_replacement_risk', 'Moderate'), 3),
        trend_map.get(info.get('market_trend', 'Stable'), 1),
        future_map.get(info.get('future_demand', 'High'), 3),
        int(info.get('is_emerging', False)),
        int(info.get('is_at_risk', False)),
    ]], columns=[
        'role_growth_pct',
        'ai_risk_encoded',
        'market_trend_encoded',
        'future_demand_encoded',
        'is_emerging_role',
        'is_at_risk_role',
    ])

# ── Pydantic schemas (define the shape of request data) ──────────────────────
class UserSkill(BaseModel):
    skill: str
    level: str  # "Beginner" | "Intermediate" | "Advanced"

class RecommendRequest(BaseModel):
    years_of_experience: float
    education_level: str
    interest_domain: Optional[str] = ""
    target_role: Optional[str] = ""
    user_skills: List[UserSkill] = []

# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — open http://localhost:8000 to confirm server is running"""
    return {"status": "TriIntel API is running ✅", "version": "1.0.0"}


@app.get("/dashboard")
def get_dashboard():
    """
    Home page data: trending roles, top skills, summary counts.
    Called when the frontend loads.
    """
    return {
        "trending_roles": trending_roles,
        "top_skills":     top_skills_data[:15],
        "total_roles":    len(all_roles),
        "total_skills":   len(all_skills_list),
    }


@app.get("/roles")
def get_all_roles():
    """
    Returns all 27 job roles with their market info.
    Used by the Explore Roles page.
    """
    roles = []
    for role in all_roles:
        info = role_info_map.get(role, {})
        roles.append({
            "name":                role,
            "domain":              info.get("domain", ""),
            "demand_score":        info.get("demand_score", 0),
            "future_demand":       info.get("future_demand", ""),
            "ai_replacement_risk": info.get("ai_replacement_risk", ""),
            "growth_pct":          info.get("growth_pct", 0),
            "market_trend":        info.get("market_trend", ""),
            "avg_salary_usd":      info.get("avg_salary_usd", 0),
            "is_emerging":         info.get("is_emerging", False),
            "is_at_risk":          info.get("is_at_risk", False),
        })
    return {"roles": roles}


@app.get("/roles/{role_name}")
def get_role_detail(role_name: str):
    """
    Detailed info for one role: skills, salary, demand prediction, roadmap.
    Called when a user clicks on a role card.
    """
    if role_name not in role_info_map:
        raise HTTPException(status_code=404, detail=f"Role '{role_name}' not found")

    info   = role_info_map[role_name]
    skills = role_skills_map.get(role_name, {"core": [], "secondary": []})

    # Use Model 1 to predict demand score for this role
    X_demand = build_demand_feature_vector(role_name)
    predicted_demand = round(float(model_demand.predict(X_demand)[0]), 1)

    # Build learning roadmap
    all_role_skills = skills["core"] + skills["secondary"][:3]
    roadmap = [
        {
            "step":     i + 1,
            "skill":    sk,
            "resource": LEARNING_RESOURCES.get(sk, "https://www.coursera.org/"),
        }
        for i, sk in enumerate(all_role_skills)
    ]

    return {
        "role":             role_name,
        "info":             info,
        "predicted_demand": predicted_demand,
        "skills":           skills,
        "roadmap":          roadmap,
    }


@app.get("/skills")
def get_skills():
    """Returns the full list of skills known to the system."""
    return {"skills": all_skills_list}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    MODEL 2 — Role Recommendation.

    Input : user's skills, experience, education, optional target role
    Output: top 3 recommended roles + readiness score + missing skills + roadmap

    This is the main ML endpoint powering the 'Get My Roadmap' page.
    """
    user_skill_names = [s.skill for s in req.user_skills]
    skill_levels     = {s.skill: s.level for s in req.user_skills}

    # Get role context if user picked a target role
    if req.target_role and req.target_role in role_info_map:
        role_info   = role_info_map[req.target_role]
        demand_sc   = role_info["demand_score"]
        growth_p    = role_info["growth_pct"]
        core_skills = role_skills_map[req.target_role]["core"]
        sec_skills  = role_skills_map[req.target_role]["secondary"]
    else:
        demand_sc   = 85
        growth_p    = 20
        core_skills = []
        sec_skills  = []

    # Calculate skill gap
    missing      = [s for s in (core_skills + sec_skills[:3]) if s not in user_skill_names]
    gap          = len(missing)
    adv_count    = sum(1 for l in skill_levels.values() if l == "Advanced")
    readiness    = max(10, min(100, int(100 - gap * 8 + req.years_of_experience * 3 + adv_count * 5)))

    # Build feature vector and run Model 2
    X = build_rec_feature_vector(
        user_skill_names, req.years_of_experience,
        req.education_level, gap, readiness, demand_sc, growth_p
    )

    probs    = model_rec.predict_proba(X)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    recommended_roles = []
    for idx in top3_idx:
        rname = le_role.inverse_transform([idx])[0]
        rinfo = role_info_map.get(rname, {})
        recommended_roles.append({
            "role":                rname,
            "match_pct":           round(float(probs[idx]) * 100, 1),
            "domain":              rinfo.get("domain", ""),
            "demand_score":        rinfo.get("demand_score", 0),
            "future_demand":       rinfo.get("future_demand", ""),
            "avg_salary_usd":      rinfo.get("avg_salary_usd", 0),
            "ai_replacement_risk": rinfo.get("ai_replacement_risk", ""),
            "growth_pct":          rinfo.get("growth_pct", 0),
            "market_trend":        rinfo.get("market_trend", ""),
        })

    # Attach resource links to each missing skill
    missing_with_resources = [
        {"skill": s, "resource": LEARNING_RESOURCES.get(s, "https://www.coursera.org/")}
        for s in missing[:8]
    ]

    # Build step-by-step learning roadmap
    roadmap_steps = [
        {
            "step":           i + 1,
            "skill":          item["skill"],
            "level_to_reach": "Beginner" if i < 2 else ("Intermediate" if i < 4 else "Advanced"),
            "resource":       item["resource"],
        }
        for i, item in enumerate(missing_with_resources[:6])
    ]

    return {
        "recommended_roles":   recommended_roles,
        "readiness_score":     readiness,
        "skill_gap_count":     gap,
        "missing_skills":      missing_with_resources,
        "learning_roadmap":    roadmap_steps,
        "user_skill_count":    len(user_skill_names),
        "years_of_experience": req.years_of_experience,
    }


@app.post("/analyze-role")
def analyze_role(req: RecommendRequest):
    """
    MODEL 3 — Skill Gap Analysis.

    Input : user's skills + a specific target role
    Output: which core/secondary skills you have vs. missing + full roadmap

    Powers the 'Role Analysis' page.
    """
    if not req.target_role or req.target_role not in role_info_map:
        raise HTTPException(status_code=400, detail="Please provide a valid target_role")

    user_skill_names = [s.skill for s in req.user_skills]

    info      = role_info_map[req.target_role]
    skills    = role_skills_map[req.target_role]
    core      = skills["core"]
    secondary = skills["secondary"]

    have_core = [s for s in core      if s in user_skill_names]
    miss_core = [s for s in core      if s not in user_skill_names]
    have_sec  = [s for s in secondary if s in user_skill_names]
    miss_sec  = [s for s in secondary if s not in user_skill_names]

    total_needed = len(core) + len(secondary[:4])
    total_have   = len(have_core) + len(have_sec)
    readiness    = int((total_have / max(total_needed, 1)) * 100)

    roadmap = [
        {
            "step":     i + 1,
            "skill":    sk,
            "priority": "High" if sk in miss_core else "Medium",
            "resource": LEARNING_RESOURCES.get(sk, "https://www.coursera.org/"),
        }
        for i, sk in enumerate(miss_core + miss_sec[:4])
    ]

    return {
        "target_role":      req.target_role,
        "role_info":        info,
        "readiness_score":  readiness,
        "have_core_skills": have_core,
        "miss_core_skills": miss_core,
        "have_secondary":   have_sec,
        "miss_secondary":   miss_sec[:5],
        "roadmap":          roadmap,
    }


@app.get("/predict-demand/{role_name}")
def predict_demand(role_name: str):
    """
    MODEL 1 — Demand Prediction.

    Predicts the demand score for a given role using market signals.
    Used on the Market Trends page.
    """
    if role_name not in role_info_map:
        raise HTTPException(status_code=404, detail=f"Role '{role_name}' not found")

    X = build_demand_feature_vector(role_name)
    predicted_score = round(float(model_demand.predict(X)[0]), 1)
    info = role_info_map[role_name]

    return {
        "role":             role_name,
        "predicted_demand": predicted_score,
        "actual_demand":    info.get("demand_score", 0),
        "market_trend":     info.get("market_trend", ""),
        "growth_pct":       info.get("growth_pct", 0),
        "ai_risk":          info.get("ai_replacement_risk", ""),
    }