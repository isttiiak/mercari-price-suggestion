# Mercari Price Suggestion Challenge
### BRACU-SICIP Certificate in Data Science — Capstone Project
**Brac University | ITS 507 Data Science**

---

## Project Overview

Mercari is Japan's largest community-powered shopping app where individuals sell new and used items. A key challenge for sellers is determining the right price for their listings — overpricing leads to unsold items, underpricing leaves money on the table.

This project builds an **end-to-end machine learning pipeline** that automatically suggests a fair product price based on the listing's title, description, category, brand, condition, and shipping information.

> **Problem Type:** Large-scale regression with NLP  
> **Dataset:** [Kaggle Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge)  
> **Evaluation Metric:** RMSLE (Root Mean Squared Logarithmic Error)

---

## Results

| Model | Split | RMSLE | MAE (USD) |
|---|---|---|---|
| Ridge Regression | 70/30 | 0.6246 | $14.17 |
| Ridge Regression | 80/20 | 0.6248 | $14.16 |
| LightGBM | 70/30 | 0.4660 | $10.48 |
| **LightGBM (Selected)** | **80/20** | **0.4650** | **$10.44** |

**Best Model: LightGBM (80/20 split)** — 25.5% improvement over Ridge Regression baseline.

---

## Project Structure

```
mercari-price-suggestion/
├── api/                          # FastAPI deployment
│   ├── main.py                   # API endpoints
│   ├── preprocess.py             # Inference preprocessing pipeline
│   ├── requirements.txt          # API dependencies
│   └── model/                    # Serialized model + encoders
│       ├── lgbm_model.pkl
│       ├── tfidf_name.pkl
│       ├── tfidf_desc.pkl
│       ├── le_cat1.pkl
│       ├── le_cat2.pkl
│       ├── le_cat3.pkl
│       └── le_brand.pkl
├── data/
│   ├── raw/                      # train.tsv, test.tsv (not committed)
│   └── processed/                # Cleaned data, feature matrix, encoders
├── models/                       # Trained model files
│   ├── lgbm_model.pkl
│   └── ridge_model.pkl
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   # Data cleaning and transformation
│   ├── 03_feature_engineering.ipynb  # Feature creation and selection
│   └── 04_model_training.ipynb  # Model training and evaluation
├── Dockerfile                    # Docker build instructions
├── docker-compose.yml            # Container orchestration
├── requirements.txt              # Project dependencies
└── README.md
```

---

## Dataset

**Source:** [Kaggle Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge)

| File | Size | Description |
|---|---|---|
| `train.tsv` | ~600MB | 1,482,535 product listings with prices |
| `test.tsv` | ~1.4GB | 3,460,725 listings without prices |

### Features

| Column | Type | Description |
|---|---|---|
| `name` | Text | Product listing title |
| `item_condition_id` | Integer (1–5) | Item condition (1=New, 5=Poor) |
| `category_name` | Text | Hierarchical category (e.g. `Women/Tops/T-Shirts`) |
| `brand_name` | Text | Brand name (42% missing) |
| `price` | Float | **Target variable** — sale price in USD |
| `shipping` | Binary | 1 = seller pays shipping, 0 = buyer pays |
| `item_description` | Text | Full item description |

---

## Methodology

### Phase I — Data Preprocessing and Feature Engineering

**EDA Findings:**
- 1,482,535 listings, price range $0–$2,009
- Price is heavily right-skewed (mean $26.74, median $17.00)
- `brand_name` has 42.68% missing values
- Branded items have 43% higher median price ($20) vs non-branded ($14)

**Preprocessing Steps:**

| Step | Action | Reason |
|---|---|---|
| Remove invalid rows | Drop 874 rows where price = 0 | Invalid listings — no real seller prices at $0 |
| Log transform target | `log_price = log1p(price)` | Normalize skewed distribution, align with RMSLE metric |
| Handle missing brands | Create `is_branded` feature, fill nulls with `"no brand"` | Preserve all rows; brand presence is itself a strong price signal |
| Handle missing categories | Fill with `"missing/missing/missing"` | Maintain split-ability for category hierarchy |
| Split categories | `cat1`, `cat2`, `cat3` from `category_name` | Expose 3-level hierarchy as separate model features |
| Clean text | Lowercase, remove `[rm]` tags, collapse spaces | Consistent tokenization for TF-IDF |

**Feature Engineering:**

| Feature Group | Technique | Output Dimensions |
|---|---|---|
| Product name | TF-IDF (50k features, bigrams, min_df=3) | 50,000 |
| Item description | TF-IDF (50k features, bigrams, min_df=3) | 50,000 |
| Categories (cat1/2/3) | Label Encoding | 3 |
| Brand name | Label Encoding | 1 |
| Item condition, shipping | Direct numeric | 2 |
| Name length, desc length | Word count | 2 |
| is_branded | Binary (0/1) | 1 |
| **Total** | | **100,009** |

**Feature Selection (Ridge-based):**

| Feature | Coefficient | Importance |
|---|---|---|
| shipping | 0.329 | Strongest numeric signal |
| is_branded | 0.291 | Second strongest — validates instructor suggestion |
| item_condition_id | 0.061 | Medium signal |
| name_len | 0.022 | Weak but non-zero |
| desc_len | 0.002 | Very weak |

### Phase II — Model Training

Two models trained and compared across two train/validation splits:

**Ridge Regression (Baseline)**
- Linear model with L2 regularization
- `solver='sparse_cg'` optimized for sparse TF-IDF matrices
- Fast (< 2s) but limited to linear feature relationships

**LightGBM (Primary Model)**
- Gradient boosting with 1,000 trees
- Captures non-linear feature interactions (e.g. brand × condition × category)
- `subsample=0.8`, `colsample_bytree=0.8` for overfitting prevention
- Selected for deployment: RMSLE 0.4650 (25.5% better than Ridge)

### Phase III — Deployment

- FastAPI REST API with `/predict` endpoint
- Same preprocessing pipeline as Phase I applied at inference time
- Dockerized and published to DockerHub

---

## API Usage

### Run with Docker

```bash
# Pull from DockerHub
docker pull isttiiak/mercari-price-suggestion

# Run the container
docker run -p 8000:8000 isttiiak/mercari-price-suggestion
```

### Run with Docker Compose

```bash
docker-compose up
```

API will be available at `http://localhost:8000`

### Interactive API Docs

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

### Health Check

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "ok",
  "message": "Mercari Price Suggestion API is running"
}
```

### Predict Price

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Nike Air Max White Size 10",
    "category_name": "Men/Shoes/Athletic",
    "brand_name": "Nike",
    "item_condition_id": 2,
    "shipping": 0,
    "item_description": "Worn twice only, great condition, no scratches or defects"
  }'
```

**Response:**
```json
{
  "predicted_price_usd": 45.80,
  "log_price": 3.8241,
  "message": "Price predicted successfully"
}
```

### Input Schema

| Field | Type | Required | Values | Example |
|---|---|---|---|---|
| `name` | string | Yes | Any text | `"Nike Air Max White"` |
| `category_name` | string | Yes | `L1/L2/L3` format | `"Men/Shoes/Athletic"` |
| `brand_name` | string | No | Brand or `"no brand"` | `"Nike"` |
| `item_condition_id` | integer | Yes | 1–5 | `2` |
| `shipping` | integer | Yes | 0 or 1 | `0` |
| `item_description` | string | No | Any text | `"Worn twice..."` |

---

## DockerHub

**Image:** `isttiiak/mercari-price-suggestion`  
**Link:** [https://hub.docker.com/r/isttiiak/mercari-price-suggestion](https://hub.docker.com/r/isttiiak/mercari-price-suggestion)

```bash
docker pull isttiiak/mercari-price-suggestion
docker run -p 8000:8000 isttiiak/mercari-price-suggestion
```

---

## Setup — Run Locally Without Docker

### Prerequisites

- Python 3.12
- pyenv (recommended)
- Homebrew (macOS)

### Installation

```bash
# Clone the repository
git clone https://github.com/isttiiak/mercari-price-suggestion.git
cd mercari-price-suggestion

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# macOS — install OpenMP for LightGBM
brew install libomp
```

### Download Dataset

Download `train.tsv.7z` and `test.tsv.7z` from [Kaggle](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data) and extract to `data/raw/`.

```bash
brew install p7zip
7z x train.tsv.7z -o data/raw/
7z x test.tsv.7z -o data/raw/
```

### Run Notebooks in Order

```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_model_training.ipynb
```

### Run API Locally

```bash
cd api
uvicorn main:app --reload --port 8000
```

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Data Processing | pandas 3.0.3, numpy 2.4.6 |
| Machine Learning | scikit-learn 1.9.0, LightGBM 4.6.0 |
| NLP / Features | TF-IDF (scikit-learn), scipy sparse matrices |
| API Framework | FastAPI 0.136.3, uvicorn 0.49.0 |
| Serialization | joblib 1.5.3 |
| Containerization | Docker, Docker Compose |
| Registry | DockerHub |
| IDE | VS Code, JupyterLab |
| Version Control | Git, GitHub |

---

## Key Design Decisions

**Why log1p on price?**
Price is heavily right-skewed (median $17, max $2,009). Log transformation normalizes the distribution and aligns with RMSLE — minimizing RMSE on log-transformed targets is mathematically equivalent to minimizing RMSLE.

**Why `is_branded` instead of just filling missing brand names?**
Instructor suggestion, validated by data: branded items have 43% higher median price. A binary `is_branded` feature gives the model an explicit signal about brand presence, whereas filling nulls with a string only provides an implicit one. Ridge coefficient confirms `is_branded` (0.291) is the second strongest numeric feature.

**Why LightGBM over Ridge?**
LightGBM captures non-linear feature interactions that are inherent in price prediction (brand × condition × category effects). It achieves RMSLE 0.4650 vs Ridge's 0.6246 — a 25.5% improvement.

**Why 80/20 split for LightGBM?**
Gradient boosting models benefit more from additional training data. 80/20 provides 148,166 extra training rows, yielding a marginal but consistent improvement (0.4660 → 0.4650 RMSLE).

---

## Business Impact

| Metric | Value |
|---|---|
| Mean Absolute Error | $10.44 per prediction |
| RMSLE | 0.4650 |
| Training data coverage | 1.48M real marketplace listings |
| API response time | < 500ms per prediction |

This system reduces pricing uncertainty for new sellers, improves listing conversion rates, and creates a more competitive, fairly-priced marketplace for buyers.

---

## Author

**Istiak Islam**  
Certificate in Data Science — BRACU-SICIP  
Brac University, Dhaka, Bangladesh  
GitHub: [@isttiiak](https://github.com/isttiiak)