import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from preprocess import load_artifacts, preprocess_input


# Global artifacts dictionary — loaded once at startup
artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model and encoders when API starts.
    This runs once — not on every request.
    """
    print("Loading model and encoders...")
    artifacts.update(load_artifacts())
    print("Model loaded successfully ✅")
    yield
    artifacts.clear()


# Initialize FastAPI app
app = FastAPI(
    title="Mercari Price Suggestion API",
    description="Predict product listing price using LightGBM",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Input / Output schemas ────────────────────────────────────────────────────

class ProductInput(BaseModel):
    name: str = Field(..., example="Nike Air Max White Size 10")
    category_name: str = Field(..., example="Men/Shoes/Athletic")
    brand_name: str = Field(default="no brand", example="Nike")
    item_condition_id: int = Field(..., ge=1, le=5, example=2)
    shipping: int = Field(..., ge=0, le=1, example=0)
    item_description: str = Field(
        default="no description",
        example="Worn twice, great condition, no scratches"
    )


class PriceOutput(BaseModel):
    predicted_price_usd: float
    log_price: float
    message: str


# ── HTML Frontend ─────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Mercari Price Suggester</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #ABC1C4;
      min-height: 100vh;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 2rem 1rem;
    }

    .wrapper {
      width: 100%;
      max-width: 580px;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    /* ── Header ── */
    .header {
      background: #126687;
      border-radius: 14px;
      padding: 1.5rem 1.75rem;
      color: #fff;
    }
    .header-tag {
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #7fa8cc;
      margin-bottom: 6px;
      font-weight: 600;
    }
    .header h1 {
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .header p {
      font-size: 13px;
      color: #a8bfcf;
      line-height: 1.5;
    }

    /* ── Card ── */
    .card {
      background: #F0E4D8;
      border-radius: 14px;
      padding: 1.75rem;
      border: 1px solid #e2e8f0;
    }
    .card-title {
      font-size: 13px;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 1.25rem;
    }

    /* ── Form ── */
    .form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .form-group.full { grid-column: 1 / -1; }

    label {
      font-size: 12px;
      font-weight: 600;
      color: #570D35;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .hint {
      font-size: 11px;
      color: #94a3b8;
      font-weight: 400;
      text-transform: none;
      letter-spacing: 0;
    }

    input[type="text"], input[type="number"], select, textarea {
      width: 100%;
      padding: 10px 12px;
      border: 1.5px solid #e2e8f0;
      border-radius: 8px;
      font-size: 14px;
      color: #1e293b;
      background: #fafafa;
      transition: border-color 0.15s, box-shadow 0.15s;
      outline: none;
      font-family: inherit;
    }
    input:focus, select:focus, textarea:focus {
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59,130,246,0.12);
      background: #fff;
    }
    textarea { resize: vertical; min-height: 80px; }

    /* ── Condition pills ── */
    .pill-group {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .pill {
      padding: 6px 12px;
      border-radius: 20px;
      border: 1.5px solid #e2e8f0;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      color: #64748b;
      background: #fafafa;
      transition: all 0.15s;
      user-select: none;
    }
    .pill:hover { border-color: #3b82f6; color: #3b82f6; }
    .pill.active {
      background: #CC0673;
      border-color: #1a2e4a;
      color: #fff;
    }
    input[name="item_condition_id"] { display: none; }

    /* ── Shipping toggle ── */
    .toggle-row {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .toggle {
      position: relative;
      width: 42px;
      height: 24px;
      flex-shrink: 0;
    }
    .toggle input { display: none; }
    .toggle-slider {
      position: absolute;
      inset: 0;
      background: #e2e8f0;
      border-radius: 12px;
      cursor: pointer;
      transition: background 0.2s;
    }
    .toggle-slider::after {
      content: '';
      position: absolute;
      width: 18px; height: 18px;
      left: 3px; top: 3px;
      background: #fff;
      border-radius: 50%;
      transition: transform 0.2s;
    }
    .toggle input:checked + .toggle-slider { background: #CC0673; }
    .toggle input:checked + .toggle-slider::after { transform: translateX(18px); }
    .toggle-label {
      font-size: 13px;
      color: #475569;
    }
    .toggle-label span { font-weight: 600; color: #CC0673; }

    /* ── Submit button ── */
    .btn {
      width: 100%;
      padding: 13px;
      background: #1a2e4a;
      color: #fff;
      border: none;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      margin-top: 1.25rem;
      transition: background 0.15s, transform 0.1s;
      letter-spacing: 0.02em;
    }
    .btn:hover { background: #8F0E54; }
    .btn:active { transform: scale(0.99); }
    .btn:disabled { background: #94a3b8; cursor: not-allowed; }

    /* ── Result ── */
    .result-label {
      font-size: 12px;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.5rem;
    }
    .price-display {
      font-size: 48px;
      font-weight: 800;
      color: #1a2e4a;
      letter-spacing: -1px;
      line-height: 1;
      margin-bottom: 0.75rem;
    }
    .price-display .currency {
      font-size: 28px;
      font-weight: 700;
      vertical-align: super;
      margin-right: 2px;
    }

    .meta-row {
      display: flex;
      gap: 1rem;
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid #f1f5f9;
    }
    .meta-item {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .meta-key {
      font-size: 11px;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .meta-val {
      font-size: 13px;
      color: #475569;
      font-weight: 500;
    }

    .error-msg {
      font-size: 14px;
      color: #dc2626;
      font-weight: 500;
    }

    /* ── Loading ── */
    .spinner {
      display: inline-block;
      width: 16px; height: 16px;
      border: 2px solid rgba(255,255,255,0.4);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Price tag badge ── */
    .tag-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #ecfdf5;
      color: #065f46;
      font-size: 12px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      margin-bottom: 1rem;
      border: 1px solid #a7f3d0;
    }
    .tag-badge .dot {
      width: 6px; height: 6px;
      background: #10b981;
      border-radius: 50%;
    }
    .error-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #fef2f2;
      color: #991b1b;
      font-size: 12px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      margin-bottom: 1rem;
      border: 1px solid #fca5a5;
    }
  </style>
</head>
<body>
<div class="wrapper">

  <!-- Header -->
  <div class="header">
    <div class="header-tag">BRACU · ITS 507 · Final Project  |  Istiak Islam</div>
    <h1>Mercari Price Suggester</h1>
    <p>LightGBM model trained on 1.4M Mercari listings. Enter product details below to get an instant price prediction.</p>
  </div>

  <!-- Form card -->
  <div class="card">
    <div class="card-title">Product Details</div>

    <div class="form-grid">

      <!-- Product name -->
      <div class="form-group full">
        <label>Product Name <span class="hint">— be specific (brand, size, color)</span></label>
        <input type="text" id="name" placeholder="e.g. Nike Air Max 90 White Size 10" />
      </div>

      <!-- Category -->
      <div class="form-group full">
        <label>Category <span class="hint">— format: Level1/Level2/Level3</span></label>
        <input type="text" id="category_name" placeholder="e.g. Men/Shoes/Athletic" />
      </div>

      <!-- Brand -->
      <div class="form-group">
        <label>Brand Name <span class="hint">— optional</span></label>
        <input type="text" id="brand_name" placeholder="e.g. Nike  (or leave blank)" />
      </div>

      <!-- Condition -->
      <div class="form-group">
        <label>Item Condition</label>
        <div class="pill-group" id="condition-pills">
          <div class="pill active" data-val="1">New</div>
          <div class="pill" data-val="2">Like New</div>
          <div class="pill" data-val="3">Good</div>
          <div class="pill" data-val="4">Fair</div>
          <div class="pill" data-val="5">Poor</div>
        </div>
        <input type="hidden" name="item_condition_id" id="item_condition_id" value="1" />
      </div>

      <!-- Shipping -->
      <div class="form-group full">
        <label>Shipping</label>
        <div class="toggle-row" style="margin-top: 4px;">
          <label class="toggle">
            <input type="checkbox" id="shipping" />
            <div class="toggle-slider"></div>
          </label>
          <span class="toggle-label">Seller pays shipping — <span id="shipping-label">OFF (buyer pays)</span></span>
        </div>
      </div>

      <!-- Description -->
      <div class="form-group full">
        <label>Item Description <span class="hint">— optional but improves accuracy</span></label>
        <textarea id="item_description" placeholder="Describe the item's condition, included accessories, measurements, etc."></textarea>
      </div>

    </div>

    <button class="btn" id="predict-btn" onclick="predict()">Get Price Suggestion</button>

    <!-- Result — lives inside the form card so it's always visible -->
    <div id="result-card" style="display:none; margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0;">
      <div id="result-inner"></div>
    </div>
  </div>

</div>

<script>
  // ── Condition pills ────────────────────────────────────────────────────────
  document.getElementById('condition-pills').addEventListener('click', (e) => {
    const pill = e.target.closest('.pill');
    if (!pill) return;
    document.querySelectorAll('#condition-pills .pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    document.getElementById('item_condition_id').value = pill.dataset.val;
  });

  // ── Shipping toggle label ─────────────────────────────────────────────────
  document.getElementById('shipping').addEventListener('change', function() {
    document.getElementById('shipping-label').textContent =
      this.checked ? 'ON (seller pays)' : 'OFF (buyer pays)';
  });

  // ── Predict ───────────────────────────────────────────────────────────────
  async function predict() {
    const btn = document.getElementById('predict-btn');
    const resultCard = document.getElementById('result-card');
    const resultInner = document.getElementById('result-inner');

    // Read values
    const name = document.getElementById('name').value.trim();
    const category_name = document.getElementById('category_name').value.trim();
    const brand_name = document.getElementById('brand_name').value.trim() || 'no brand';
    const item_condition_id = parseInt(document.getElementById('item_condition_id').value);
    const shipping = document.getElementById('shipping').checked ? 1 : 0;
    const item_description = document.getElementById('item_description').value.trim() || 'no description';

    // Basic validation
    if (!name) { alert('Please enter a product name.'); return; }
    if (!category_name) { alert('Please enter a category (e.g. Men/Shoes/Athletic).'); return; }

    // Loading state
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Predicting…';
    resultCard.style.display = 'none';

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name, category_name, brand_name,
          item_condition_id, shipping, item_description
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed.');
      }

      const price = data.predicted_price_usd.toFixed(2);
      const logP  = data.log_price.toFixed(4);

      // Condition label map
      const condMap = { 1:'New', 2:'Like New', 3:'Good', 4:'Fair', 5:'Poor' };

      resultInner.innerHTML = `
        <div class="tag-badge"><div class="dot"></div>Prediction ready</div>
        <div class="price-display"><span class="currency">$</span>${price}</div>
        <div class="meta-row">
          <div class="meta-item">
            <div class="meta-key">Log Price</div>
            <div class="meta-val">${logP}</div>
          </div>
          <div class="meta-item">
            <div class="meta-key">Condition</div>
            <div class="meta-val">${condMap[item_condition_id]}</div>
          </div>
          <div class="meta-item">
            <div class="meta-key">Shipping</div>
            <div class="meta-val">${shipping === 1 ? 'Seller pays' : 'Buyer pays'}</div>
          </div>
          <div class="meta-item">
            <div class="meta-key">Brand</div>
            <div class="meta-val">${brand_name === 'no brand' ? '—' : brand_name}</div>
          </div>
        </div>
      `;
      resultCard.style.display = 'block';
      resultCard.style.borderTop = '1px solid #e2e8f0';

    } catch (err) {
      resultInner.innerHTML = `
        <div class="error-badge">⚠ Error</div>
        <div class="error-msg">${err.message}</div>
      `;
      resultCard.style.display = 'block';
      resultCard.style.borderTop = '1px solid #fca5a5';
    }

    btn.disabled = false;
    btn.innerHTML = 'Get Price Suggestion';
  }

  // Allow Enter key to trigger prediction
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.tagName !== 'TEXTAREA') predict();
  });
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the interactive price suggestion frontend."""
    return HTML_PAGE


@app.get("/health")
def health_check():
    """Health check endpoint — confirms API is running."""
    return {"status": "ok", "message": "Mercari Price Suggestion API is running"}


@app.post("/predict", response_model=PriceOutput)
def predict_price(product: ProductInput):
    """
    Predict the price of a product listing.

    Applies the same preprocessing pipeline from Phase I,
    then uses the trained LightGBM model to predict log_price.
    Converts back to USD using np.expm1().
    """
    try:
        X = preprocess_input(
            name=product.name,
            category_name=product.category_name,
            brand_name=product.brand_name,
            item_condition_id=product.item_condition_id,
            shipping=product.shipping,
            item_description=product.item_description,
            artifacts=artifacts,
        )

        log_price = float(artifacts["model"].predict(X)[0])
        log_price = max(0.0, log_price)
        predicted_price = float(np.expm1(log_price))

        return PriceOutput(
            predicted_price_usd=round(predicted_price, 2),
            log_price=round(log_price, 4),
            message="Price predicted successfully",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))