# Project Outcome Summary — Business Perspective

## What This Project Solves

Sellers on Mercari struggle to provide the price for their listings correctly. Underpricing increases loss and overpricing means items don't sell. This project delivers an automated price suggestion system that recommends a fair market price for any product listing — instantly, based on patterns learned from 1.48 million real Mercari transactions.

## What Was Built

An end-to-end ML pipeline that takes a seller's product details (name, category, brand, condition, description, and shipping preference) and returns a suggested listing price in USD. The system is deployed as a web API with a clean browser interface, so any seller can get a price estimate without technical knowledge.

## Key Business Outcomes

| Outcome | Detail |
|---|---|
| Price accuracy | Mean prediction error of ~$10.44 (MAE) on 1.48M listings |
| Coverage | Works across all Mercari product categories — electronics, clothing, beauty, toys, and more |
| Speed | Price prediction in under 100ms per request |
| Accessibility | Usable via browser UI or REST API — no coding required |
| Scalability | Containerized with Docker; deployable on any cloud platform |

## Why This Has Real Business Value

**For sellers:** A new seller listing a used iPhone has no reliable reference point. This system surfaces a data-driven price in seconds, reducing the guesswork that causes mispricing.

**For the platform:** Correctly priced listings sell faster. Faster sales improve platform liquidity, repeat usage, and seller retention — all key marketplace metrics.

**For pricing fairness:** Prices grounded in market data reduce the advantage that experienced sellers have over new ones, creating a more level playing field.

## Limitations and Next Steps

The model was trained on historical Mercari data from a Kaggle snapshot, so it does not reflect real-time market shifts (e.g., a product trending after a news event). For a production deployment, the model would benefit from periodic retraining on fresh transaction data. Additionally, integrating image-based features (product photos) could further improve accuracy, particularly for high-value categories like electronics and luxury goods.