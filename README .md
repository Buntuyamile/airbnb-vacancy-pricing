# Cape Town Airbnb Vacancy Insurance Pricing

Predicting Airbnb listing vacancy in Cape Town using SVM and Neural Network models, to support premium pricing for a short-term rental insurance product.

## Business Problem

**VacancyShield** is a (hypothetical) short-term rental insurance startup. Working as their data scientist, I set out to build a model predicting which Cape Town Airbnb listings are likely to have **at least one vacant night in the next 30 days**.

The predicted probability `P̂(vacancy)` feeds directly into VacancyShield's premium pricing engine — listings with higher predicted vacancy risk receive higher premiums. This means the model's outputs need to be:

- **Well-ranked** : able to reliably order listings from lowest to highest vacancy risk
- **Well-calibrated** : the predicted probabilities need to reflect real-world likelihoods, not just relative ranking

## Data

| | |
|---|---|
| Training listings | 10,000 |
| Features after engineering | 23 |
| Vacant listings (positive class) | 81.7% |
| Missing values | 0 |

**Engineered features** include:
- `host_experience` : years since `host_since`, a proxy for host reliability
- `num_verifications` : count of verification methods (0–3)
- `response_time_ord` : ordinal encoding of host response time (1 = fastest, 4 = slowest)
- `is_superhost`, `is_instant`, `is_private_room` — binary flags

All models were trained with `preProcess = c('center', 'scale')`, which matters given the range of raw features (prices up to R323,000 vs. review scores of 1–5).

## Models

| Model | AUROC | AUPRC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| SVM (Linear) | 0.6006 | 0.8532 | 0.7403 | 0.6608 | 0.8417 | 0.6215 |
| SVM (Nonlinear, RBF) | 0.6497 | 0.8727 | **0.8162** | **0.7717** | 0.8660 | **0.7161** |
| Neural Network | **0.6953** | **0.9032** | 0.7399 | 0.6342 | **0.8879** | 0.6359 |

*τ = 0.817 (positive class prevalence), 5-fold cross-validation, n = 10,000*

### 1. Linear SVM
A baseline linear-kernel SVM, tuned over 10 log-spaced values of the cost parameter `C` (0.01–10) via 5-fold CV.

- Best `C ≈ 0.046`, CV AUROC = 0.6006
- AUROC near 0.60 is barely better than random, commercially unviable for pricing
- A jagged tuning curve across `C` values signaled model instability
- Conclusion: vacancy vs. fully booked is **not linearly separable** in this feature space, motivating a nonlinear approach

### 2. Nonlinear SVM (RBF Kernel)
An RBF-kernel SVM with `C` and `sigma` tuned jointly over a 5×5 grid (25 models) via 5-fold CV.

- Best `sigma = 0.1`, `C = 1.778`, CV AUROC = 0.6497
- A +0.049 AUROC improvement over the linear SVM confirmed real nonlinear structure in the data
- `sigma = 0.1` dominated across all `C` values, giving a stable result
- The optimum landed at the edge of the tuning grid, suggesting the true optimum may lie beyond it

### 3. Feed-Forward Neural Network
A single hidden-layer network tuned over 4 hidden-layer sizes (3, 5, 7, 10) × 4 weight decay values (16 combinations), 5-fold CV, `maxit = 500`.

- CV AUROC = 0.6953, CV AUPRC = 0.9032
- Best AUROC, best AUPRC, and best precision (0.8879) of the three models
- Output is directly interpretable as `P̂(fully_booked_30 = 'no')` — a natural fit for premium pricing
- **Selected as VacancyShield's production model**

## Model Selection

**AUROC** was used as the primary selection metric because it's threshold-independent and directly reflects how reliably a model ranks a vacant listing above a fully booked one — which is exactly what premium pricing requires. The Neural Network led on both AUROC (0.6953) and AUPRC (0.9032), indicating it produces the best-calibrated vacancy probabilities of the three approaches.

## Conclusion

| Model | AUROC | Verdict |
|---|---|---|
| Linear SVM | 0.60 | Linear boundary insufficient — commercially unviable |
| Nonlinear SVM | 0.65 | RBF kernel captured nonlinear structure; tuning grid boundary reached |
| **Neural Network** | **0.70** | Best probability calibration (AUPRC 0.90) — selected as production model |

Final predictions (`P̂(fully_booked_30 = 'no')` from the Neural Network) were used for the pricing submission, using threshold `τ = 0.817` (positive class prevalence) for computing F1, recall, and precision.

## What I Learned

- **AUROC isn't the whole story for pricing use cases.** With an 82/18 class imbalance, AUPRC and probability calibration mattered more than raw accuracy for a model whose output feeds directly into pricing.
- **Instability is visible, not just theoretical.** The linear SVM's jagged tuning curve across `C` was a concrete, visual example of the bias-variance tradeoff in action.
- **Model choice should follow the business decision, not the leaderboard.** Picking a "winner" meant identifying which metric actually reflected what the business needed (reliable ranking + calibrated probabilities), not just picking the highest number in the table.

Data Source

This project uses Cape Town Airbnb listing data derived from Inside Airbnb, a mission-driven project providing data and advocacy about Airbnb's impact on residential communities. Inside Airbnb's data is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).

Features were engineered from the raw listing data to build the target variable and predictors used here (e.g. fully_booked_30, host_experience, num_verifications). The hypothetical VacancyShield scenario and pricing framing are original to this project and not affiliated with Inside Airbnb or Airbnb itself.

## Tech Stack

- R (`caret`, `kernlab`/`e1071` for SVMs, `nnet` for the feed-forward network)
- `pROC` / `PRROC` for AUROC and AUPRC evaluation
- `doParallel` for parallelised 5-fold cross-validation
- `ggplot2` for tuning curves and model comparison plots

## Repo Structure

```
├── vacancy_model.R   # Full pipeline: data cleaning, feature engineering,
│                      # model training/tuning, evaluation, and final predictions
└── README.md
```

The script covers:
1. Data cleaning and feature engineering (host experience, verification counts, response-time encoding, binary flags)
2. Model training with 5-fold CV for all three models (linear SVM, RBF SVM, neural network)
3. Evaluation (AUROC, AUPRC, F1, recall, precision, accuracy) and tuning/ROC visualisations
4. Final predictions on the held-out test set using the selected model

## Author

**Buntu Mlonyeni**
