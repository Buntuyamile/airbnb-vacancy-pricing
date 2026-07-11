# Cape Town Airbnb Vacancy Insurance Pricing

### Predicting listing vacancy with SVMs and a neural network, and what it takes to turn a model into a pricing decision

---

## The Idea

Most machine learning projects stop at "which model has the best score." This one was built around a different question: **if a real business had to act on this prediction, what would actually matter?**

I framed the project around a hypothetical short-term rental insurance startup, **VacancyShield**. Their product only works if they can price risk accurately: insure a listing against vacancy, and the premium they charge needs to reflect how likely that listing actually is to sit empty. Get it wrong in one direction and they lose money on every policy; get it wrong in the other and they price themselves out of the market.

That framing changes what "a good model" means. It's not enough to be right most of the time, since the output has to be a **probability that can be trusted**, because that probability becomes a price. This is the thread that runs through every decision in the project: which features to engineer, which models to try, and, most importantly, which metric actually tells you if the model is good enough to build a business on.

The data itself is real: 10,000 Cape Town Airbnb listings, sourced from [Inside Airbnb](https://insideairbnb.com/get-the-data/) (CC BY 4.0 licensed). The business scenario is hypothetical; the listings, prices, and booking patterns behind it are not.

---

## The Problem, Precisely

**Target:** predict whether a listing will have **at least one vacant night in the next 30 days** (`fully_booked_30 = 'no'` is the positive class, "vacant," since that's the outcome VacancyShield needs to price for).

**Why this framing matters:** the predicted probability, `P̂(vacancy)`, feeds directly into a pricing engine. That means the model's output needs two properties that a typical classification project doesn't always prioritise:

- **Well-ranked**, meaning listings that are actually riskier need to consistently score higher than listings that aren't, regardless of where you draw the line between "risky" and "safe."
- **Well-calibrated**, meaning a listing predicted at 70% vacancy risk should actually behave like a 70%-risk listing in the real world, not just rank above a 50%-risk one.

Most classification tutorials optimise for accuracy. Neither of the two properties above is accuracy. That mismatch became the central design decision of the project.

---

## The Data

| | |
|---|---|
| Listings | 10,000 |
| Features after engineering | 23 |
| Class balance | 81.7% vacant ("no") / 18.3% fully booked ("yes") |
| Missing values | 0 |

The raw data includes host attributes, pricing, room characteristics, and booking history. Getting it into a usable shape meant engineering a handful of features that don't exist in the raw export:

- **`host_experience`**: years since `host_since`, used as a rough proxy for how established (and presumably reliable) a host is
- **`num_verifications`**: a count of verification methods on the host account (0 to 3), rather than treating the raw list as a single categorical blob
- **`response_time_ord`**: host response time turned into an ordinal scale (1 = fastest, 4 = slowest), since "within an hour" and "a few days or more" have a natural order that a plain categorical encoding would throw away
- **`is_superhost` / `is_instant` / `is_private_room`**: binary flags collapsed from their original string form

One easy-to-miss detail that mattered a lot in practice: raw prices in the data range up to R323,000, while review scores sit on a 1 to 5 scale. Feeding that straight into a distance-based model like an SVM would let price dominate every other feature purely because of its scale, not because it's more predictive. Every model in this project was trained with `preProcess = c('center', 'scale')` for exactly that reason.

The class split, roughly 82/18, also isn't cosmetic. It's the reason accuracy alone is a misleading metric here: a model that predicts "vacant" for every single listing would already be right 82% of the time while being completely useless for pricing anything.

---

## Model 1: Linear SVM, establishing a baseline and its limits

A linear SVM tries to separate the two classes with a straight-line boundary (a hyperplane), maximizing the margin between them. Its single hyperparameter, cost `C`, controls how much the model is penalized for margin violations. A small `C` tolerates more misclassification in exchange for a wider, more stable margin (higher bias), while a large `C` chases a narrower margin with fewer violations (higher variance).

**Tuning:** 5-fold cross-validation across 10 log-spaced values of `C`, from 0.01 to 10.

**Result:** best `C ≈ 0.046`, CV AUROC = **0.6006**.

An AUROC of 0.60 is barely better than random guessing. Two things stood out from this result:

1. **The tuning curve was jagged** across different values of `C`, a visual, concrete example of model instability, not just a textbook description of it.
2. **The conclusion wasn't "the model failed," it was diagnostic**: vacancy and full-booking status aren't linearly separable in this feature space. A straight-line boundary was never going to be enough, which is exactly the kind of result that should redirect your next step rather than just be reported as a disappointing number.

---

## Model 2: Nonlinear SVM (RBF Kernel), testing for real structure

The RBF (radial basis function) kernel implicitly projects the data into a higher-dimensional space, allowing a curved decision boundary back in the original feature space, without ever explicitly computing that higher-dimensional representation.

This introduces a second hyperparameter, `sigma`, alongside `C`. Intuitively:
- **Small sigma** means each point's influence reaches further, producing a smoother, more global boundary (risk of underfitting)
- **Large sigma** means each point's influence is tightly local, producing a boundary that can overfit individual points
- **High `C` combined with high `sigma`** is a particularly dangerous combination, since both push toward overfitting simultaneously, which is why the two need to be tuned *jointly*, not one after the other.

**Tuning:** a 5×5 grid of `C` and `sigma` combinations (25 models total), 5-fold CV.

**Result:** best `sigma = 0.1`, `C = 1.778`, CV AUROC = **0.6497**.

That's a +0.049 improvement over the linear SVM, modest in absolute terms, but meaningful as evidence: it confirms there's genuine nonlinear structure in the relationship between these features and vacancy, not just noise. `sigma = 0.1` dominated across every value of `C` tested, giving a stable, consistent result rather than a fragile one.

One caveat worth being upfront about: the best result landed **at the edge of the searched grid**. That's a signal the true optimum might sit outside the range I tested, a good reminder that a tuning grid is a hypothesis about where the optimum lives, not a guarantee.

---

## Model 3: Feed-Forward Neural Network, the production candidate

A single hidden-layer network: inputs feed into a layer of hidden units, which feed into an output probability via a sigmoid activation. Two hyperparameters were tuned:

- **`size`**: the number of hidden units, controlling model capacity
- **`decay`**: L2 weight regularisation, functionally similar to ridge regression, used to prevent overfitting

**Tuning:** a grid of 4 hidden-layer sizes (3, 5, 7, 10) × 4 decay values (16 combinations), 5-fold CV, with `maxit = 500` to make sure the optimizer actually converges rather than stopping early.

**Result:** CV AUROC = **0.6953**, CV AUPRC = **0.9032**.

This was the best-performing model on every metric that matters for the business problem: best AUROC (best ranking), best AUPRC (best-calibrated probabilities), and best precision (fewest false vacancy predictions). Its output is also directly interpretable as `P̂(fully_booked_30 = 'no')`, exactly the number a pricing engine needs, with no extra transformation required.

**This became VacancyShield's production model.**

---

## Model Comparison

| Model | AUROC | AUPRC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| SVM (Linear) | 0.6006 | 0.8532 | 0.7403 | 0.6608 | 0.8417 | 0.6215 |
| SVM (Nonlinear, RBF) | 0.6497 | 0.8727 | **0.8162** | **0.7717** | 0.8660 | **0.7161** |
| Neural Network | **0.6953** | **0.9032** | 0.7399 | 0.6342 | **0.8879** | 0.6359 |

*(★ = best in column; τ = 0.817, the positive class prevalence, used as the classification threshold for F1/recall/precision/accuracy; 5-fold cross-validation, n = 10,000)*

**Why AUROC was the deciding metric, not accuracy or F1:** AUROC is threshold-independent, meaning it measures how reliably the model ranks a genuinely vacant listing above a genuinely fully-booked one, across every possible cutoff. That's precisely what a pricing engine needs, since the engine will use the *raw probability*, not a single yes/no cutoff. The Neural Network's lead on both AUROC and AUPRC meant it wasn't just ranking well, its probabilities were the most trustworthy of the three, which is the property that actually gets used downstream.

**A nuance worth being honest about:** the story isn't a clean sweep. The nonlinear SVM actually *beat* the neural network on F1, recall, and accuracy. If the goal had been "catch as many actually-vacant listings as possible" (maximizing recall), the SVM would have been the better choice. Model selection here wasn't about finding a single best model, it was about matching the model to what the business decision actually required.

---

## Where the Results Actually Stand

It's worth being direct about this rather than only presenting the headline numbers: an AUROC of 0.70 is a real improvement over the linear baseline, but it isn't where a production-grade pricing model would ideally sit. In real underwriting contexts, something closer to **0.80 and above** is generally the bar for a model that's genuinely reliable enough to price risk on. Anything meaningfully above 0.90 on tabular behavioral data like this would actually be worth treating with suspicion, often a sign of a leaked feature rather than a stronger model, not a "great result" to take at face value.

That gap matters more as a signpost than as a failure. This project was a first full pass at comparing model families end to end on a business-shaped problem, and knowing precisely where the model falls short, and why, is worth more than a number that looks better but hides the same limitation.

There's also an honest limitation in how these results were evaluated. All reported metrics come from 5-fold cross-validation on the training data, each fold gives a genuine, if internal, estimate of how the model generalizes to data it hasn't seen during that fold's training. A separate blind test set was used to generate final predictions, but without ground-truth labels for it, there's no way to verify true out-of-sample accuracy. Cross-validation remains the most credible evidence available here, presented as exactly that: a credible estimate, not a substitute for real-world validation.

---

## What I'd Do Differently

A few concrete next steps, each tied to a specific gap in the current approach rather than a generic "do more":

- **Location-aware features.** The model currently sees only raw latitude/longitude. Distance to the city centre or coastline, and neighbourhood-level clustering, would likely capture more of the actual vacancy signal than host attributes alone. Location plausibly matters more than almost anything else in short-term rental demand.
- **Seasonality.** This is a 30-day-ahead prediction with zero awareness of time of year. Cape Town's rental market has an obvious seasonal rhythm that the current feature set can't see at all.
- **A wider tuning grid for the RBF SVM.** Its best result landed right at the edge of the range searched, a strong hint that a broader grid could meaningfully improve on 0.65.
- **Gradient-boosted trees.** Something like XGBoost or LightGBM is a natural next baseline. Tree ensembles tend to outperform both SVMs and shallow neural networks on tabular data like this, and would be a fair comparison point against the current production model.
- **Explicit calibration diagnostics.** AUPRC is a reasonable proxy for calibration, but it isn't a direct measurement. A proper reliability plot would confirm, or challenge, whether the neural network's probabilities are as trustworthy as the AUPRC number suggests.

---

## Data Source & License

This project uses Cape Town Airbnb listing data derived from [Inside Airbnb](https://insideairbnb.com/get-the-data/), a project providing data and advocacy about Airbnb's impact on residential communities. The data is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).

The `fully_booked_30` target and other engineered features (`host_experience`, `num_verifications`, `response_time_ord`, etc.) were built specifically for this project. The VacancyShield scenario and pricing framing are original to this write-up and are not affiliated with Inside Airbnb or Airbnb.

---

## Tech Stack

- **R**: `caret` for the modeling workflow, `kernlab`/`e1071` for both SVMs, `nnet` for the neural network
- **pROC** / **PRROC** for AUROC and AUPRC evaluation
- **doParallel** for parallelized 5-fold cross-validation
- **ggplot2** for tuning curves, ROC curves, and model comparison visuals

---

## Author

**Buntu Mlonyeni**
