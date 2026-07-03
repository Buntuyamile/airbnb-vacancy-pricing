
# Cape Town Airbnb Vacancy Insurance Pricing
# Predicting listing vacancy using SVM and Neural Network models
# Author: Buntu Mlonyeni


library(tidyverse)
library(caret)
library(glmnet)
library(pROC)
library(PRROC)
library(pdp)
library(knitr)
library(kableExtra)
library(gridExtra)
library(scales)
library(doParallel)
library(e1071)
library(nnet)

options(knitr.table.format = "latex")


# 1. INTRODUCTION AND DATA PREPARATION



# 1.1 Data Preparation


# Read both datasets exactly as provided
train_raw <- read.csv("listings.csv", stringsAsFactors = FALSE)
test_raw  <- read.csv("testing.csv",  stringsAsFactors = FALSE)

# Store summary values silently for use in text
n_missing <- sum(is.na(train_raw))
n_no      <- sum(train_raw$fully_booked_30 == "no")
n_yes     <- sum(train_raw$fully_booked_30 == "yes")

# Single preprocessing function applied identically to train and test
clean_data <- function(df) {
  df |>
    mutate(
      # Years of hosting experience from host_since to end of 2025
      host_experience = as.numeric(
        difftime(as.Date("2025-12-31"),
                 as.Date(host_since, format = "%Y/%m/%d"),
                 units = "days")
      ) / 365.25,

      # Count of verification methods via exact string matching
      num_verifications = case_when(
        host_verifications == "['email', 'phone', 'work_email']" ~ 3L,
        host_verifications == "['email', 'phone']"               ~ 2L,
        host_verifications == "['phone', 'work_email']"          ~ 2L,
        host_verifications == "['phone']"                        ~ 1L,
        host_verifications == "['email']"                        ~ 1L,
        TRUE                                                     ~ 0L
      ),

      # Ordinal encoding: 1 = fastest response, 4 = slowest
      response_time_ord = case_when(
        host_response_time == "within an hour"     ~ 1L,
        host_response_time == "within a few hours" ~ 2L,
        host_response_time == "within a day"       ~ 3L,
        host_response_time == "a few days or more" ~ 4L,
        TRUE                                       ~ NA_integer_
      ),

      # Binary flags: 1 = true, 0 = false
      is_superhost    = ifelse(host_is_superhost == "t", 1L, 0L),
      is_instant      = ifelse(instant_bookable  == "t", 1L, 0L),
      is_private_room = ifelse(room_type == "Private room", 1L, 0L)
    ) |>
    # Drop original columns replaced by the engineered features above
    select(-host_id, -host_since, -host_verifications,
           -host_response_time, -host_is_superhost,
           -instant_bookable, -room_type)
}

train_clean <- clean_data(train_raw)
test_clean  <- clean_data(test_raw)

# Binary outcome: 1 = "no" (not fully booked) = positive class
train_clean$y <- ifelse(train_clean$fully_booked_30 == "no", 1L, 0L)
train_clean   <- select(train_clean, -fully_booked_30)

# Feature matrix and target vector
X <- select(train_clean, -y)
y <- train_clean$y

# tau = proportion of positive class, used as the classification threshold
tau <- mean(y == 1)

# Factor outcome
y_factor <- factor(ifelse(y == 1, "no", "yes"), levels = c("yes", "no"))

# Single modelling data frame used by all models
model_data <- cbind(X, outcome = y_factor)

#Control defined

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)


# Evaluation
get_cv_preds <- function(fit_obj) {
  
  preds <- as.data.frame(fit_obj$pred)
  
  if (fit_obj$method == "glm") {
    preds <- preds[!duplicated(preds$rowIndex), ]
    rownames(preds) <- NULL
    return(preds[order(as.integer(preds$rowIndex)), , drop = FALSE])
  }


  best_params  <- fit_obj$bestTune
  valid_params <- intersect(names(best_params), names(preds))
  
  keep <- rep(TRUE, nrow(preds))
  for (param in valid_params) {
    best_val <- best_params[[param]]
    col_vals <- preds[[param]]
    if (is.numeric(col_vals)) {
      keep <- keep & (abs(col_vals - best_val) < 1e-9)
    } else {
      keep <- keep & (col_vals == best_val)
    }
  }
  
  preds <- preds[keep, , drop = FALSE]
  preds <- preds[!duplicated(preds$rowIndex), ]
  
  if (nrow(preds) == 0) {
    stop(paste("get_cv_preds: no rows left after filtering for", fit_obj$method))
  }
  
  rownames(preds) <- NULL
  preds[order(as.integer(preds$rowIndex)), , drop = FALSE]
}

compute_metrics <- function(fit_obj, tau) {
  
  preds  <- get_cv_preds(fit_obj)
  p_hat  <- as.numeric(preds$no)
  y_true <- ifelse(preds$obs == "no", 1L, 0L)
  
  keep  <- complete.cases(p_hat, y_true)
  p_hat <- p_hat[keep];  y_true <- y_true[keep]
  
  
  # AUROC
  auroc <- as.numeric(auc(roc(y_true, p_hat, quiet = TRUE)))
  
  # AUPRC
  auprc <- tryCatch({
    pr.curve(scores.class0 = p_hat[y_true == 1],
             scores.class1 = p_hat[y_true == 0],
             curve = FALSE)$auc.integral
  }, error = function(e) NA)
  
  # Threshold-based metrics at tau
  y_pred <- ifelse(p_hat >= tau, 1L, 0L)
  TP <- sum(y_pred == 1 & y_true == 1)
  FP <- sum(y_pred == 1 & y_true == 0)
  TN <- sum(y_pred == 0 & y_true == 0)
  FN <- sum(y_pred == 0 & y_true == 1)
  
  recall    <- TP / (TP + FN)
  precision <- TP / (TP + FP)
  f1        <- 2 * precision * recall / (precision + recall)
  accuracy  <- (TP + TN) / (TP + FP + TN + FN)
  
  round(c(AUROC     = auroc,
          AUPRC     = auprc,
          F1        = f1,
          Recall    = recall,
          Precision = precision,
          Accuracy  = accuracy), 4)
}


# Paralleisation

n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)
clusterSetRNGStream(cl, iseed = 2488)


# 2. MODEL TRAINING

# Lnear SVM

svm_linear_grid <- expand.grid(C = 10^seq(-2, 1, length.out = 10))

set.seed(2488)
fit_svm_linear <- train(
  outcome    ~ .,
  data       = model_data,
  method     = "svmLinear",
  metric     = "ROC",
  tuneGrid   = svm_linear_grid,
  trControl  = ctrl,
  preProcess = c("center", "scale")
)

stopCluster(cl)
registerDoSEQ()

# Tuning plot
ggplot(fit_svm_linear$results, aes(x = C, y = ROC)) +
  geom_line(colour = "skyblue", linewidth = 1) +
  geom_point(colour = "red", size = 3) +
  geom_point(
    data   = fit_svm_linear$results[which.max(fit_svm_linear$results$ROC), ],
    aes(x = C, y = ROC),
    colour = "darkgreen", size = 6, shape = 18
  ) +
  scale_x_log10() +
  labs(x = "Cost C (log scale)", y = "CV AUROC",
       title = "Linear SVM: tuning the cost parameter") +
  theme_bw(base_size = 10)

view(fit_svm_linear$results)

library(doParallel)

n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)
clusterSetRNGStream(cl, iseed = 2488)

# Verify
getDoParWorkers()  # should return > 1

# non-linear SVM

svm_rbf_grid <- expand.grid(
  C     = 10^seq(-2, 1, length.out = 5),
  sigma = 10^seq(-3, -1, length.out = 5)
)

set.seed(2488)
fit_svm_rbf <- suppressWarnings(
  train(
    outcome    ~ .,
    data       = model_data,
    method     = "svmRadial",
    metric     = "ROC",
    tuneGrid   = svm_rbf_grid,
    trControl  = ctrl,
    preProcess = c("center", "scale")
  )
)

fit_svm_rbf

ggplot(fit_svm_rbf$results, aes(x = C, y = ROC,
                                colour = factor(round(sigma, 4)),
                                group  = factor(round(sigma, 4)))) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_point(
    data = fit_svm_rbf$results[which.max(fit_svm_rbf$results$ROC), ],
    aes(x = C, y = ROC),
    colour = "darkgreen", size = 6, shape = 18, inherit.aes = FALSE
  ) +
  scale_x_log10() +
  labs(x = "Cost C (log scale)", y = "CV AUROC",
       colour = "Sigma",
       title  = "Nonlinear SVM (kernal): tuning C and sigma") +
  theme_bw(base_size = 10)


# feed-forward Neural network
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)
clusterSetRNGStream(cl, iseed = 2488)

nn_grid <- expand.grid(
  size  = c(3, 5, 7, 10),
  decay = 10^seq(-4, -1, length.out = 4)
)

set.seed(2488)
fit_nn <- train(
  outcome    ~ .,
  data       = model_data,
  method     = "nnet",
  metric     = "ROC",
  tuneGrid   = nn_grid,
  trControl  = ctrl,
  preProcess = c("center", "scale"),
  maxit      = 500,
  trace      = FALSE
)

ggplot(fit_nn$results, aes(x = size, y = ROC,
                           colour = factor(round(decay, 5)),
                           group  = factor(round(decay, 5)))) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_point(
    data = fit_nn$results[which.max(fit_nn$results$ROC), ],
    aes(x = size, y = ROC),
    colour = "darkgreen", size = 6, shape = 18, inherit.aes = FALSE
  ) +
  labs(x = "Hidden units (size)", y = "CV AUROC",
       colour = "Decay",
       title  = "Neural network: tuning size and decay") +
  theme_bw(base_size = 10)

fit_nn

# 8. STOP PARALLEL

stopCluster(cl)
registerDoSEQ()

# 9. MODEL EVALUATION 
m_svm_linear <- compute_metrics(fit_svm_linear, tau)
m_svm_rbf    <- compute_metrics(fit_svm_rbf,    tau)
m_nn         <- compute_metrics(fit_nn,         tau)

results_df <- rbind(
  "SVM (Linear)"    = m_svm_linear,
  "SVM (Nonlinear)" = m_svm_rbf,
  "Neural Network"  = m_nn
)

kable(results_df, booktabs = TRUE, align = "ccccccc",
      caption = paste0(
        "5-fold CV performance metrics for all three models. ",
        "AUROC is the primary metric (higher is better). ",
        "Threshold-based metrics computed at tau = ", round(tau, 3), "."
      )) |>
  kable_styling(latex_options = c("hold_position", "scale_down"))

# Faceted bar chart
metrics_long <- as.data.frame(results_df) |>
  rownames_to_column("Model") |>
  pivot_longer(-Model, names_to = "Metric", values_to = "Value") |>
  mutate(
    Metric = factor(Metric, levels = c("AUROC", "AUPRC", "F1",
                                       "Recall", "Precision", "Accuracy")),
    Model  = factor(Model,  levels = c("SVM (Linear)", "SVM (Nonlinear)",
                                       "Neural Network"))
  )

model_colours <- c(
  "SVM (Linear)"    = "skyblue",
  "SVM (Nonlinear)" = "blue",
  "Neural Network"  = "red"
)

ggplot(metrics_long, aes(x = Model, y = Value, fill = Model)) +
  geom_col(alpha = 0.9, width = 0.65) +
  facet_wrap(~ Metric, scales = "free_y", nrow = 2) +
  scale_fill_manual(values = model_colours) +
  labs(x = NULL, y = NULL, fill = NULL,
       title = "CV performance metrics by model") +
  theme_bw(base_size = 9) +
  theme(
    axis.text.x     = element_blank(),
    axis.ticks.x    = element_blank(),
    legend.position = "bottom",
    strip.text      = element_text(face = "bold"),
    legend.text     = element_text(size = 8)
  )

# Overlaid ROC curves
get_roc <- function(fit_obj) {
  preds  <- get_cv_preds(fit_obj)
  p_hat  <- as.numeric(preds$no)
  y_true <- ifelse(preds$obs == "no", 1L, 0L)
  roc(y_true, p_hat, quiet = TRUE)
}

roc_linear <- get_roc(fit_svm_linear)
roc_rbf    <- get_roc(fit_svm_rbf)
roc_nn     <- get_roc(fit_nn)

plot(roc_linear, col = "skyblue", lwd = 2,
     main = "ROC curves: 5-fold CV out-of-fold predictions",
     xlab = "1 - Specificity", ylab = "Sensitivity")
plot(roc_rbf, col = "blue", lwd = 2, add = TRUE)
plot(roc_nn,  col = "red",  lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "grey")
legend("bottomright",
       legend = c(paste0("SVM Linear    AUC = ", round(auc(roc_linear), 3)),
                  paste0("SVM Nonlinear AUC = ", round(auc(roc_rbf),    3)),
                  paste0("Neural Net    AUC = ", round(auc(roc_nn),     3))),
       col  = c("skyblue", "blue", "red"),
       lwd  = 2, bty = "n", cex = 0.85)




best_model <- fit_nn

#final model output on testdataset

set.seed(2488)
test_probs <- predict(best_model, newdata = test_clean, type = "prob")[["no"]]


