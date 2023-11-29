# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(rpart)
library(ranger)
library(stacks)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)
library(keras)
library(bonsai)
library(lightgbm)
library(dbarts)
library(timetk)
library(modeltime)

# Reading in Data
setwd("~/Desktop/Stat348/Store-Item_Demand/")
train <- vroom("train.csv")
test <- vroom("test.csv")
test_ids <- test
test <- test %>% select(-id)

train316 <- train %>%
  filter(store==3, item==16) %>% 
  select(-item, -store)


my_recipe <- recipe(sales ~ ., data = train316) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_date(date, features="dow") %>% 
  step_lencode_glm(date_dow, outcome = vars(sales)) %>% 
  step_date(date, features="month") %>% 
  step_lencode_glm(date_month, outcome = vars(sales)) %>% 
  step_date(date, features="year") %>% 
  step_date(date, features="decimal") %>% 
  step_date(date, features="doy") %>% 
  step_range(date_doy, min=0, max=pi) %>% 
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy))
  
bake(prep(my_recipe), new_data = train316)

# nStores <- max(train$store)
# nItems <- max(train$item)
# for(s in 1:nStores){
#   for(i in 1:nItems){
#     storeItemTrain <- train %>%
#       filter(store==s, item==i)
#     storeItemTest <- test %>%
#       filter(store==s, item==i)
#     ## Fit storeItem models here
#     ## Predict storeItem sales
#     ## Save storeItem predictions
#     if(s==1 & i==1){
#       all_preds <- preds
#     } else {
#       all_preds <- bind_rows(all_preds, preds)
#     }
#   }
# }


# EDA ---------------------------------------------------------------------

nStores <- max(train$store)
nItems <- max(train$item)
storeItemTrain <- train %>%
  filter(store==3, item==16)

# Time Series Plot
storeItemTrain %>%
  plot_time_series(date, sales, .interactive=FALSE)

# Weekly correlation, every 7 is highest
storeItemTrain %>%
  pull(sales) %>% 
  forecast::ggAcf(.)

# Seasonal correlation
storeItemTrain %>%
  pull(sales) %>% 
  forecast::ggAcf(., lag.max=2*365)

# Four ACF Plots
library(gridExtra)
plots <- list()
for (i in 1:4) {
  storeItemTrain <- train %>%
    filter(store == sample(1:10, 1), item == sample(1:50, 1))
  
  acf_plot <- storeItemTrain %>%
    pull(sales) %>% 
    forecast::ggAcf() 
  
  plots[[i]] <- acf_plot
}
grid.arrange(grobs = plots, ncol = 2)


# Feature Engineering -----------------------------------------------------

storeItemTrain <- train %>%
  filter(store==3, item==16)

storeItemTest <- test %>%
  filter(store==3, item==16)

storeItemTest_ids <- test_ids %>%
  filter(store==3, item==16) %>% 
  select(id)

my_recipe <- recipe(sales ~ ., data = storeItemTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_date(date, features="dow") %>% 
  step_lencode_glm(date_dow, outcome = vars(sales)) %>% 
  step_date(date, features="month") %>% 
  step_lencode_glm(date_month, outcome = vars(sales)) %>% 
  step_date(date, features="year") %>% 
  step_date(date, features="decimal") %>% 
  step_date(date, features="doy") %>% 
  step_range(date_doy, min=0, max=pi) %>% 
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) 
  #step_lag(sales, 1)

bake(prep(my_recipe), new_data = storeItemTrain)

#### Making Random Forest Model for this set
rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

rf_tuning_grid <- grid_regular(mtry(c(1,ncol(storeItemTrain - 1))), min_n(), levels=5)

folds <- vfold_cv(storeItemTrain, v = 5, repeats = 1)

rf_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = rf_tuning_grid,
            metrics = metric_set(smape))

rf_bestTune <- rf_results %>% 
  select_best("smape")

rf_final_wf <- rf_wf %>% 
  finalize_workflow(rf_bestTune) %>% 
  fit(data=storeItemTrain)

rf_preds <- predict(rf_final_wf,new_data=storeItemTest) 

rf_submit <- cbind(storeItemTest_ids, rf_preds)
colnames(rf_submit) <- c("id", "sales")

rf_results %>% 
  collect_metrics() %>% 
  select(mean) %>% 
  min()


# Cross Validation Plot ---------------------------------------------------

storeItemTrain <- train %>%
  filter(store==3, item==16)

cv_split <- time_series_split(storeItemTrain, assess="3 months", cumulative = TRUE)

cv_split %>% 
  tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)


# Exponential Smoothing In-class assignment------------------------------------

train316 <- train %>%
  filter(store==3, item==16)

test316 <- test_ids %>%
  filter(store==3, item==16)

cv_split316 <- time_series_split(train316, assess="3 months", cumulative = TRUE)

es_mod316 <- exp_smoothing() %>% 
  set_engine("ets") %>% 
  fit(sales~date, data = training(cv_split316))

cv_results316 <- modeltime_calibrate(es_mod316, new_data = testing(cv_split316))

p1 <- cv_results316 %>%
  modeltime_forecast(new_data = testing(cv_split316),
                     actual_data = train316) %>%
  plot_modeltime_forecast(.interactive=TRUE)

cv_results316 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

es_fullfit316 <- cv_results316 %>%
  modeltime_refit(data = train316)

es_preds316 <- es_fullfit316 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test316, by="date") %>%
  select(id, sales)

p2 <- es_fullfit316 %>%
  modeltime_forecast(h = "3 months", actual_data = train316) %>%
  plot_modeltime_forecast(.interactive=FALSE)

#### Second round

train824 <- train %>%
  filter(store==8, item==24)

test824 <- test_ids %>%
  filter(store==8, item==24)

cv_split824 <- time_series_split(train824, assess="3 months", cumulative = TRUE)

es_mod824 <- exp_smoothing() %>% 
  set_engine("ets") %>% 
  fit(sales~date, data = training(cv_split824))

cv_results824 <- modeltime_calibrate(es_mod824, new_data = testing(cv_split824))

p3 <- cv_results824 %>%
  modeltime_forecast(new_data = testing(cv_split824),
                     actual_data = train824) %>%
  plot_modeltime_forecast(.interactive=TRUE)

cv_results824 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

es_fullfit824 <- cv_results824 %>%
  modeltime_refit(data = train824)

es_preds824 <- es_fullfit824 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test824, by="date") %>%
  select(id, sales)

p4 <- es_fullfit824 %>%
  modeltime_forecast(h = "3 months", actual_data = train824) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(p1,p3,p2,p4, nrows=2)


# Sarima ------------------------------------------------------------------

arima_mod <- arima_reg(seasonal_period=365,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
                         ) %>%
set_engine("auto_arima")

### Store 3, item 16
train316 <- train %>%
  filter(store==3, item==16) %>% 
  select(-item, -store)

test316 <- test_ids %>%
  filter(store==3, item==16) %>% 
  select(-item, -store)

cv_split316 <- time_series_split(train316, assess="3 months", cumulative = TRUE)

arima_wf316 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(arima_mod) %>%
  fit(data=training(cv_split316))

cv_arima_results316 <- modeltime_calibrate(arima_wf316, new_data = testing(cv_split316))

p1 <- cv_arima_results316 %>%
  modeltime_forecast(new_data = testing(cv_split316),
                     actual_data = train316) %>%
  plot_modeltime_forecast(.interactive=TRUE)

arima_fullfit316 <- cv_arima_results316 %>%
  modeltime_refit(data = train316)

arima_preds316 <- arima_fullfit316 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test316, by="date") %>%
  select(id, sales)

p2 <- arima_fullfit316 %>%
  modeltime_forecast(h = "3 months", actual_data = train316) %>%
  plot_modeltime_forecast(.interactive=FALSE)

### Store 8, item 24
train824 <- train %>%
  filter(store==8, item==24) %>% 
  select(-item, -store)

test824 <- test_ids %>%
  filter(store==8, item==24) %>% 
  select(-item, -store)

cv_split824 <- time_series_split(train824, assess="3 months", cumulative = TRUE)

arima_wf824 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(arima_mod) %>%
  fit(data=training(cv_split824))

cv_arima_results824 <- modeltime_calibrate(arima_wf824, new_data = testing(cv_split824))

p3 <- cv_arima_results824 %>%
  modeltime_forecast(new_data = testing(cv_split824),
                     actual_data = train824) %>%
  plot_modeltime_forecast(.interactive=TRUE)

arima_fullfit824 <- cv_arima_results824 %>%
  modeltime_refit(data = train824)

arima_preds824 <- arima_fullfit824 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test824, by="date") %>%
  select(id, sales)

p4 <- arima_fullfit824 %>%
  modeltime_forecast(h = "3 months", actual_data = train824) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(p1,p2,p3,p4, nrows=2)


# Prophet Model -----------------------------------------------------------

prophet_mod <- prophet_reg() %>% 
  set_engine("prophet")

### Store 3, item 16
train316 <- train %>%
  filter(store==3, item==16) %>% 
  select(-item, -store)

test316 <- test_ids %>%
  filter(store==3, item==16) %>% 
  select(-item, -store)

cv_split316 <- time_series_split(train316, assess="3 months", cumulative = TRUE)

prophet_wf316 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(prophet_mod) %>%
  fit(data = training(cv_split316))

cv_prophet_results316 <- modeltime_calibrate(prophet_wf316, new_data = testing(cv_split316))

p1 <- cv_prophet_results316 %>%
  modeltime_forecast(new_data = testing(cv_split316),
                     actual_data = train316) %>%
  plot_modeltime_forecast(.interactive=TRUE)

prophet_fullfit316 <- cv_prophet_results316 %>%
  modeltime_refit(data = train316)

prophet_preds316 <- prophet_fullfit316 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test316, by="date") %>%
  select(id, sales)

p2 <- prophet_fullfit316 %>%
  modeltime_forecast(h = "3 months", actual_data = train316) %>%
  plot_modeltime_forecast(.interactive=FALSE)

### Store 8, item 24
train824 <- train %>%
  filter(store==8, item==24) %>% 
  select(-item, -store)

test824 <- test_ids %>%
  filter(store==8, item==24) %>% 
  select(-item, -store)

cv_split824 <- time_series_split(train824, assess="3 months", cumulative = TRUE)

prophet_wf824 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(prophet_mod) %>%
  fit(data = training(cv_split316))

cv_prophet_results824 <- modeltime_calibrate(prophet_wf824, new_data = testing(cv_split824))

p3 <- cv_prophet_results824 %>%
  modeltime_forecast(new_data = testing(cv_split824),
                     actual_data = train824) %>%
  plot_modeltime_forecast(.interactive=TRUE)

prophet_fullfit824 <- cv_prophet_results824 %>%
  modeltime_refit(data = train824)

prophet_preds824 <- prophet_fullfit824 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test824, by="date") %>%
  select(id, sales)

p4 <- prophet_fullfit824 %>%
  modeltime_forecast(h = "3 months", actual_data = train824) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(p1,p2,p3,p4, nrows=2)
