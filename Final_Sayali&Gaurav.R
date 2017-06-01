library(forecast) 
library(fpp) 
library(caret) 
library(neuralnet) 
library(randomForest) 
library(psych) 
library(VIM) 
library(mice) 
library(ResourceSelection) 
library(corrplot) 
library(party)
library(mlbench)
library(imputeTS)
library(hts)
library(h2o)

train_feature <- read_csv("~/Documents/College/Predictive Analysis/DengAI/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv")
train_lable <- read_csv("~/Documents/College/Predictive Analysis/DengAI/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv")
test_Data_Features <- read_csv("~/Documents/College/Predictive Analysis/DengAI/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv")


#combine train features and labels data sets 
new_data <- cbind(train_feature,train_lable$total_cases)

#table(new_data$ndvi_ne)
#know the count of null values in each columns
Num_NA <- sapply(new_data, function(x) sum(is.na(x)))
sum(Num_NA)
#there are total 548 null values 
#know the exact Null value percentage in each columns
NA_percentage <- function(x){sum(is.na(x))/length(x)*100}
Percent_data<- apply(new_data,2,NA_percentage)

mycor=cor(new_data[,5:25]) 
print(mycor)
highlyCorrelated <- findCorrelation(mycor, cutoff=0.1)
#highlyCorrelated is empty because there are so many missing values
corrplot(mycor) 

aggr_plot <- aggr(new_data, numbers=TRUE, sortVars=TRUE, 
                  labels=names(new_data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))
#using package imputeTS
new1 <- new_data
new1<-na.interpolation(new1)
test<-test_Data_Features
test[is.na(test)==TRUE] <- 0
test1<-na.interpolation(test)

Num_NA1 <- sapply(new1, function(x) sum(is.na(x)))
sum(Num_NA1)
sj=new1[1:936,]
sum(is.na(sj))
#-------------------
mydata <- read_csv("~/Documents/College/Predictive Analysis/DengAI/mydata.csv")
my_ts <-ts(mydata, frequency =52, start = c(2000,26))
plot(my_ts)
#-------------------
par(mfrow=c(2,2), mar=c(4,4,2,1))
sj_time_series <- sj[,-c(1,2,3,4)]
my_sj_ts <-ts(sj_time_series, frequency =52, start = c(1990,18))
decompose(my_sj_ts, type="additive")  #to extract the trend, seasonality and error

plot(stl(beerprod, "periodic")) 
acf(my_sj_ts)
pacf(my_sj_ts)

acf.plot(my_sj_ts)

localh2o=h2o.init()
train1<-as.h2o(new_data)
h2o.test<-as.h2o(test1)



#Loading the required library
library(forecast)

#Loading the required data
Data<-dengue_features_train
train<-dengue_features_train
#Checking the data type of the time column of the dataset
str(Data$weekstartdate)

#Changing the format of the date column to %m/%d/%Y
Data$weekstartdate <- as.Date(Data$weekstartdate, "%m/%d/%Y")

#PLotting the time series data
plot(Data$weekstartdate, Data$total_cases, type = "l")
plot(Data$year, Data$total_cases)
boxplot(total_cases~year,data=Data, main="Data")
years <- format(Data$weekstartdate , "%Y")
tab <- table(years)
tab

mean(tab[1:(length(tab) - 1)])

forecastStl <- function(Data, n.ahead = 30) {
  myTs <- ts(Data$total_cases, start = 1, frequency = 71.5)
  fit.stl <- stl(myTs, s.window = 71.4)
  sts <- fit.stl$time.series
  trend <- sts[, "trend"]
  fore <- forecast(fit.stl, h = n.ahead, level = 95)
  plot(fore)
  pred <- fore$mean
  upper <- fore$upper
  lower <- fore$lower
  output <- data.frame(actual = c(Data$total_cases, rep(NA, n.ahead)), 
                       trend = c(trend, rep(NA, n.ahead)), pred = c(rep(NA, 
                                                                        nrow(Data)), pred), lower = c(rep(NA, nrow(Data)), lower), 
                       upper = c(rep(NA, nrow(Data)), upper), date = c(Data$weekstartdate, 
                                                                       max(Data$weekstartdate) + (1:n.ahead)))
  return(output)
}
result.stl <- forecastStl(Data, n.ahead = 90)



model2 =
  h2o.deeplearning(x = 4:24,  # column numbers for predictors
                   y = 25,   # column number for label
                   model_id = "model2",
                   l2 = 1e-3,       #L2 regularization
                   rate_decay = 10.0,
                   training_frame = train1, # data in H2O format
                   activation = "RectifierWithDropout", # algorithm
                   input_dropout_ratio = 0.1, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.2), # % for nodes dropout,
                   variable_importances = TRUE,
                   hidden = c(500,300), # 3 layers of 700,500,300  nodes respectively
                   momentum_stable = 0.99,   #Final momentum after the amp is over
                   stopping_tolerance = 1e-2,
                   stopping_rounds=4,      #Early stopping based on convergence
                   fold_assignment = "AUTO",
                   nesterov_accelerated_gradient = T, # use it for speed
                   epochs = 600, # no. of epochs
                   seed=1234)

pred_value <- predict(model2, h2o.test)
pred_value_df= as.data.frame(pred_value)
fina_submit<- data.frame(city = test$city , year = test$year , weekofyear = test$weekofyear, total_cases = round(pred_value_df$predict))
write.csv(fina_submit,file = "h2o_model2.csv")

#-------------------------

rf3 <- h2o.randomForest(  
  training_frame = train1,        
  x=4:24,                        ## the predictor columns, by column index
  y=25,                          ## the target index (what we are predicting)
  model_id = "rf3",             ## name the model in H2O
  ntrees = 60,                  ## use a maximum of 200 trees to create the
  ##  the random forest is sufficiently accurate
  balance_classes = TRUE,
  max_depth = 7,
  stopping_metric = c("MAE"),
  fold_assignment = c("AUTO"),
  stopping_tolerance = 0.001,
  sample_rate=0.41,
  stopping_rounds = 10,           ## Stop fitting new trees when the 2-tree
  ##  average is within 0.001 (default) of 
  ##  the prior two 2-tree averages.
  ##  Can be thought of as a convergence setting
  score_each_iteration = T,      ## Predict against training and validation for
  ##  each tree. Default will skip several.
  seed = 12345)                ## Set the random seed so that this can be  reproduced.

pred_value <- predict(rf3, h2o.test)
pred_value_df= as.data.frame(pred_value)
fina_submit<- data.frame(city = test$city , year = test$year , weekofyear = test$weekofyear, total_cases = round(pred_value_df$predict))
write.csv(fina_submit,file = "h2o_rf_2.csv")


#---------------------------------------------

gbm2<- h2o.gbm(
  training_frame = train1,     
  x=4:24,                     ##predictors
  y=25,                       ## target variable
  ntrees = 30,                ## decrease the trees, mostly to allow for run time  ##  (from 50)
  learn_rate = 0.2,           ## increase the learning rate (from 0.1)
  max_depth = 10,             ## increase the depth (from 5)
  stopping_rounds = 2,      
  stopping_tolerance = 0.01,
  score_each_iteration = T, 
  model_id = "gbm2",  ##
  seed = 1234)


h2o_gbm_predict <- h2o.predict(gbm2, h2o.test)
test_frame_gbm = as.data.frame(h2o_gbm_predict)
final_submit<- data.frame(city = test$city , year = test$year , weekofyear = test$weekofyear, total_cases = round(pred_value_df$predict))
write.csv(final_submit,file = "h2o_glm2.csv")

#--------------
