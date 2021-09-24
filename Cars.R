rm(list = ls())

library(ISLR)
library(tree)
library(MASS)
library(rpart)
library(randomForest)
library(class) 
library(kknn)
library(dplyr)
library(ggplot2)
library(tree)
library(tidyverse)
library(caret)
library(rpart)
library(gbm)

my_ggtheme <- function() {
  theme_minimal(base_family = "Fira Sans") +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(vjust = -1)
    )
}

##############################################

df = read.csv("~/Desktop/MSBA/Summer/STA-Summer/Project/Cars/car_data_modified.csv", stringsAsFactors=TRUE)

names(df)[names(df) == "ï..Brand"] <- "Brand"
df$Brand=as.factor(df$Brand)
df$Body=as.factor(df$Body)
df$Engine.Type=as.factor(df$Engine.Type)
df$Model=as.factor(df$Model)        ####################
df$Registration=as.factor(df$Registration)
str(df)


######################################
# Exploratory Analysis
######################################
#Read the dataset and get high level understanding
head(df) #six first observations
tail(df) #six last observations. Now we cans see there is 'NA'.

class(df) #see the class
str(df) #see the structure of data frame

#Convert character variables to factors
df$Brand <- as.factor(df$Brand)
df$Body <- as.factor(df$Body)
df$Engine.Type <- as.factor(df$Engine.Type)
df$Registration <- as.factor(df$Registration)
df$Model <- as.factor(df$Model) #312 levels

#Model column is categorical variable and having 312 unique values, 
#which implies, after converting it to dummy, 
#it will add 312 new columns to the dataframe, 
#so we will drop this column.
df <- df[,-9]


#Check missing values
colSums(is.na(df))


#Viewing trends in price, mileage, enginev, age
par(mfrow=c(2,2))
plot(density(df$Price), main='Price Density Spread')
plot(density(df$Mileage), main='Mileage Density Spread')
plot(density(df$EngineV), main='EngineV Density Spread')
plot(density(df$Age), main='Age Density Spread')

#From above, we can say that price and enginev
#are not normally distributed, 
#so we need to remove some outliers from data.
car_data1 <- df[df$Price < 161000, ]
car_data2 <- car_data1[car_data1$EngineV < 8, ]
car_data_modified <- na.omit(car_data2)

str(df)
summary(df)

#Use the pairs() function to produce a scatterplot matrix
pairs(df)

par(mfrow=c(2,2))

plot(car_data_modified$Mileage, car_data_modified$Price,
     pch = 19,
     col = "darkgray",
     xlim = range(car_data_modified$Mileage),
     ylim = range(car_data_modified$Price)
)

plot(car_data_modified$EngineV, car_data_modified$Price,
     pch = 19,
     col = "darkgray",
     xlim = range(car_data_modified$EngineV),
     ylim = range(car_data_modified$Price)
)

plot(car_data_modified$Age, car_data_modified$Price,
     pch = 19,
     col = "darkgray",
     xlim = range(car_data_modified$Age),
     ylim = range(car_data_modified$Price)
)




####################################
### OLS / Regression
####################################
set.seed(48)
n <- nrow(df)
train <- sample(1:n, n*0.8)

cars.train <- df[train,]
cars.test <- df[-train,]


lm2 <- lm(Price ~ factor(Brand) + factor(Body) + Mileage + EngineV + 
            factor(Registration) + Age, data = cars.train)

lm2_tidy <- tidy(lm2)
lm2_tidy <- lm2_tidy %>% select(-std.error, -statistic) %>%
  mutate(p.value = round(p.value, 4),
         estimate = round(estimate, 4))

table <- reactable(lm2_tidy, style = list(fontFamily = "Fira Sans"), defaultPageSize = 16)

###############################
### KNN
###############################
knn_model_2 <- train(
  Price ~ factor(Brand) + factor(Body) + Mileage + EngineV + 
    factor(Registration) + Age,
  data = cars.train,
  method = "knn",
  type = "anova", # tells knn this is regression
  trControl = trainControl("cv", number = 10),
  preProcess = c("center", "scale"),
  tuneGrid  = expand.grid(k = c(1, 100, 1)),
  try = 50
)
plot(knn_model_2)

ggplot(knn_model_2) + my_ggtheme()

knn.predict <- knn_model_2 %>% predict(cars.test)
RMSE(knn.predict, cars.test$Price)


knn_normal_df <- copy(cars.test)
knn_normal_df <- knn_normal_df %>%
  mutate(
    Predicted = knn_model_2 %>% predict(knn_normal_df),
    difference = Predicted - Price
  )

knn_normal_df %>% 
  ggplot(aes(x = Price, y = difference)) +
  geom_point(color = "#002055")+
  labs(
    title = "KNN Model",
    y = "Residuals",
    x = "Actual Worth ($)"
  ) + 
  my_ggtheme() +
  scale_x_continuous(labels = comma)


knn_normal_df %>% 
  ggplot(aes(x = Price, y = difference)) + 
  geom_point(color = "dark orange", alpha = 0.8) + 
  my_ggtheme()

RMSE(knn_normal_df$Predicted, knn_normal_df$Price)

####################################
###### BOOSTING 
####################################
# create subset samples train, validation, test
set.seed(5)
n = nrow(df)
n1 = floor(n/2)
n2 = floor(n/4)
n3 = n-n1-n2
ii = sample(1:n,n)
carstrain =df[ii[1:n1],]
carsval = df[ii[n1+1:n2],]
carstest = df[ii[n1+n2+1:n3],]
carstrainval = rbind(carstrain,carsval)

# finding optimal boosting parameters for validation set
treedepth = c(2, 4, 10)
treesnum = c(50, 500, 5000)
treelambda = c(.001, .2)
parmb = expand.grid(treedepth,treesnum,treelambda)
colnames(parmb) = c('tdepth','ntree','lam')
print(parmb)
nset = nrow(parmb)
olb = rep(0,nset)
ilb = rep(0,nset)
bfitv = vector('list',nset)

# create boosting parameters using values specified above 
for(i in 1:nset) {
  cat('doing boost ',i,' out of ',nset,'\n')
  #boosttrain = gbm(Price~.-Engine.Type,data=carstrain,distribution='gaussian',interaction.depth=parmb[i,1],n.trees=parmb[i,2],shrinkage=parmb[i,3])
  boosttrain = gbm(Price~., data=carstrain, distribution='gaussian', interaction.depth=parmb[i,1], n.trees=parmb[i,2], shrinkage=parmb[i,3])
  ifit = predict(boosttrain, n.trees=parmb[i,2])
  ofit = predict(boosttrain, newdata=carsval, n.trees=parmb[i,2])
  olb[i] = sum((carsval$Price-ofit)^2)
  ilb[i] = sum((carstrain$Price-ifit)^2)
  bfitv[[i]] = boosttrain
}

# compute for rmse in vs out of sample
ilb = round(sqrt(ilb/nrow(carstrain)),3) 
olb = round(sqrt(olb/nrow(carsval)),3)

# print rmse values and find where it is at lowest
print(cbind(parmb, olb, ilb))
which.min(olb)

# which variables have the most influence
vimportance = summary(boosttrain)
# try out different predictor variables

# Best Out of Sample RMSE for Price(all predictors):  6184.931
# depth: 10, trees: 50, shrinkage: .2

# Best Out of Sample RMSE for Price(all but Age):  10022.04
# depth: 4, trees: 5000, shrinkage: .001

# Best Out of Sample RMSE for Price(all but Mileage): 6390.090
# depth: 10, trees: 50, shrinkage: .2

# Best Out of Sample RMSE for Price(all but Registration):  6410.236
# depth: 10, trees: 5000, shrinkage: .001

# Best Out of Sample RMSE for Price(all but Registration and Engine Type):  6365.701
# depth: 10, trees: 50, shrinkage: .2

# Best Out of Sample RMSE for Price(all but Engine Type):  6294.828
# depth: 10, trees: 5000, shrinkage: .001


#--------------------------------------------------
# use best fit on test
boosttest = gbm(Price~.,data=carstrainval,distribution='gaussian', interaction.depth=10,n.trees=50,shrinkage=.2)
boosttestpred=predict(boosttest,newdata=carstest,n.trees=50)
boosttestrmse = sqrt(sum((carstest$Price-boosttestpred)^2)/nrow(carstest))
print(boosttestrmse)

# plot actual vs predicted values from boosting model
plot(carstest$Price,boosttestpred,xlab='Test Price',ylab='Boosting Prediction')
abline(0,1, col='red')
vimportance=summary(boosttest)

# Test RMSE for Price(all predictors): 6032.43
# Test RMSE for Price(all but Age): 8915.955
# Test RMSE for Price(all but Mileage): 6582.323
# Test RMSE for Price(all but Registration): 6216.434
# Test RMSE for Price(all but Registration and Engine Type): 6376.156
# Test RMSE for Price(all but Engine Type): 6127.151


# Conclusion: Price as a function of all predictors still turned out to have the lowest RMSE
# When fitting our best predictive model parameters on all 7 predictors to our test set, our RMSE is 6032.43



####################################
#####--SIMPLE REGRESSION TREE--#####
####################################

#SETTING UP TRAIN AND TEST
set.seed(32)
train = sample(1:nrow(df), nrow(df)/1.3)

#BIG TREE
temp = tree(Price~.,data=df, subset = train, mindev=.0001)
summary(temp)
plot(temp)
text(temp)

##CROSS VALIDATION. SIMPLEST MODEL WITH LOWEST DEVIATION = 9
cv_car = cv.tree(temp)
plot(cv_car$size, cv_car$dev, type='b')


#PRUNING
prune_car = prune.tree(temp, best=9)
plot(prune_car)
text(prune_car)

#PREDICTION
yhat = predict(prune_car, newdata = df[-train,])

#yhat = 10^yhat
car.test = df[-train, 'Price']
plot(yhat,car.test)
abline(0,1)

sqrt(mean((yhat-car.test)^2))

####################################
####################################
####--BAGGING & RANDOM FOREST--#####
####################################
####################################





df$Brand=as.factor(df$Brand)
df$Body=as.factor(df$Body)
df$Engine.Type=as.factor(df$Engine.Type)
df$Registration=as.factor(df$Registration)
str(df)

#SETTING UP TRAIN AND TEST
set.seed(32)
train = sample(1:nrow(df), nrow(df)/1.3)

##BUILDING THE MODEL
bag_car = randomForest(Price~., data = df, subset = train,
                       mtry= 7, importance = TRUE)

##PREDICTION
yhat_bag = predict(bag_car, newdata = df[-train,])
car.test = df[-train, 'Price']
plot(yhat_bag,car.test)
abline(0,1, col='red')
sqrt(mean((yhat_bag-car.test)^2))

varImpPlot(bag_car)


#######################################
## RANDOM FOREST
######################################


set.seed(11)
n=nrow(df)
n1=floor(3*n/5)
n2=floor(n/5)
n3=n-n1-n2
ii = sample(1:n,n)
train=df[ii[1:n1],]
val = df[ii[n1+1:n2],]
test = df[ii[n1+n2+1:n3],]


trainval = rbind(train,val)
finrf = randomForest(Price~.,data=trainval,ntree=5000)

par(mfrow=c(1,1))
plot(finrf)

plot()

#Getting Convergence at tree size = 1500

testpred = predict(finrf, test)
finrfrmse = sqrt(sum((test$Price-testpred)^2)/nrow(test))

varImpPlot(finrf)

# TEST RMSE = 6294.555  at set seed = 11
# Registration (1) and Engine.Type (2) are least important Variable.



#####################################################
###  Validation

mtryv = c(3,4,5,6,7)
ntreev = c(1500,2000)
parmrf = expand.grid(mtryv,ntreev)
colnames(parmrf)=c('mtry','ntree')
nset = nrow(parmrf)
olrf = rep(0,nset)
ilrf = rep(0,nset)
tlrf = rep(0,nset)
rffitv = vector('list',nset)
for(i in 1:nset) {
  cat('doing rf ',i,' out of ',nset,'\n')
  temprf = randomForest(Price~.,train,mtry=parmrf[i,1],ntree=parmrf[i,2])
  ifit = predict(temprf)
  ofit = predict(temprf,val)
  olrf[i] = sum((val$Price-ofit)^2)
  ilrf[i] = sum((train$Price-ifit)^2)
  rffitv[[i]]=temprf
}
ilrf = round(sqrt(ilrf/nrow(train)),3)
olrf = round(sqrt(olrf/nrow(val)),3)

#--------------------------------------------------

print(cbind(parmrf,olrf,ilrf))
which.min(olrf)

# Out of Sample RMSE= 5940.971
# Best was m = 5 and Tree size = 2000

temprf = randomForest(Price~.,train,mtry=5,ntree=2000)
finrfpred = predict(temprf,test)
sqrt(sum((test$Price-finrfpred)^2)/nrow(test))
varImpPlot(temprf)

# Test RMSE = 6157.296 


########################################################
# Running the same Model but with Log Price

print(cbind(parmrf,olrf,ilrf))
which.min(olrf)

# Best was m = 6 and Tree size = 1500

temprf = randomForest(Price~.,train,mtry=4,ntree=1500)
varImpPlot(temprf)

finrfpred = predict(temprf,test)
sqrt(sum((exp(test$Price)-exp(finrfpred))^2)/nrow(test))

# Test RMSE = 6237.396
# It got slighlty worse.


########################################################
# Running after removing Engine Type
# Got m = 4 and n = 1500

print(cbind(parmrf,olrf,ilrf))
which.min(olrf)

# Best was m = 4 and Tree size = 2000

temprf = randomForest(Price~.,train,mtry=4,ntree=2000)
varImpPlot(temprf)

finrfpred = predict(temprf,test)
sqrt(sum((test$Price-finrfpred)^2)/nrow(test))

# Test MSE = 6785.248
# Got Much Worse.



########################################################
# Conclusion
# Best Model is All Variables, no Transformations, m = 5, @ Tree Size = 2000
# Test RMSE 6178

Final_RF = randomForest(Price~.,train,mtry=5,ntree=2000)
RF_Predict = predict(Final_RF,test)
sqrt(sum((test$Price-RF_Predict)^2)/nrow(test))
varImpPlot(Final_RF)


plot(test$Price,RF_Predict,xlab='Test_Data_Set Price',ylab='RF Predictions')
abline(0,1,col='red',lwd=2)

