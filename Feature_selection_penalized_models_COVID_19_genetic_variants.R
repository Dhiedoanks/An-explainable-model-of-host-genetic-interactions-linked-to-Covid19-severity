####### NAME: ONOJA ANTHONY ##########################################

####### PhD Data Science, Scuola Normale Superiore, Pisa-Italy #######

#### DATE: 13/10/2020#################################################

###### COURSE: STATISTICAL METHODS FOR DATA SCIENCE ##################

#import libraries 
library(ISLR)
library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)
library(dataPreparation)
df = read.csv("C:/Users/Hp/Downloads/latest_feature_count_matrx43012.csv")
names(df)
dim(df)
class(df)
summary(df)

#checking for missing observations in the data feature count matrix 
sum(is.na(df))
#view data head
print(head(df, n = 4))


#omit the sample_ID column
df_1 = df[,2:1817]

# create a filter for removing highly correlated variables
# if two variables are highly correlated only one of them
# is removed
corrFilt=preProcess(df_1, method = "corr",cutoff = 0.9)
df_1=predict(corrFilt,df_1)

#Prepare data for train and test split 
# Random sample indexes
train_index <- sample(1:nrow(df_1), 0.8 * nrow(df_1))
test_index <- setdiff(1:nrow(df_1), train_index)
train <- df_1[train_index, ]
test = df_1[test_index, ]
# Build X_train, y_train, X_test, y_test
X_train <- df_1[train_index, -1816]
y_train <- df_1[train_index, "grouping"]

X_test <- df_1[test_index, -1816]
y_test <- df_1[test_index, "grouping"]

#Scale train and test dataset 
scales <- build_scales(dataSet = X_train, verbose = TRUE)
print(scales)


#apply scaling on X_train 
X_train <- fastScale(dataSet = X_train, scales = scales, verbose = TRUE)

#Scale test set

X_test <- fastScale(dataSet = X_test, scales = scales, verbose = TRUE)

###################### Regularization Models ########################### 



library(glmnet)
X = model.matrix(grouping ~., data = train)
y = train$grouping 


###### FIt LASSO ########
set.seed(123) 
fit_lasso = glmnet(x=X, y=y, alpha = 1)
plot(fit_lasso, xvar = "lambda", label = TRUE)

######### Cross-validation for LASSO #########
cv_lasso = cv.glmnet(x=X, y= y, alpha = 1)

###Visualize the plot of CV LASSO ## 
plot(cv_lasso)

cv_lasso$lambda.min
### The plot shows that the best model is at the minimum of 6.3 log(lambda) within 1-standard deviation error we have a 691
cv = coef(cv_lasso)
library(Matrix)
cv_11= data.frame(as.matrix((cv)))

#write.csv(cv_11, "C:/Users/Hp/Downloads/cv_11.csv", row.names=TRUE) #save scores as .csv file in your local folder

# Fit the final LASSO model on the training data
lasso_model <- glmnet(x=X, y=y, alpha = 1, lambda = cv_lasso$lambda.min)
# Dsiplay regression coefficients
cv = coef(lasso_model)
cv_12= data.frame(as.matrix((cv)))
#write.csv(cv_12, "C:/Users/Hp/Downloads/cv_12.csv", row.names=TRUE) #save scores as .csv file in your local folder
cv_12

# Make predictions on the test data
X_test <- model.matrix(grouping ~., test)

predictions <- lasso_model %>% predict(X_test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test$grouping),
  Rsquare = R2(predictions, test$grouping)
)

# Fit Ridge regression with alpha = 0, fit a LASSO with alpha = 1 
fit.ridge =glmnet(x= X,y,alpha = 0 )
#Visualize plot of Ridge 
plot(fit.ridge, xvar = "lambda", label = TRUE)

#Now carry out Cross-validation on the ridge 
cv.ridge = cv.glmnet(x=X, y=y, alpha = 0)
plot(cv.ridge)
# Display the best lambda value
cv.ridge$lambda.min

# Fit the final model on the training data
ridge.model <- glmnet(x=X, y=y, alpha = 0, lambda = cv.ridge$lambda.min)
# Display regression coefficients
coef(ridge.model)

# Make prediction on test set
predictions <- ridge.model %>% predict(X_test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test$grouping),
  Rsquare = R2(predictions, test$grouping)
)




### Fit the Elastic Net Model #########
fit.elnet = glmnet(x = X, y, family="gaussian", alpha=.5)

#cross validate for Elastic NET
fit.elnet.cv <- cv.glmnet(x=X, y, type.measure="mse", alpha=.5,
                          family="gaussian")
plot(fit.elnet.cv)
# this method controls everything about training
# Build the model using the training for Elastic Net hyper-parameter tuning 
set.seed(123)
elnet.model <- train(
  grouping ~., data = train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
# Best tuning parameter
elnet.model$bestTune

# Coefficient of the final model. You need
# to specify the best lambda
coef(elnet.model$finalModel, elnet.model$bestTune$lambda)
plot(elnet.model)
# Make predictions on the test data
x_test <- model.matrix(grouping ~., test)
predictions <- elnet.model %>% predict(x_test)
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test$grouping),
  Rsquare = R2(predictions, test$grouping)
)

elasticnet = coef(elnet.model$finalModel, elnet.model$bestTune$lambda)
elasticnet = data.frame(as.matrix((elasticnet)))
#write.csv(elasticnet, "C:/Users/Hp/Downloads/elasticnet.csv", row.names=TRUE) #save scores as .csv file in your local folder

# Visulize LASSO, Ridge and ELastic NET in one plot 
for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(x = X, y, type.measure="mse", 
                                            alpha=i/10,family="gaussian"))
}

par(mfrow=c(3,2))
# For plotting options, type '?plot.glmnet' in R console
plot(fit.lasso, xvar="lambda")
plot(fit10, main="LASSO")

plot(fit.ridge, xvar="lambda")
plot(fit0, main="Ridge")

plot(fit.elnet, xvar="lambda")
plot(fit5, main="Elastic Net")