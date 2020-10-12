####### NAME: ONOJA ANTHONY ##########################################

####### PhD Data Science, Scuola Normale Superiore, Pisa-Italy #######

###### Date: 13/10/2020 ##############################################

##### Course: Statistical Methods for Data Science ###################

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

# Regularization Models 

# Fit Ridge regression with alpha = 0, fit a LASSO with alpha = 1

library(glmnet)
X = model.matrix(grouping ~., data = train)
y = train$grouping 

fit_ridge1 =glmnet(x= X,y,alpha = 0 )
#Visualize plot of Ridge 
plot(fit_ridge1, xvar = "lambda", label = TRUE)

#Now carry out Cross-validation on the ridge 
cv.ridge_1 = cv.glmnet(x=X, y=y, alpha = 0)
plot(cv.ridge_1)
# Display the best lambda value
cv.ridge_1 $lambda.min

# Fit the final model on the training data
ridge_model <- glmnet(x=X, y=y, alpha = 0, lambda = cv.ridge_1$lambda.min)
# Display regression coefficients
coef(ridge_model)

# Make predictions on the test data
X_test <- model.matrix(grouping ~., test)
predictions <- ridge_model %>% predict(X_test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test$grouping),
  Rsquare = R2(predictions, test$grouping)
)

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
predictions <- lasso_model %>% predict(X_test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test$grouping),
  Rsquare = R2(predictions, test$grouping)
)


# this method controls everything about training
# Build the model using the training set
set.seed(123)
elnet_model <- train(
  grouping ~., data = train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
# Best tuning parameter
elnet_model$bestTune

# Coefficient of the final model. You need
# to specify the best lambda
coef(elnet_model$finalModel, elnet_model$bestTune$lambda)
plot(elnet_model)

# Make predictions on the test data
x_test <- model.matrix(grouping ~., test)
predictions <- elnet_model %>% predict(x_test)
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test$grouping),
  Rsquare = R2(predictions, test$grouping)
)

elasticnet = coef(elnet_model$finalModel, elnet_model$bestTune$lambda)
elasticnet = data.frame(as.matrix((elasticnet)))
#write.csv(elasticnet, "C:/Users/Hp/Downloads/elasticnet.csv", row.names=TRUE) #save scores as .csv file in your local folder
