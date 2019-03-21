# CS412/512 Machine Learning, 2017-2018 Fall
# Recitation - October 12, 2017

getwd() # check working directory
# Use setwd(" --directory-- ") to set working directory

# install the packages first; if you have already installed, no need to install again, just call the library
install.packages(e1071) # for naiveBayes function
library(e1071)
install.packages(farff) # read and write arff files
library(farff)
install.packages(MLmetrics) # ML evaluation metrics (ConfusionMatrix, Accuracy, etc.)
library(MLmetrics)

# Load the data
train <- readARFF("weather.arff")
test <- readARFF("weather-test.arff")

# Check dimension, number of rows and number of columns of train and test data
dim(train)
dim(test)
nrow(train)
nrow(test)
ncol(train)
ncol(test)

# Check the classes of variables
sapply(train, class)
sapply(test, class) 

# Display first few rows (default=6) of train and test data
head(train) 
head(test)

# Display last few rows (default=6) of train and test data
tail(train) 
tail(test)

# Train the Naive Bayes model
nb.model <- naiveBayes(train$play ~ ., data = train)
print(nb.model) # display the model
cat("\n")

# Evaluate the model
nb.prediction <- predict(nb.model, newdata = test, type = "class")
cat("Prediction: \n")
print(nb.prediction) # display the prediction
cat("\n")

# Check confusion matrix
confusionMatrix <- table('Actual Class' = test$play, 'Predicted Class' = nb.prediction)
print(confusionMatrix)
cat("\n")

# Another way of checking confusion matrix (ConfusionMatix is in MLmetrics)
confusionMatrix.ml <- ConfusionMatrix(nb.prediction, test$play)
print(confusionMatrix.ml)
cat("\n")

# Check accuracy
accuracy <- 1 - sum(test$play != nb.prediction)/nrow(test)
print(paste0("Accuracy: ", accuracy))
cat("\n")

# Another way of checking accuracy (Accuracy is in MLmetrics)
accuracy.ml <- Accuracy(nb.prediction, test$play)
print(paste0("Accuracy: ", accuracy.ml))