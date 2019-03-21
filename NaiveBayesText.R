# CS412/512 Machine Learning, 2017-2018 Fall
# Recitation - October 12, 2017

library(e1071) # for naiveBayes function
library(caret) # splitting the data with stratified sampling with createDataPartition function
library(tm) # for basic NLP tasks
library(MLmetrics) # ML evaluation metrics (ConfusionMatrix, Accuracy, etc.)

# Load the data
rawData <- read.csv("sms_spam.csv")

sapply(rawData, class) # Check the classes of variables
rawData$text <- iconv(rawData$text, to="utf-8") # Convert the text to utf-8 format
sapply(rawData, class) # Check the classes of variables again

table(rawData$type) # Check the type values and their distributions
prop.table(table(rawData$type))*100 # Check the type values and their distributions as percentage

set.seed(1234)
# Create a training set with 75% of the data (stratified sampling)
trainIndex <- createDataPartition(rawData$type, p=.75, list=FALSE, times = 1)
trainData <- rawData[trainIndex,]
testData <- rawData[-trainIndex,]

# Check the distributions in training and test set
prop.table(table(trainData$type))*100
prop.table(table(testData$type))*100

# Apply some preprocessing to text data (Cleaning the data)
corpus.train <- Corpus(VectorSource(trainData$text)) # create the corpus
print(corpus.train) # info about the corpus
corpus.train <- tm_map(corpus.train, content_transformer(tolower)) # convert all letters to lowercase
corpus.train <- tm_map(corpus.train, removeNumbers) # remove numbers
corpus.train <- tm_map(corpus.train, removeWords, stopwords()) # remove stopwords
corpus.train <- tm_map(corpus.train, removePunctuation) # remove punctuation
corpus.train <- tm_map(corpus.train, stripWhitespace) # normalize whitespaces
dtm.train <- DocumentTermMatrix(corpus.train) # Creation of DTM (tokenization)

# Same preprocessing applied to test set
corpus.test <- Corpus(VectorSource(testData$text))
print(corpus.test)
corpus.test <- tm_map(corpus.test, content_transformer(tolower))
corpus.test <- tm_map(corpus.test, removeNumbers)
corpus.test <- tm_map(corpus.test, removeWords, stopwords())
corpus.test <- tm_map(corpus.test, removePunctuation)
corpus.test <- tm_map(corpus.test, stripWhitespace)
dtm.test <- DocumentTermMatrix(corpus.test)

print(dtm.train) # info about DTM (there are 4182 documents and 6813 terms)
inspect(dtm.train[1:10, 5:10])

sms_features <- findFreqTerms(dtm.train, 5) # find words with frequency of 5 and more
summary(sms_features)
head(sms_features)

# Create DTM again with new word features
dtm.train <- DocumentTermMatrix(corpus.train, list(dictionary=sms_features))
dtm.test <- DocumentTermMatrix(corpus.test, list(dictionary=sms_features))

print(dtm.train) # info about DTM (now there are 4182 documents and 1232 terms)
inspect(dtm.train[1:10, 5:10])

# Convert counts to yes/no factor
convert_counts <- function(x){
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0,1), labels = c("No", "Yes"))
  return (x)
}

dtm.train <- apply(dtm.train, MARGIN = 2, convert_counts)
dtm.test <- apply(dtm.test, MARGIN = 2, convert_counts)

head(dtm.train[, 5:10])

# Train the Naive Bayes model
nb.model <- naiveBayes(dtm.train, trainData$type)

# Evaluate the model
nb.prediction <- predict(nb.model, dtm.test)
ConfusionMatrix(nb.prediction, testData$type)
Accuracy(nb.prediction, testData$type)
