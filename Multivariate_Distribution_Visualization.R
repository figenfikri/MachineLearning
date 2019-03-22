library(mvtnorm)
library(car)
library(MASS)
library(klaR)
library(caret)
library(MLmetrics)

mu1 <- matrix(c(5,-5), nrow = 1, ncol = 2)
mu2 <- matrix(c(0,0), nrow = 1, ncol = 2)

sigma1 <- matrix(c(2,0,0,2), nrow = 2, ncol = 2)
sigma2 <- matrix(c(5,5,5,5), nrow = 2, ncol = 2)

N=500

set.seed(1234)

samp1 <- rmvnorm(N, mu1, sigma1)
samp2 <- rmvnorm(N, mu2, sigma2)

plot(samp1[,1], samp1[,2], xlim = c(-10,10), ylim = c(-10,10), pch = 1, col = "blue", xlab = "", ylab = "")
par(new=TRUE)
plot(samp2[,1], samp2[,2], xlim = c(-10,10), ylim = c(-10,10), pch = 2, col = "red", xlab = "", ylab = "")
legend(-9, 7, legend=c("d1", "d2"),col=c("blue", "red"), pch=1:2, cex=0.8)


sigma2 <- matrix(c(5,2,2,5), nrow = 2, ncol = 2)
samp2 <- rmvnorm(N, mu2, sigma2)

plot(samp1[,1], samp1[,2], xlim = c(-10,10), ylim = c(-10,10), pch = 1, col = "blue", xlab = "", ylab = "")
par(new=TRUE)
plot(samp2[,1], samp2[,2], xlim = c(-10,10), ylim = c(-10,10), pch = 2, col = "red", xlab = "", ylab = "")
legend(-9, 7, legend=c("d1", "d2"),col=c("blue", "red"), pch=1:2, cex=0.8)

sampmu1 <- matrix(c(sum(train1[,1])/N,sum(train1[,2])/N), nrow = 1, ncol = 2)
sampmu2 <- matrix(c(sum(train2[,1])/N,sum(train2[,2])/N), nrow = 1, ncol = 2)

sampcov1 <- matrix(0,2,2)
sampcov2 <- matrix(0,2,2)

for (i in 1:N) {
  sampcov1 <- sampcov1 + (t(samp1[i,]-sampmu1)) %*% (samp1[i,]-sampmu1)
  sampcov2 <- sampcov2 + (t(samp2[i,]-sampmu2)) %*% (samp2[i,]-sampmu2)
}

sampcov1 <- sampcov1/N
sampcov2 <- sampcov2/N


dataEllipse(samp1, levels=c(.90,.95,.99))
dataEllipse(samp2, levels=c(.90,.95,.99))

################################################

library(caret)
library(MASS)
library(MLmetrics)
library(klaR)

iris <- data.frame(rbind(iris3[,,1], iris3[,,2], iris3[,,3]), Species = rep(c("s","c","v"), rep(50,3)))

set.seed(1234)
#train <- sample(1:150, 100)
#test <- iris[-train, ]

trainIndex <- createDataPartition(iris$Species, p=.5, list=FALSE, times = 1)
train <- iris[trainIndex,]
test <- iris[-trainIndex,]

lda.fit <- lda(Species ~ ., train, prior = c(1,1,1)/3)
lda.predict <- predict(lda.fit, test)$class
lda.accuracy <- Accuracy(lda.predict, test$Species)

qda.fit <- qda(Species ~ ., train, prior = c(1,1,1)/3)
qda.predict <- predict(qda.fit, test)$class
qda.accuracy <- Accuracy(qda.predict, test$Species)

partimat(Species ~ ., data = test, method = "lda", prior = c(1,1,1)/3)
partimat(Species ~ ., data = test, method = "qda", prior = c(1,1,1)/3)
