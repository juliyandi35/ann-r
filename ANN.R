library(nnet)
library(NeuralNetTools)

data("iris")
iris

sampel <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))
sampel

#data training & testing
iris.train <- iris[sampel,]
iris.test <- iris[-sampel,]
iris.train
iris.test

iris1 <- nnet(Species~.,data=iris.train, size=2, decay=5, maxit=200)
iris1

plotnet(iris1)

#hasil prediksi terkait dengan data testing
y <- iris.test$Species
p <- predict(iris1,iris.test, type="class")
tabelhasil <- table (y,p)
sum(diag(table(y,p)))/75
jumlaherror <- mean(y!=p)
jumlaherror

Sepal.Length <- 4.9
Sepal.Width <-3.3
Petal.Length <- 1.3
Petal.Width <- 0.2
class(iris)
bungabaru <-data.frame(Sepal.Length,Sepal.Width,Petal.Length,Petal.Width)
bungabaru
cobaprediksi <- predict(iris1,bungabaru)
cobaprediksi

