library(neuralnet)
library(caret)
library(readxl)

data <- read_excel("Data Laporan 1505J.xlsx")
data <- data[,-c(1,4)]
head(data)
names(data)
colnames(data) <- c("Y","X1","X2","X3","X4","X5","X6","X7")
data <- data.frame(data)
str(data)
summary(data)
colSums(is.na(data))

data$X1 <- (data$X1 - min(data$X1))/(max(data$X1) - min(data$X1))

# Data Partition
set.seed(12)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train <- data[ind==1,]
head(train)
test <- data[ind==2,]
head(test)

set.seed(12)#pengaturan banyaknya pengacakan
nn1 <- neuralnet(Y~.,
                data = train,
                hidden = 1, #hidden layer 5(bisa diubah)
                err.fct = "ce", #kalkulasi eror dengan metode cross enthopy
                linear.output = FALSE)
library(devtools)
plot(nn1,information = FALSE, show.weights = FALSE)

nn2 <- neuralnet(Y~.,
                 data = train,
                 hidden = 2, #hidden layer 5(bisa diubah)
                 err.fct = "ce", #kalkulasi eror dengan metode cross enthopy
                 linear.output = FALSE)

plot(nn2,information = FALSE, show.weights = FALSE)

