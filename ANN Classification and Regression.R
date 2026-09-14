library(tensorflow)
library(keras)
library(reticulate)
library(caret)

### Model klasifikasi ###
data <- read_excel("Data Laporan 1505J.xlsx")
data <- data[,-c(1,4)]
head(data)
names(data)
colnames(data) <- c("Y","X1","X2","X3","X4","X5","X6","X7")
data <- data.frame(data)
str(data)
for (i in c(1,3:8)) {
  data[,i] <- as.factor(data[,i])
}
summary(data)
colSums(is.na(data))

# Merubah features kategorik menjadi OHE
data$X2 <- to_categorical(as.integer(data$X2)-1)
data$X3 <- to_categorical(as.integer(data$X3)-1)
data$X4 <- to_categorical(as.integer(data$X4)-1)
data$X5 <- to_categorical(as.integer(data$X5)-1)
data$X6 <- to_categorical(as.integer(data$X6)-1)
data$X7 <- to_categorical(as.integer(data$X7)-1)

# Contoh hasil OHE pada kolom MTRANS
head(data$Y)

# Partisi Data dan Features Scaling
# Membagi data menjadi data latih dan data uji dengan createDataPartition
set.seed(123)

train.index <- createDataPartition(data$Y, p = 0.7, list = FALSE)
train <- data[train.index, ]
test <- data[-train.index, ]

# Melakukan Feature Scaling min max (0, 1)
preprocessParams <- preProcess(train[, -1], method=c("range"))
train_X <- as.matrix(predict(preprocessParams, train[, -1]))
test_X <- as.matrix(predict(preprocessParams, test[, -1]))

# OHE pada peubah target
train_y <- to_categorical(as.integer(train[, 1])-1)
test_y <- to_categorical(as.integer(test[, 1])-1)

# Pemodelan dengan 1 Hidden Layer
# Membuat model neural network dengan 1 hidden layer
model <- keras_model_sequential() %>%
  layer_dense(units = 10, activation = "relu", input_shape = ncol(train_X)) %>% # Hidden Layer 1
  layer_dense(units = ncol(train_y), activation = "softmax")                    # Output Layer

# Mengkompilasi model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

print(model)

# Melatih model
history <- model %>% fit(
  train_X, train_y,
  shuffle = T,
  epochs = 100,          
  batch_size = 10,       
  validation_split = 0.2
)

# Menampilkan plot pembelajaran model pada setiap epoch
plot(history)

# Mengevaluasi model menggunakan data uji
scores <- model %>% evaluate(test_X, test_y)
print(scores)

# Melakukan Prediksi
prediksi <- predict(model, test_X)
head(prediksi)

label_pred <- apply(prediksi, 1, which.max)
label_true <- as.integer(test$Y)

# Evaluasi Model
confusionMatrix(as.factor(label_true), as.factor(label_pred))

# Membuat model neural network dengan 2 hidden layer
model <- keras_model_sequential() %>%
  layer_dense(units = 20, activation = "relu", input_shape = ncol(train_X)) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = ncol(train_y), activation = "softmax")

# Mengkompilasi model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# Melakukan tahapan pelatihan model
history <- model %>% fit(
  train_X, train_y,
  shuffle = T,
  epochs = 100,
  batch_size = 10,
  validation_split = 0.2, 
  verbose = F       # tidak menampilkan teks ouput pada setiap epoch
)

print(model)

# Menampilkan plot pembelajaran model pada setiap epoch
plot(history)

# Mengevaluasi model menggunakan data uji
scores <- model %>% evaluate(test_X, test_y)
print(scores)

# Melakukan Prediksi
prediksi <- predict(model, test_X)
head(prediksi)

label_pred <- apply(prediksi, 1, which.max)
label_true <- as.integer(test$Y)

# Evaluasi Model
confusionMatrix(as.factor(label_true), as.factor(label_pred))

### Model Regresi ###
# Membuat model neural network dengan 1 hidden layer
model_lm <- keras_model_sequential() %>%
  layer_dense(units = 10, activation = "relu", input_shape = ncol(train_X)) %>%
  layer_dense(units = 1, activation = "linear")

# Mengkompilasi model
model_lm %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam",
  metrics = list("mean_squared_error", "mean_absolute_error")
)

# Melakukan tahapan pelatihan model
history_lm <- model_lm %>% fit(
  train_X, train_y,
  shuffle = T,
  epochs = 100,
  batch_size = 10,
  validation_split = 0.2
)

print(model_lm)

# Menampilkan plot pembelajaran model pada setiap epoch
plot(history_lm)

# Melakukan prediksi
prediksi_lm <- predict(model_lm, test_X)
head(prediksi_lm)

# Mengevaluasi model menggunakan data uji
scores_lm <- model_lm %>% evaluate(test_X, test_y)
print(scores_lm)

# Evaluasi Model
label_pred_lm <- NULL
for (i in 1:length(prediksi_lm)) {
  if(prediksi_lm[i]>0.5){
    label_pred_lm[i] <- 2 # Ya
  }
  else{
    label_pred_lm[i] <- 1 # Tidak
  }
}

label_true_lm <- as.integer(test$Y)
confusionMatrix(as.factor(label_true_lm), as.factor(label_pred_lm))


# Membuat model neural network dengan 2 hidden layer
model_lm <- keras_model_sequential() %>%
  layer_dense(units = 20, activation = "relu", input_shape = ncol(train_X)) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = "linear")

# Mengkompilasi model
model_lm %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam",
  metrics = list("mean_squared_error", "mean_absolute_error")
)

# Melakukan tahapan pelatihan model
history_lm <- model_lm %>% fit(
  train_X, train_y,
  shuffle = T,
  epochs = 100,
  batch_size = 10,
  validation_split = 0.2
)

print(model_lm)

# Menampilkan plot pembelajaran model pada setiap epoch
plot(history_lm)

# Melakukan prediksi
prediksi_lm <- predict(model_lm, test_X)
head(prediksi_lm)

# Mengevaluasi model menggunakan data uji
scores_lm <- model_lm %>% evaluate(test_X, test_y)
print(scores_lm)
