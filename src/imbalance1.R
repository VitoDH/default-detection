library(ROSE)
library(e1071)
library(randomForest)
library(pROC)
library(xgboost)

setwd("~/Documents/Study/ComplexDataAnalysis/Homework/big homework/IJCAI15 Data/data_format1")

trainSet <- read.csv("trainSet5.csv")
trainSet$X <- NULL
trainSet <- data.matrix(trainSet)

mapminmax <- function(x){
  minx <- min(x)
  maxx <- max(x)
  x <- (x-minx)/(maxx-minx)
}

trainSet[,3:12] <- apply(trainSet[,3:12],2,mapminmax)


pca <- princomp(trainSet[,3:12])
summary(pca)
screeplot(pca,type = "l",main = "Screeplot")
trainSet2 <- pca$scores[,1:4]
trainSet <- cbind(trainSet2,trainSet[,13])

train <- trainSet[1:90000,]
test <- trainSet[90001:90917,]

trainX <- train[,1:4];trainY <- train[,5] 
testX <- test[,1:4]; testY <- test[,5]


#---------check the distribution of the label
table(trainY)
prop.table(table(trainY))




train <- as.data.frame(train);colnames(train)<-c("x1","x2","x3","x4","label")
test <- as.data.frame(test);colnames(test)<-c("x1","x2","x3","x4","label")




numOfOne <- table(train$label)[2]
numOfZero <- table(train$label)[1]
#-----------------oversample 不停地抽1，少的那一类
data_balanced_over <- ovun.sample(label ~ ., data = train, method = "over",N = 2*numOfZero)$data
table(data_balanced_over$label)
data_balanced_over <- data.matrix(data_balanced_over)


#-------------------undersample 少点抽0
data_balanced_under <- ovun.sample(label ~ ., data = train, method = "under", N = 2*numOfOne, seed = 1)$data
table(data_balanced_under$label)
data_balanced_under <- data.matrix(data_balanced_under)


#-------------both over and under
data_balanced_both <- ovun.sample(label ~ ., data = train, method = "both", p=0.5,  seed = 1)$data
table(data_balanced_both$label)
data_balanced_both <- data.matrix(data_balanced_both)


#---------------ROSE
data.rose <- ROSE(label ~ ., data = train, seed = 1)$data
table(data.rose$label)
data.rose <- data.matrix(data.rose)


FScore <- function(x,beta=2){
  precision <- x$precision; recall <- x$recall
  F2 <- (1+beta^2)*(precision*recall)/(beta^2*precision+recall)
}


#-----------------------------xgboost
#-----------------rose
par(mfrow=c(1,2))
bst.rose <- xgboost(data = data.rose[,-5], label = data.rose[,5], max.depth = 5, eta = 0.1,nround = 800, objective = "binary:logistic")
pred.train.rose <- predict(bst.rose,data.rose[,-5])
modelrocTrainRose <- roc(data.rose[,5],pred.train.rose)
plot(modelrocTrainRose, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,main = "ROC Curve - ROSE - Train")
acrose.train <- accuracy.meas(data.rose[,5], pred.train.rose,threshold = 0.492)
F2Score <- FScore(acrose.train,2)


pred.rose <- predict(bst.rose, testX)
modelrocTestRose <- roc(testY,pred.rose)
plot(modelrocTestRose, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,main = "ROC Curve - ROSE - Test")
acrose.test <- accuracy.meas(testY, pred.rose,threshold = 0.413)
F2Score <- FScore(acrose.test,2)

#---------------------both
#---------------------find the ROC,AUC,FSCORE of BothSample

bst.both <- xgboost(data = data_balanced_both[,-5], label = data_balanced_both[,5], max.depth = 5, eta = 0.11,nround = 800, objective = "binary:logistic")
pred.train.both <- predict(bst.both,data_balanced_both[,-5])
modelrocTrainboth <- roc(data_balanced_both[,5],pred.train.both)
plot(modelrocTrainboth, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,main = "ROC Curve - BOTH - Train")
acboth.train <- accuracy.meas(data_balanced_both[,5], pred.train.both,threshold = 0.491)
F2Score <- FScore(acboth.train,2)

pred.both <- predict(bst.both, testX)
modelrocTestboth <- roc(testY,pred.both)
plot(modelrocTestboth, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,main = "ROC Curve - BOTH - Test")
acboth.test <- accuracy.meas(testY, pred.both,threshold = 0.506)
F2Score <- FScore(acboth.test,2)

#-------------------over
#---------------------find the ROC,AUC,FSCORE of OverSample

par(mfrow=c(1,2))
bst.over <- xgboost(data = data_balanced_over[,-5], label = data_balanced_over[,5], max.depth = 5, eta = 0.1,nround = 800, objective = "binary:logistic")
pred.train.over <- predict(bst.over,data_balanced_over[,-5])
modelrocTrainover <- roc(data_balanced_over[,5],pred.train.over)
plot(modelrocTrainover, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,main = "ROC Curve - OVER - Train")
acover.train <- accuracy.meas(data_balanced_over[,5], pred.train.over,threshold = 0.485)
F2Score <- FScore(acover.train,2)

pred.over <- predict(bst.over, testX)
modelrocTestover <- roc(testY,pred.over)
plot(modelrocTestover, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,main = "ROC Curve - OVER - Test")
acover.test <- accuracy.meas(testY, pred.over,threshold = 0.507)
F2Score <- FScore(acover.test,2)

#--------------------under
#---------------------find the ROC,AUC,FSCORE of UnderSample
par(mfrow=c(1,2))
bst.under <- xgboost(data = data_balanced_under[,-5], label = data_balanced_under[,5], max.depth = 5, eta = 0.1,nround = 800, objective = "binary:logistic")
pred.train.under <- predict(bst.under,data_balanced_under[,-5])
modelrocTrainunder <- roc(data_balanced_under[,5],pred.train.under)
plot(modelrocTrainunder, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,main = "ROC Curve - UNDER - Train")
acunder.train <- accuracy.meas(data_balanced_under[,5], pred.train.under,threshold = 0.476)
F2Score <- FScore(acunder.train,2)


pred.under <- predict(bst.under, testX)
modelrocTestunder <- roc(testY,pred.under)
plot(modelrocTestunder, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,main = "ROC Curve - UNDER - Test")
acunder.test <- accuracy.meas(testY, pred.under,threshold = 0.455)
F2Score <- FScore(acunder.test,2)





#-------simple decision tree
library(rpart)
treeimb <- rpart(label ~ ., data = train)
pred.treeimb <- predict(treeimb, newdata = test)
accuracy.meas(test$label, pred.treeimb,threshold = 0.052)
roc.curve(test$label, pred.treeimb, plotit = T)

# -------------------------------------训练决策树
data.rose <- as.data.frame(data.rose)
data_balanced_over <- as.data.frame(data_balanced_over)
data_balanced_under <- as.data.frame(data_balanced_under)
data_balanced_both <- as.data.frame(data_balanced_both)

tree.rose <- rpart(label ~ ., data = data.rose)
tree.over <- rpart(label ~ ., data = data_balanced_over)
tree.under <- rpart(label ~ ., data = data_balanced_under)
tree.both <- rpart(label ~ ., data = data_balanced_both)
# 在测试集上做预测
pred.tree.rose <- predict(tree.rose, newdata = test[,-4])
pred.tree.over <- predict(tree.over, newdata = test[,-4])
pred.tree.under <- predict(tree.under, newdata = test[,-4])
pred.tree.both <- predict(tree.both, newdata = test[,-4])


modelrocTreeRose <- roc(testY,pred.tree.rose)
plot(modelrocTreeRose, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
accuracy.meas(testY,pred.tree.rose,threshold = 0.520)

modelrocTreeOver <- roc(testY,pred.tree.over)
plot(modelrocTreeOver, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
accuracy.meas(testY,pred.tree.over,threshold = 0.500)

modelrocTreeUnder <- roc(testY,pred.tree.under)
plot(modelrocTreeUnder, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
accuracy.meas(testY,pred.tree.under,threshold = 0.520)

modelrocTreeBoth <- roc(testY,pred.tree.both)
plot(modelrocTreeBoth, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
accuracy.meas(testY,pred.tree.both,threshold = 0.520)



roc.curve(test$label, pred.tree.rose)
roc.curve(test$label, pred.tree.over)
roc.curve(test$label, pred.tree.under)
roc.curve(test$label, pred.tree.both)

#------------------------------------------svm & randomforest
#--------------svm
x <- subset(data.rose, select = -label)
y <- data.rose$tabel
model <- svm(x, y)
print(model)
summary(model)

pred.svm <- as.numeric(predict(model, testX))
modelrocSVMrose <- roc(testY,pred.svm)
plot(modelrocSVMrose, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
accuracy.meas(testY, pred.svm,threshold = 0.5)



#--------------randomForest
model.forest <- randomForest(label~.,data = data.rose)
pre.forest <- predict(model.forest,testX)
modelrocRFrose <- roc(testY,pred.forest)
plot(modelrocRFrose, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
accuracy.meas(data.rose[,4], pred.train.rose,threshold = 0.5)


