#----------include packages
library(data.table);library(stringr)
library(ROSE);library(randomForest)
library(pROC);library(caret)
library(Hmisc)
library(car)
library(varSelRF)
library(DMwR)
#----------set the path
setwd("E:/文档/毕业论文/数据")
#setwd("C:/Users/CQG/Desktop/毕业论文/数据")
#setwd("E:/R/DH/FE/FinancialStatement")
source("NecessaryFunction.R")
#----------import the data
Data <- fread("p2pData.csv",header=TRUE)
NewData <- Data[,-c("Key","AIR","Borrower","Credit_Rank","Credit_Score","Times_Request","Times_Loan","Times_Repayment","Credit_Lines","Amount_Loan","Unpaid_Loan","Overdue_Payment","Cum_Overdue_Payment","Times_Overdue","Severe_Overdue","Loan_Statement")]

NewData[NewData$Label=="CLOSED","Label"] <- 0
NewData[NewData$Label=="BAD_DEBT","Label"] <- 1
NewData <- as.data.frame(NewData)
tmpData <- rbind(colnames(NewData),NewData)
tmpVec <- c("Repayment_Method","Property_Loan","Marriage","Census_Register"
            ,"House_Property","House_Loan","Car","Car_Loan","Type_Company","Industry","Workplace","Type_Job")
NewData[,tmpVec] <- as.data.frame(apply(tmpData[,tmpVec],2,Chinese2Number))
#NewData[,"Log_Target"] <- log(NewData[,"Target"])
#NewData$Target <- NULL


NewData <- apply(NewData,2,as.numeric)
NewData <- as.data.frame(NewData)
NewData$Label <- factor(NewData$Label)

LoanStatement <- Data[,c("Key","Loan_Statement")]
#--0:12613  1:1556
#--------------------Outlier Test

outlierKD(NewData,Target)
#outlierKD(NewData,AIR)
outlierKD(NewData,Age)

#---------------Find NA
DetectNA <- apply(apply(NewData,2,is.na),2,any)
NAFeatureNames <- names(which(DetectNA==TRUE))
#"Target"          "AIR"             "Property_Loan"   "Age"             "Education"       "Marriage" "Census_Register" 
#"Income"         "Type_Company"    "Industry"        "Scale_Company"   "Type_Job"        "Workplace"       "Time_Job"        

#-------------split the data then impute
PaidIndex <- which(NewData$Label == 0)
PaidData <- NewData[PaidIndex,]
DefaultIndex <- which(NewData$Label == 1)
DefaultData <- NewData[DefaultIndex,]
#--------------impute value for Education Income Scale_Company Time_Job
ordinalVariable <- c("Education","Income","Scale_Company","Time_Job")
#---batch impute
PaidData[,ordinalVariable] <- apply(PaidData[,ordinalVariable],2,ImputeMedianBatch)
DefaultData[,ordinalVariable] <- apply(DefaultData[,ordinalVariable],2,ImputeMedianBatch)

#---------------omit the NA
PaidData <- na.omit(PaidData)
DefaultData <- na.omit(DefaultData)
BindData <- rbind(PaidData,DefaultData)  #---0:11319 1:1492
#---------------normalize the "Target" colum
BindData$Target <- Normalize(BindData$Target)
NormData <- BindData
# NAIndexList <- apply(NewData[,NAFeatureNames],2,FindNAIndex)
# NAIndexArray <- unlist(NAIndexList)
# names(NAIndexArray) <- NULL
# Union_NAIndex <- unique(NAIndexArray)
# length(Union_NAIndex)
# CompleteData <- NewData[-Union_NAIndex,]
#---------------Distribution of Data
table(NormData$Label)
prop.table(table(NormData$Label))



#-------------split into train validation and test
set.seed(50)
splitIndexOne <- createDataPartition(NormData$Label,times = 1,p = 0.7,list = FALSE)
Train <- NormData[splitIndexOne,]
NormWithoutTrain <- NormData[-splitIndexOne,]
set.seed(50)
splitIndexTwo <- createDataPartition(NormWithoutTrain$Label,times = 1,p = 0.5,list = FALSE)
Validation <- NormWithoutTrain[splitIndexTwo,]
Test <- NormWithoutTrain[-splitIndexTwo,]




#---------------basic model
par(mfrow=c(1,1))
set.seed(50)
forest <- randomForest(Label~.,data = Train,importance = TRUE,ntree=100)
# pred.newtrain.forest <- predict(forest,Train[,-1])
# pred.newtrain.forest <- as.numeric(as.character(pred.newtrain.forest))
# rocnewTrainforest <- roc(as.numeric(as.character(Train[,1])),pred.newtrain.forest)
# plot(rocnewTrainforest,print.auc=TRUE, auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),
#      max.auc.polygon=TRUE,auc.polygon.col="Skyblue",print.thres=TRUE,main="ROC Curve-Random Forest - Train")
# acforest.newtrain <- accuracy.meas(Train[,1],pred.newtrain.forest,threshold = 0.5)
# 
# errornewTrainforest <- (sum(pred.newtrain.forest!=Train[,1])/length(Train[,1]))
# F1Score <- FScore(acforest.newtrain)
# foresttrainAssess <- c(acforest.newtrain$precision,acforest.newtrain$recall,F1Score,errornewTrainforest,rocnewTrainforest$auc)
# confusionMatrix(pred.newtrain.forest,Train[,1],positive = "1")
# plot(forest)

ImportanceDF <- as.data.frame(importance(forest))
varImpPlot(forest)


#-----------------Feature Selection
#--------OOB
Imp_OOB_sort <- ImportanceDF[order(-ImportanceDF$MeanDecreaseAccuracy),]
Feat_OOB <- rownames(Imp_OOB_sort)[1:15]
Feat_OOB
#"Scale_Company"   "Age"             "Car"             "Education"       "Term"            "Property_Loan"   "Target"         "Type_Company"    
#"Workplace"       "Census_Register" "Time_Job"        "Industry"        "Income"          "House_Loan"     "Type_Job"     
#--------Gini
Imp_Gini_sort <- ImportanceDF[order(-ImportanceDF$MeanDecreaseGini),]
Feat_Gini <- rownames(Imp_Gini_sort)[1:15]
Feat_Gini
#"Scale_Company"   "Age"             "Car"             "Education"       "Term"            "Property_Loan"   "Target"         
#"Type_Company"    "Workplace"       "Census_Register" "Time_Job"        "Industry"        "Income"          "House_Loan"     
#"Type_Job"   
Feat_Select <- intersect(Feat_OOB,Feat_Gini)


#------------------------------Strategy 1:Bases on Balanced Data Sey
#-----------------Parameter Selection
Feat_Select <- c("Label",Feat_Select)
Train <- Train[,Feat_Select]
Validation <- Validation[,Feat_Select]
Test <- Test[,Feat_Select]

#----------------select ntree
#--------train
par(mfrow=c(1,1))
set.seed(50)
forest <- randomForest(Label~.,data = Train,importance = TRUE,ntree=300)

plot(forest,main = "Error vs Trees")
legend(70, 0.07, c('Total Error', '1-Error', '0-Error'),lty=c(1,2,2),lwd=c(1,1,1),col=c('black','green', 'red' )) 
#ImportanceDF <- as.data.frame(importance(forest))
#varImpPlot(forest)
#--------steady when n>=60 pick ntree=50
best_ntree <- 50

#----------------select mtry
n <- length(Feat_Select)
OOB_Record <- c()
for(i in 1:(n-1)){
  set.seed(50)
  tmpForest <- randomForest(Label~.,data = Train,importance = TRUE,mtry = i,ntree=best_ntree)
  tmpOOBMat <- tmpForest$err.rate
  tmpOOB <- tmpOOBMat[nrow(tmpOOBMat),1]
  OOB_Record <- c(OOB_Record,tmpOOB)
}
plot(1:(n-1),OOB_Record,type = "o")
#---------lowest when mtry=4 ,pick mtry = 3
best_mtry <- 3

#----------------select max_nodes & maxnodes
#----grid search
maxNodeVec <- seq(10,300,50)
Weight_NVec <- seq(0.1,0.9,0.1)
num_row <- length(maxNodeVec) * length(Weight_NVec)
i <- 1

TrainRecord <- data.frame(matrix(NA,num_row,8))
colnames(TrainRecord) <- c("MaxNodes","W_negative","cutoff","Accuracy","F1_Score","AUC","TPR","TNR")

ValidationRecord <- data.frame(matrix(NA,num_row,8))
colnames(ValidationRecord) <- c("MaxNodes","W_negative","cutoff","Accuracy","F1_Score","AUC","TPR","TNR")

for(tmpWeight_N in Weight_NVec){
  for(tmpMaxNode in maxNodeVec){
    set.seed(50)
    forest <- randomForest(Label~.,data = Train,importance = TRUE,mtry = best_mtry,ntree=best_ntree,maxnodes = tmpMaxNode,classwt = c(tmpWeight_N,1-tmpWeight_N))
    #------输出概率值
    pred.train.forest <- predict(forest,Train[,-1],type = "prob")[,2]
    #pred.train.forest <- as.numeric(as.character(pred.train.forest))
    rocTrainforest <- roc(as.numeric(as.character(Train[,1])),pred.train.forest)
    #-------画出roc曲线
    plot(rocTrainforest,print.auc=TRUE, auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),
         max.auc.polygon=TRUE,auc.polygon.col="Skyblue",print.thres=TRUE,main="ROC Curve-Random Forest - Train")
    #-------找出最佳cutoff
    best_threshold <- rocTrainforest$thresholds[which.max(rocTrainforest$sensitivities+rocTrainforest$specificities)][1]
    #---------计算F1-得分
    acforest.train <- accuracy.meas(as.numeric(as.character(Train[,1])),pred.train.forest,threshold = best_threshold)
    F1Score_Train <- FScore(acforest.train)
    #--------根据最佳cutoff输出0，1预测结果
    pred.train.forest <- predict(forest,Train[,-1],type = "response",cutoff = c(1-best_threshold,best_threshold))
    #---------计算准确率
    Accuracy_Train <- 1-(sum(pred.train.forest!=Train[,1])/length(Train[,1]))
    
    #--------confusion matrix
    conf_Train <- confusionMatrix(pred.train.forest,Train[,1],positive = "1")
    TPR_Train <- conf_Train$byClass["Sensitivity"];names(TPR_Train) <- NULL
    TNR_Train <- conf_Train$byClass["Specificity"];names(TNR_Train) <- NULL
    TrainRecord[i,] <- c(tmpMaxNode,tmpWeight_N,best_threshold,Accuracy_Train,F1Score_Train,rocTrainforest$auc,TPR_Train,TNR_Train)
    
    
    #------输出概率值
    pred.validation.forest <- predict(forest,Validation[,-1],type = "prob")[,2]
    #pred.validation.forest <- as.numeric(as.character(pred.validation.forest))
    rocvalidationforest <- roc(as.numeric(as.character(Validation[,1])),pred.validation.forest)
    #-------画出roc曲线
    plot(rocvalidationforest,print.auc=TRUE, auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),
         max.auc.polygon=TRUE,auc.polygon.col="Skyblue",print.thres=TRUE,main="ROC Curve-Random Forest - validation")
    #-------找出最佳cutoff
    best_threshold <- rocvalidationforest$thresholds[which.max(rocvalidationforest$sensitivities+rocvalidationforest$specificities)][1]
    #---------计算F1-得分
    acforest.validation <- accuracy.meas(as.numeric(as.character(Validation[,1])),pred.validation.forest,threshold = best_threshold)
    F1Score_validation <- FScore(acforest.validation)
    #--------根据最佳cutoff输出0，1预测结果
    pred.validation.forest <- predict(forest,Validation[,-1],type = "response",cutoff = c(1-best_threshold,best_threshold))
    #---------计算准确率
    Accuracy_validation <- 1-(sum(pred.validation.forest!=Validation[,1])/length(Validation[,1]))
    #--------confusion matrix
    conf_Validation <- confusionMatrix(pred.validation.forest,Validation[,1],positive = "1")
    TPR_Validation <- conf_Validation$byClass["Sensitivity"];names(TPR_Validation) <- NULL
    TNR_Validation <- conf_Validation$byClass["Specificity"];names(TNR_Validation) <- NULL
    
    ValidationRecord[i,] <- c(tmpMaxNode,tmpWeight_N,best_threshold,Accuracy_validation,F1Score_validation,rocvalidationforest$auc,TPR_Validation,TNR_Validation)
    
    i <- i+1
  }

}


#---------USE ggplot to modify the following plots
plot(TrainRecord[,"MaxNodes"],TrainRecord[,"Accuracy"],type = "l")
plot(TrainRecord[,"MaxNodes"],TrainRecord[,"F1_Score"],type = "l")
plot(TrainRecord[,"MaxNodes"],TrainRecord[,"AUC"],type = "l")

plot(ValidationRecord[,"MaxNodes"],ValidationRecord[,"Accuracy"],type = "l")
plot(ValidationRecord[,"MaxNodes"],ValidationRecord[,"F1_Score"],type = "l")
plot(ValidationRecord[,"MaxNodes"],ValidationRecord[,"AUC"],type = "l")
#set.seed(50)
best_maxnodes <- 70 
best_cutoff <- ValidationRecord[as.character(best_maxnodes),]$cutoff
#---------based on validation,best when maXnodes=70, pick maxnodes=70 

#------------------------------------Another Strategy:USE Classwt
#----------------select classwt



for(tmpWeight_N in Weight_NVec){
  set.seed(50)
  forest <- randomForest(Label~.,data = Train,importance = TRUE,mtry = 3,ntree=60,maxnodes = 200,classwt = c(1,tmpWeight_N))
  #------输出概率值
  pred.newtrain.forest <- predict(forest,Train[,-1],type = "prob")[,2]
  #pred.newtrain.forest <- as.numeric(as.character(pred.newtrain.forest))
  rocnewTrainforest <- roc(as.numeric(as.character(Train[,1])),pred.newtrain.forest)
  #-------画出roc曲线
  plot(rocnewTrainforest,print.auc=TRUE, auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),
       max.auc.polygon=TRUE,auc.polygon.col="Skyblue",print.thres=TRUE,main="ROC Curve-Random Forest - Train")
  #-------找出最佳cutoff
  best_threshold <- rocnewTrainforest$thresholds[which.max(rocnewTrainforest$sensitivities+rocnewTrainforest$specificities)][1]
  #--------根据最佳cutoff输出0，1预测结果
  pred.newtrain.forest <- predict(forest,Train[,-1],type = "response",cutoff = c(1-best_threshold,best_threshold))
  #---------计算准确率
  errornewTrainforest <- (sum(pred.newtrain.forest!=Train[,1])/length(Train[,1]))
  acforest.newtrain <- accuracy.meas(Train[,1],pred.newtrain.forest,threshold = best_threshold)
  F1Score <- FScore(acforest.newtrain)
  
  
  confusionMatrix(pred.newtrain.forest,Train[,1],positive = "1")
  TrainRecord[as.character(tmpWeight_N),2:5] <- c(best_threshold,errornewTrainforest,F1Score,rocnewTrainforest$auc)
  
  
  pred.validation.forest <- predict(forest,newdata = Validation[,-1],type = "prob")[,2]
  pred.validation.forest <- as.numeric(as.character(pred.validation.forest))
  table(as.numeric(pred.validation.forest),Validation[,1])
  rocValidationforest <- roc(as.numeric(as.character(Validation[,1])),pred.validation.forest)
  plot(rocValidationforest,print.auc=TRUE, auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),
       max.auc.polygon=TRUE,auc.polygon.col="Skyblue",print.thres=TRUE,main="ROC Curve-Random Forest - Validation")
  errorValidationforest <- (sum(pred.validation.forest!=Validation[,1])/length(Validation[,1]))
  acforest.validation <- accuracy.meas(Validation[,1],pred.validation.forest,threshold = 0.5)
  F1Score <- FScore(acforest.validation)
  ValidationRecord[as.character(tmpWeight_N),2:4] <- c(errorValidationforest,F1Score,rocValidationforest$auc)
  
  
}
#---------USE ggplot to modify the following plots
plot(TrainRecord[,"Weight_N"],TrainRecord[,"Error"],type = "l")
plot(TrainRecord[,"Weight_N"],TrainRecord[,"F1_Score"],type = "l")
plot(TrainRecord[,"Weight_N"],TrainRecord[,"AUC"],type = "l")

plot(ValidationRecord[,"Weight_N"],ValidationRecord[,"Error"],type = "l")
plot(ValidationRecord[,"Weight_N"],ValidationRecord[,"F1_Score"],type = "l")
plot(ValidationRecord[,"Weight_N"],ValidationRecord[,"AUC"],type = "l")


#-------------check the performance on test set
set.seed(50)
forest <- randomForest(Label~.,data = BalancedTrain,importance = TRUE,mtry = best_mtry,ntree=best_ntree,maxnodes = best_maxnodes)
#------输出概率值
pred.test.forest <- predict(forest,Test[,-1],type = "prob")[,2]
#pred.test.forest <- as.numeric(as.character(pred.test.forest))
roctestforest <- roc(as.numeric(as.character(Test[,1])),pred.test.forest)
#-------画出roc曲线
plot(roctestforest,print.auc=TRUE, auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),
     max.auc.polygon=TRUE,auc.polygon.col="Skyblue",print.thres=TRUE,main="ROC Curve-Random Forest - test")
#---------计算F1-得分
acforest.test <- accuracy.meas(as.numeric(as.character(Test[,1])),pred.test.forest,threshold = best_cutoff)
F1Score_test <- FScore(acforest.test)
#--------根据最佳cutoff输出0，1预测结果
pred.test.forest <- predict(forest,Test[,-1],type = "response",cutoff = c(1-best_cutoff,best_cutoff))
#---------计算准确率
Accuracy_test <- 1-(sum(pred.test.forest!=Test[,1])/length(Test[,1]))

TestPerformance <- c(Accuracy_test,F1Score_test,roctestforest$auc)
names(TestPerformance) <- c("Accuracy","F1_Score","AUC")




