#----------include packages
library(data.table);library(stringr)
library(ROSE);library(randomForest)
library(pROC);library(caret)
library(Hmisc);library(car)
library(varSelRF);library(DMwR)
library(ggplot2);library(reshape2)
#----------set the path
setwd("E:/文档/毕业论文/数据")
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


NewData <- apply(NewData,2,as.numeric)
NewData <- as.data.frame(NewData)
NewData$Label <- factor(NewData$Label)

LoanStatement <- Data[,c("Key","Loan_Statement")]

#--------------------Outlier Test
outlierKD(NewData,Target)
outlierKD(NewData,Age)

#---------------Find NA
DetectNA <- apply(apply(NewData,2,is.na),2,any)
NAFeatureNames <- names(which(DetectNA==TRUE))
#"Target"          "Property_Loan"   "Age"             "Education"       "Marriage" "Census_Register" 
#"Income"         "Type_Company"    "Industry"        "Scale_Company"   "Type_Job"        "Workplace"       "Time_Job"        

#-------------split the data then impute with median
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
BindData <- rbind(PaidData,DefaultData)  
#---------------normalize the "Target" colum
BindData$Target <- Normalize(BindData$Target)
NormData <- BindData

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


#-------------SMOTE:deal with imbalance
set.seed(50)
BalancedTrain <- SMOTE(Label~.,Train,perc.over=500,k=5,perc.under=120)
table(BalancedTrain$Label)

#------------------------basic model
par(mfrow=c(1,1))
set.seed(50)
forest <- randomForest(Label~.,data = BalancedTrain,importance = TRUE,ntree=200)


ImportanceDF <- as.data.frame(importance(forest))
varImpPlot2(forest)
#----plot the forest error rate
ggForest(forest)


#-----------------Feature Selection
#--------OOB
Imp_OOB_sort <- ImportanceDF[order(-ImportanceDF$MeanDecreaseAccuracy),]
Feat_OOB <- rownames(Imp_OOB_sort)[1:15]
Feat_OOB
#"Scale_Company"   "Workplace"       "Car"             "Education"       "Property_Loan"  
#"Target"          "Age"             "Term"            "Census_Register" "Industry"       
#"Type_Company"    "Time_Job"        "Type_Job"        "Income"          "House_Loan" 
#--------Gini
Imp_Gini_sort <- ImportanceDF[order(-ImportanceDF$MeanDecreaseGini),]
Feat_Gini <- rownames(Imp_Gini_sort)[1:15]
Feat_Gini
# "Scale_Company"   "Term"            "Target"         
# "Time_Job"        "Property_Loan"   "Type_Company"   
# "Education"       "Income"          "House_Loan"     
# "Marriage"        "Industry"        "Workplace"      
# "House_Property"  "Age"             "Census_Register"

#-------------rfcv
set.seed(50)
folds <- createFolds(BalancedTrain$Label,k=5,list = TRUE)
Fold1_Data <- BalancedTrain[folds$Fold1,]
Fold2_Data <- BalancedTrain[folds$Fold2,]
Fold3_Data <- BalancedTrain[folds$Fold3,]
Fold4_Data <- BalancedTrain[folds$Fold4,]
Fold5_Data <- BalancedTrain[folds$Fold5,]
#-------bind the data
tmp_Data1 <- rbind(Fold2_Data,Fold3_Data,Fold4_Data,Fold5_Data)
tmp_Data2 <- rbind(Fold1_Data,Fold3_Data,Fold4_Data,Fold5_Data)
tmp_Data3 <- rbind(Fold1_Data,Fold2_Data,Fold4_Data,Fold5_Data)
tmp_Data4 <- rbind(Fold1_Data,Fold2_Data,Fold3_Data,Fold5_Data)
tmp_Data5 <- rbind(Fold1_Data,Fold2_Data,Fold3_Data,Fold4_Data)
#------select Feat base on cv
#-----cv 1
set.seed(50)
fit1 <- varSelRF(tmp_Data1[,-1],tmp_Data1[,1],ntree = 100,ntreeIterat = 500,vars.drop.frac = 0.1)
Feat_CV1 <- fit1$selected.vars
#---13 vars
# "Car"            "Education"      "House_Loan"    
# "House_Property" "Income"         "Industry"      
# "Marriage"       "Property_Loan"  "Scale_Company" 
# "Target"         "Term"           "Time_Job"      # "Type_Company"  
#-----cv 2
set.seed(50)
fit2 <- varSelRF(tmp_Data2[,-1],tmp_Data2[,1],ntree = 100,ntreeIterat = 500,vars.drop.frac = 0.1)
Feat_CV2 <- fit2$selected.vars
#---12 vars
# "Car"            "Education"      "House_Loan"    
# "House_Property" "Income"         "Marriage"      
# "Property_Loan"  "Scale_Company"  "Target"        
# "Term"           "Time_Job"       "Type_Company"  
#-----cv 3
set.seed(50)
fit3 <- varSelRF(tmp_Data3[,-1],tmp_Data3[,1],ntree = 100,ntreeIterat = 500,vars.drop.frac = 0.1)
Feat_CV3 <- fit3$selected.vars
#---14 vars
# "Car"            "Education"      "House_Loan"    
# "House_Property" "Income"         "Industry"      
# "Marriage"       "Property_Loan"  "Scale_Company" 
# "Target"         "Term"           "Time_Job"      
# "Type_Company"   "Type_Job"      
#-----cv 4
set.seed(50)
fit4 <- varSelRF(tmp_Data4[,-1],tmp_Data4[,1],ntree = 100,ntreeIterat = 500,vars.drop.frac = 0.1)
Feat_CV4 <- fit4$selected.vars
#---12 vars
# "Car"            "Education"      "House_Loan"    
# "House_Property" "Income"         "Marriage"      
# "Property_Loan"  "Scale_Company"  "Target"        
# "Term"           "Time_Job"       "Type_Company"  
#-----cv 5
set.seed(50)
fit5 <- varSelRF(tmp_Data5[,-1],tmp_Data5[,1],ntree = 100,ntreeIterat = 500,vars.drop.frac = 0.1)
Feat_CV5 <- fit5$selected.vars
# #---13 vars
# "Car"            "Education"      "House_Loan"    
# "House_Property" "Income"         "Marriage"      
# "Property_Loan"  "Scale_Company"  "Target"        
# "Term"           "Time_Job"       "Type_Company"   "Workplace"

#-------------Feature  Selection
Feat_All <- c(Feat_OOB,Feat_Gini,Feat_CV1,Feat_CV2,Feat_CV3,Feat_CV4,Feat_CV5)
Feat_Table <- table(Feat_All)
Select_Index <- which(Feat_Table>=5)
Feat_Select <- names(Select_Index)
# "Car"            "Education"      "House_Loan"     "House_Property" "Income"         "Marriage"      
# "Property_Loan"  "Scale_Company"  "Target"         "Term"           "Time_Job"       "Type_Company"  

#------------------------------Strategy 1:Without LDA
#-----------------Parameter Selection
Feat_Select <- c("Label",Feat_Select)
Train <- Train[,Feat_Select]
BalancedTrain <- BalancedTrain[,Feat_Select]
Validation <- Validation[,Feat_Select]
Test <- Test[,Feat_Select]

#----------------select ntree
#--------train
par(mfrow=c(1,1))
set.seed(50)
forest <- randomForest(Label~.,data = BalancedTrain,importance = TRUE,ntree=100)
ggForest(forest)
#--------steady when n>=60 pick ntree=60
best_ntree <- 80

#----------------select mtry
n <- length(Feat_Select)
OOB_Record <- c()
for(i in 1:(n-1)){
  set.seed(50)
  tmpForest <- randomForest(Label~.,data = BalancedTrain,importance = TRUE,mtry = i,ntree=best_ntree)
  tmpOOBMat <- tmpForest$err.rate
  tmpOOB <- tmpOOBMat[nrow(tmpOOBMat),1]
  OOB_Record <- c(OOB_Record,tmpOOB)
}

mtryData <- data.frame(mtry=1:(n-1),OOB_Error=OOB_Record)
#write.csv(mtryData,"mtry_Selection.csv")
ggplot(data = mtryData,aes(x=mtry,y=OOB_Error))+geom_line(size=1.5,linetype="dashed")+geom_point(size=4,shape=2,position = position_dodge(1))+scale_x_continuous(limits=c(min(mtryData$mtry),max(mtryData$mtry)),breaks=round(seq(min(mtryData$mtry),max(mtryData$mtry),length.out=13),0))+
  theme_bw()+theme(panel.grid = element_blank())+
  theme(legend.key.size=unit(3,'cm'))+ theme(legend.position="bottom")+ theme(legend.title = element_blank())+
  theme(legend.text = element_text(colour = 'black', angle = 0, size = 15, hjust = 1, vjust = 1, face = 'bold'))+
  theme(axis.title.x= element_text(size=15,  color="black", face= "bold", vjust=0.5, hjust=0.5))+
  theme(axis.title.y= element_text(size=15,  color="black", face= "bold", vjust=0.5, hjust=0.5))+
  theme(axis.text.x= element_text(size=13, color="black", face= "bold", vjust=0.5, hjust=0.5))+
  theme(axis.text.y= element_text(size=13, color="black", face= "bold", vjust=0.5, hjust=0.5))


#---------lowest when mtry=5 ,pick mtry = 5
best_mtry <- 5

#----------------select max_nodes
maxNodeVec <- seq(10,500,10)
TrainRecord <- data.frame(matrix(NA,length(maxNodeVec),9))
colnames(TrainRecord) <- c("MaxNodes","cutoff","Accuracy","Precision","Recall","F1_Score","Sensitivity","Specificity","AUC")
TrainRecord$MaxNodes <- maxNodeVec
rownames(TrainRecord) <- maxNodeVec

ValidationRecord <- data.frame(matrix(NA,length(maxNodeVec),9))
colnames(ValidationRecord) <- c("MaxNodes","cutoff","Accuracy","Precision","Recall","F1_Score","Sensitivity","Specificity","AUC")
ValidationRecord$MaxNodes <- maxNodeVec
rownames(ValidationRecord) <- maxNodeVec

for(tmpMaxNode in maxNodeVec){
  set.seed(50)
  forest <- randomForest(Label~.,data = BalancedTrain,importance = TRUE,mtry = best_mtry,ntree=best_ntree,maxnodes = tmpMaxNode)
  #------输出概率值
  pred.train.forest <- predict(forest,BalancedTrain[,-1],type = "prob")[,2]
  #pred.train.forest <- as.numeric(as.character(pred.train.forest))
  rocTrainforest <- roc(as.numeric(as.character(BalancedTrain[,1])),pred.train.forest)
  #-------画出roc曲线
  plot(rocTrainforest,print.auc=TRUE, auc.polygon=TRUE,
       max.auc.polygon=TRUE,max.auc.polygon.col="white",auc.polygon.col="gainsboro",print.thres=TRUE,main="ROC Curve-Random Forest - train",
       cex.axis=1,cex.lab=1.3,font=2)
  #-------找出最佳cutoff
  maxIndex_Train <- which.max(rocTrainforest$sensitivities+rocTrainforest$specificities)
  best_threshold <- rocTrainforest$thresholds[maxIndex_Train][1]
  sensitivity_Train <- rocTrainforest$sensitivities[maxIndex_Train]
  specificity_Train <- rocTrainforest$specificities[maxIndex_Train]
  #---------计算F1-得分
  acforest.train <- accuracy.meas(as.numeric(as.character(BalancedTrain[,1])),pred.train.forest,threshold = best_threshold)
  F1Score_Train <- FScore(acforest.train)
  #--------根据最佳cutoff输出0，1预测结果
  pred.train.forest <- predict(forest,BalancedTrain[,-1],type = "response",cutoff = c(1-best_threshold,best_threshold))
  #---------计算准确率
  Accuracy_Train <- 1-(sum(pred.train.forest!=BalancedTrain[,1])/length(BalancedTrain[,1]))
  
  
  TrainRecord[as.character(tmpMaxNode),2:9] <- c(best_threshold,Accuracy_Train,acforest.train$precision,acforest.train$recall,F1Score_Train,sensitivity_Train,specificity_Train,rocTrainforest$auc)
  
  #------输出概率值
  pred.validation.forest <- predict(forest,Validation[,-1],type = "prob")[,2]
  #pred.validation.forest <- as.numeric(as.character(pred.validation.forest))
  rocvalidationforest <- roc(as.numeric(as.character(Validation[,1])),pred.validation.forest)
  #-------画出roc曲线
  plot(rocvalidationforest,print.auc=TRUE, auc.polygon=TRUE,
       max.auc.polygon=TRUE,max.auc.polygon.col="white",auc.polygon.col="gainsboro",print.thres=TRUE,main="ROC Curve-Random Forest - validation",
       cex.axis=1,cex.lab=1.3,font=2)
  #-------找出最佳cutoff
  maxIndex_Validation <- which.max(rocvalidationforest$sensitivities+rocvalidationforest$specificities)
  best_threshold <- rocvalidationforest$thresholds[maxIndex_Validation][1]
  sensitivity_Validation <- rocvalidationforest$sensitivities[maxIndex_Validation]
  specificity_Validation <- rocvalidationforest$specificities[maxIndex_Validation]
  #---------计算F1-得分
  acforest.validation <- accuracy.meas(as.numeric(as.character(Validation[,1])),pred.validation.forest,threshold = best_threshold)
  F1Score_validation <- FScore(acforest.validation)
  #--------根据最佳cutoff输出0，1预测结果
  pred.validation.forest <- predict(forest,Validation[,-1],type = "response",cutoff = c(1-best_threshold,best_threshold))
  #---------计算准确率
  Accuracy_validation <- 1-(sum(pred.validation.forest!=Validation[,1])/length(Validation[,1]))
  
  
  ValidationRecord[as.character(tmpMaxNode),2:9] <- c(best_threshold,Accuracy_validation,acforest.validation$precision,acforest.validation$recall,F1Score_validation,sensitivity_Validation,specificity_Validation,rocvalidationforest$auc)
  
  
}
#---------USE ggplot to modify the following plots
ggCompare(maxNodeVec,TrainRecord$Accuracy,ValidationRecord$Accuracy,x_axis="MaxNodes",type="Accuracy",ylim=seq(0.8,1,0.05))
ggCompare(maxNodeVec,TrainRecord$F1_Score,ValidationRecord$F1_Score,x_axis="MaxNodes",type="F1_Score",ylim=seq(0.65,1,0.05))
ggCompare(maxNodeVec,TrainRecord$AUC,ValidationRecord$AUC,x_axis="MaxNodes",type="AUC",ylim=seq(0.9,1,0.02))

#set.seed(50)
maxnodes_index <- which.max(ValidationRecord$F1_Score)
best_maxnodes <- ValidationRecord[maxnodes_index,]$MaxNodes
best_cutoff <- ValidationRecord[maxnodes_index,]$cutoff
#---------based on validation,best when maXnodes=70, pick maxnodes=70 

#-------------check the performance on test set
set.seed(50)
forest <- randomForest(Label~.,data = BalancedTrain,importance = TRUE,mtry = best_mtry,ntree=best_ntree,maxnodes = best_maxnodes)
#--------根据最佳cutoff输出0，1预测结果
pred.test.forest <- predict(forest,Test[,-1],type = "response",cutoff = c(1-best_cutoff,best_cutoff))

roctestforest <- roc(as.numeric(as.character(Test[,1])),as.numeric(as.character(pred.test.forest)))
#-------画出roc曲线
plot(roctestforest,print.auc=TRUE, auc.polygon=TRUE,
     max.auc.polygon=TRUE,max.auc.polygon.col="white",auc.polygon.col="gainsboro",print.thres=TRUE,main="ROC Curve-Random Forest - test",
     cex.axis=1,cex.lab=1.3,font=2)

#---------计算F1-得分
pred.test.forest <- predict(forest,Test[,-1],type = "prob",cutoff = c(1-best_cutoff,best_cutoff))[,2]
acforest.test <- accuracy.meas(as.numeric(as.character(Test[,1])),pred.test.forest,threshold = best_cutoff)
F1Score_test <- FScore(acforest.test)

#---------计算准确率
pred.test.forest <- predict(forest,Test[,-1],type = "response",cutoff = c(1-best_cutoff,best_cutoff))
Accuracy_test <- 1-(sum(pred.test.forest!=Test[,1])/length(Test[,1]))

TestPerformance <- c(Accuracy_test,acforest.test$precision,acforest.test$recall,F1Score_test,roctestforest$auc)
names(TestPerformance) <- c("Accuracy","Precision","Recall","F1_Score","AUC")

















