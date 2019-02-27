

#----------include packages
library(data.table);library(stringr)
library(ROSE);library(randomForest)
library(pROC);library(caret)
library(Hmisc);library(car)
library(varSelRF);library(DMwR)
library(ggplot2);library(wordcloud2)
library(RColorBrewer);library(topicmodels)
#----------set the path
setwd("E:/文档/毕业论文/数据")
source("NecessaryFunction.R")
source("postProcess.R")
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

Loan_Statement <- Data$Loan_Statement
NewData$Loan_Statement <- Loan_Statement
NewData$Stament_Length <- nchar(NewData$Loan_Statement)

#--------------------Outlier Test

outlierKD(NewData,Target)
outlierKD(NewData,Age)

#---------------Find NA
DetectNA <- apply(apply(NewData,2,is.na),2,any)
NAFeatureNames <- names(which(DetectNA==TRUE))
#"Target"          "Property_Loan"   "Age"             "Education"       "Marriage" "Census_Register" 
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


#--------------split statement
Train_Statement <- Train$Loan_Statement
Validation_Statement <- Validation$Loan_Statement
Test_Statement <- Test$Loan_Statement


library(rJava)
library(Rwordseg)
installDict("E:\\文档\\毕业论文\\数据\\中华人民共和国行政区划.scel","ProvinceCity","scel")

#-------input stopwords
stopwords <- readLines(con <- file("E:/文档/毕业论文/数据/stopwords.dat", encoding = "UTF-8"))
close(con)
place_stopwords <- read.table("E:/文档/毕业论文/数据/行政区划.txt")  #--remove place stopwords
place_stopwords <- as.character(place_stopwords$V1)
province_stopwords <- place_stopwords[grep(pattern = "省",place_stopwords)]
province_short <- unlist(lapply(province_stopwords,removeProCity))
province_stopwords <- c(province_stopwords,province_short)

city_stopwords <- place_stopwords[grep(pattern = "市",place_stopwords)]
city_short <- unlist(lapply(city_stopwords,removeProCity))
city_stopwords <- c(city_stopwords,city_short)

stopwords <- c(stopwords,province_stopwords,city_stopwords)
filterAgain <- c("符合","用于","友","众","信息","提供","资料","贷","月","每月","信业","元","现居","年","记录","证","做","安","中","万元","null","现","雇","需","最多")
#filterAgain <- c("贷款","认证","审核","符合","借款","还款","借款人","标准","用于","友","众","信息","提供","考察","资料","实地","真实有效","贷","月","每月","信用","信业","元","现居","公司","职员","工作","年","记录","证","做","安","中","万元","钱","null","现","谢谢","想","希望","雇","需","行业","全国","工资","最多")

stopwords <- c(stopwords,filterAgain)

#-------remove numbers
Train_Statement <- lapply(Train_Statement,removeNumbers)
Validation_Statement <- lapply(Validation_Statement,removeNumbers)
Test_Statement <- lapply(Test_Statement,removeNumbers)

#loan_statement <- lapply(loan_statement$V1,removeNumbers)

#------word segment
Train_Statement <- lapply(Train_Statement,wordsegment)
Validation_Statement <- lapply(Validation_Statement,wordsegment)
Test_Statement <- lapply(Test_Statement,wordsegment)
#loan_statement <- lapply(loan_statement, wordsegment)

#-----remove stopwords
Train_Statement <- lapply(Train_Statement,removeStopWords,stopwords)
Validation_Statement <- lapply(Validation_Statement,removeStopWords,stopwords)
Test_Statement <- lapply(Test_Statement,removeStopWords,stopwords)

#loan_statement <- lapply(loan_statement, removeStopWords,stopwords)

#-----corpus
library(tm)
# library(tmcn)
# library(wordcloud)
#构建语料库
Train_corpus <-  Corpus(VectorSource(Train_Statement))
Validation_corpus <- Corpus(VectorSource(Validation_Statement))
Test_corpus <- Corpus(VectorSource(Test_Statement))

#建立文档-词条矩阵
Train.dtm <- DocumentTermMatrix(Train_corpus, control = list(wordLengths = c(2, Inf)))
Validation.dtm <- DocumentTermMatrix(Validation_corpus, control = list(wordLengths = c(2, Inf)))
Test.dtm <- DocumentTermMatrix(Test_corpus, control = list(wordLengths = c(2, Inf)))



#-------perplexity
#### 将主题设定为2～25
num_topics <- 100
ldas_Gibbs<-lda.rep(train=Train.dtm,validation=Validation.dtm,n_topic=num_topics,method='Gibbs')

#ldas_vem <- ldas
#save(ldas_Gibbs,file='ldas_Gibbs.rda')
load("E:/文档/毕业论文/数据/ldas_Gibbs.rda")
####困惑度的计算
index <- seq(10,num_topics,10)
train_perp <- ldas_Gibbs$train_perp
validation_perp <- ldas_Gibbs$validation_perp
loglik <- ldas_Gibbs$loglik
train_perp <- train_perp[index]
validation_perp <- validation_perp[index]
loglik <- loglik[index]

Train_Perp_DF <- data.frame(Num_Topics=seq(10,num_topics,10),Train_Perp = train_perp)
Validation_Perp_DF <- data.frame(Num_Topics = seq(10,num_topics,10),Validation_Perp = validation_perp)
ggplot(Train_Perp_DF,aes(x=Num_Topics,y=Train_Perp))+geom_line(size=1.5,linetype="dashed")+geom_jitter(size=4,shape=1,width = 0.05,height = 0.05)+
  labs(list(title ='', x = "number of topic", y = "perplexity"))+
  theme_bw()+ theme(panel.grid = element_blank())+theme(plot.title = element_text(hjust = 0.5))+
  theme(legend.key.size=unit(3,'cm'))+ theme(legend.position="bottom")+ theme(legend.title = element_blank())+
  theme(legend.text = element_text(colour = 'black', angle = 0, size = 18, hjust = 1, vjust = 1, face = 'bold'))+
  theme(axis.title.x= element_text(size=18,  color="black", face= "bold", vjust=0.5, hjust=0.5))+
  theme(axis.title.y= element_text(size=18,  color="black", face= "bold", vjust=0.5, hjust=0.5))+
  theme(axis.text.x= element_text(size=15, color="black", face= "bold", vjust=0.5, hjust=0.5))+
  theme(axis.text.y= element_text(size=15, color="black", face= "bold", vjust=0.5, hjust=0.5))




#-----------lda
library(slam)
summary(col_sums(Train.dtm));summary(col_sums(Validation.dtm));summary(col_sums(Test.dtm))
Train.dtm  <-  Train.dtm[row_sums(Train.dtm)  >  0,]
Validation.dtm  <-  Validation.dtm[row_sums(Validation.dtm)  >  0,]
Test.dtm  <-  Test.dtm[row_sums(Test.dtm)  >  0,]

best_topic <- 40
## k=8 ,文档-主题概率分布  !!!!!!!!!!!!!!!!!!!!!!!!!
SEED <- 2010
model <- LDA(Train.dtm, control = list(verbose=0,seed = SEED),method = "Gibbs", k = best_topic)

Train_topic <- posterior(model)$topics
Validation_topic <- posterior(model,newdata = Validation.dtm)$topics
Test_topic <- posterior(model,newdata = Test.dtm)$topics


#png('topic_dist.png',width=1280,height=640)
#par(mfrow=c(1,3))
#topic.image(Train_topic,main='Train data ')
#topic.image(Validation_topic,main='Validation data ')
#topic.image(Test_topic,main='Testing data ')
#dev.off()


Topic <- topics(model, 1)
table(Topic)
#每个Topic前5个Term
Terms <- terms(model, 10)
#Terms[,]
Terms_ColNames <- colnames(Terms)
Terms_ColNames <- gsub(" ","",Terms_ColNames)


#----------------combine data
Train_topic <- as.data.frame(Train_topic);colnames(Train_topic) <- Terms_ColNames
Validation_topic <- as.data.frame(Validation_topic);colnames(Validation_topic) <- Terms_ColNames
Test_topic <- as.data.frame(Test_topic);colnames(Test_topic) <- Terms_ColNames

backup_Train <- Train
backup_Validation <- Validation
backup_Test <- Test

Train <- cbind(Train,Train_topic);Train$Loan_Statement <- NULL
Validation <- cbind(Validation,Validation_topic);Validation$Loan_Statement <- NULL
Test <- cbind(Test,Test_topic);Test$Loan_Statement <- NULL

#-------------SMOTE:deal with imbalance
set.seed(50)
BalancedTrain <- SMOTE(Label~.,Train,perc.over=500,k=5,perc.under=120)
table(BalancedTrain$Label)



#---------------basic model
par(mfrow=c(1,1))
set.seed(50)
forest <- randomForest(Label~.,data = BalancedTrain,importance = TRUE,ntree=200)


ImportanceDF <- as.data.frame(importance(forest))
varImpPlot2(forest)
ggForest(forest)#write.csv(ImportanceDF,"OrgImportance.csv")

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
# "Target"         "Term"           "Time_Job"      
# "Type_Company"  
#-----cv 2
set.seed(50)
fit2 <- varSelRF(tmp_Data2[,-1],tmp_Data2[,1],ntree = 100,ntreeIterat = 500,vars.drop.frac = 0.1)
Feat_CV2 <- fit2$selected.vars
#---12 vars
# "Car"            "Education"      "House_Loan"    
# "House_Property" "Income"         "Marriage"      
# "Property_Loan"  "Scale_Company"  "Target"        
# "Term"           "Time_Job"       "Type_Company"  #-----cv 3
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
# "Term"           "Time_Job"       "Type_Company"  #-----cv 5
set.seed(50)
fit5 <- varSelRF(tmp_Data5[,-1],tmp_Data5[,1],ntree = 100,ntreeIterat = 500,vars.drop.frac = 0.1)
Feat_CV5 <- fit5$selected.vars
# #---13 vars
# "Car"            "Education"      "House_Loan"    
# "House_Property" "Income"         "Marriage"      
# "Property_Loan"  "Scale_Company"  "Target"        
# "Term"           "Time_Job"       "Type_Company"  
# "Workplace"

#-------------Feature Intersect Selection
Feat_All <- c(Feat_OOB,Feat_Gini,Feat_CV1,Feat_CV2,Feat_CV3,Feat_CV4,Feat_CV5)
Feat_Table <- table(Feat_All)
Select_Index <- which(Feat_Table>=5)
Feat_Select <- names(Select_Index)
# "Car"           "Education"     "House_Loan"    "Property_Loan" "Scale_Company"   "Target"
# "Term"          "Time_Job"      "Topic1"        "Topic2"        "Topic3"       "Topic6"
#------------------------------Strategy 1:Bases on Balanced Data Sey
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
forest <- randomForest(Label~.,data = BalancedTrain,importance = TRUE,ntree=200)

ggForest(forest)
#ImportanceDF <- as.data.frame(importance(forest))
#varImpPlot(forest)
#--------steady when n>=60 pick ntree=60
best_ntree <- 100

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


#---------lowest when mtry=4 ,pick mtry = 3
best_mtry <- 3

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

write.csv(ValidationRecord,"ValidationRecord_topic.csv")
# ggplot(data = TrainRecord,aes(x=MaxNodes,y=Accuracy))+geom_line(size=1)+geom_point(size=3)+labs(title = "Train Accuracy")+theme(plot.title=element_text(hjust=0.5))
# ggplot(data = TrainRecord,aes(x=MaxNodes,y=F1_Score))+geom_line(size=1)+geom_point(size=3)+labs(title = "Train F1_Score")+theme(plot.title=element_text(hjust=0.5))
# ggplot(data = TrainRecord,aes(x=MaxNodes,y=AUC))+geom_line(size=1)+geom_point(size=3)+labs(title = "Train AUC")+theme(plot.title=element_text(hjust=0.5))
# 
# 
# ggplot(data = ValidationRecord,aes(x=MaxNodes,y=Accuracy))+geom_line(size=1)+geom_point(size=3)+labs(title = "Validation Accuracy")+theme(plot.title=element_text(hjust=0.5))
# ggplot(data = ValidationRecord,aes(x=MaxNodes,y=F1_Score))+geom_line(size=1)+geom_point(size=3)+labs(title = "Validation F1_Score")+theme(plot.title=element_text(hjust=0.5))
# ggplot(data = ValidationRecord,aes(x=MaxNodes,y=AUC))+geom_line(size=1)+geom_point(size=3)+labs(title = "Validation AUC")+theme(plot.title=element_text(hjust=0.5))


#set.seed(50)
maxnodes_index <- which.max(ValidationRecord$F1_Score)
best_maxnodes <- ValidationRecord[maxnodes_index,]$MaxNodes
best_cutoff <- ValidationRecord[maxnodes_index,]$cutoff
#write.csv(ValidationRecord,"ValidationRecord_Topic.csv")
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



confusionMatrix(pred.test.forest,Test[,1],positive = "1")



#-----------------if you want picture!



term_tfidf  <- tapply(loan.dtm$v/row_sums( loan.dtm)[ loan.dtm$i],loan.dtm$j,mean)*log2(nDocs(loan.dtm)/col_sums(loan.dtm  >  0))
summary(term_tfidf)
loan.dtm  <-  loan.dtm[,  term_tfidf  >=  0.1]
loan.dtm  <-  loan.dtm[row_sums(loan.dtm)  >  0,]
library(topicmodels)
###########################################################

#########################################################



k <- 30
SEED <- 2010



loan_TM <-
  list(
    VEM = LDA(loan.dtm, k = k, control = list(seed = SEED)),
    VEM_fixed = LDA(loan.dtm, k = k,control = list(estimate.alpha = FALSE, seed = SEED)),
    Gibbs = LDA(loan.dtm, k = k, method = "Gibbs",control = list(seed = SEED, burnin = 1000,thin = 100, iter = 1000)),
    CTM = CTM(loan.dtm, k = k,control = list(seed = SEED,var = list(tol = 10^-4), em = list(tol = 10^-3)))
  )


sapply(loan_TM[1:2], slot, "alpha")
sapply(loan_TM, function(x) mean(apply(posterior(x)$topics,1, function(z) - sum(z * log(z)))))

Topic <- topics(loan_TM[["VEM"]], 1)
table(Topic)
#每个Topic前5个Term
Terms <- terms(loan_TM[["VEM"]], 5)
Terms[,1:10]

######### auto中每一篇文章中主题数目
(topics_auto <-topics(loan_TM[["VEM"]])[ grep("auto", csv[[1]]) ])
most_frequent_auto <- which.max(tabulate(topics_auto))
######### 与auto主题最相关的10个词语
terms(loan_TM[["VEM"]], 10)[, most_frequent_auto]



#-----------------------
setwd('E:/myRproj/trunk/CloudAtlas/')
load('rda/cloudAtlas500.rda')
source('postProcess.R',encoding="UTF-8")


#### 将主题设定为2～25
ldas<-lda.rep(train=train_dtm,test=test_dtm,n=25,method='VEM')



####困惑度的计算
train_perp=ldas$train_perp
test_perp=ldas$test_perp
loglik=ldas$loglik
est=data.frame(x=c(1:25,1:25),
               y=c(test_perp,train_perp),
               type=c(rep('test_perplexity',25),
                      rep('train_perplexity',25) ))

png('png/perplextity.png',width=720,height=480)
ggplot(est, aes(x, y, color=type)) + 
  geom_line() + geom_point()+ 
  facet_wrap(~ type, ncol=2)+
  labs(list(title ='', x = "number of topic", y = "perplexity"))
dev.off()
####熵计算
alpha=sapply(ldas[[1]], slot, "alpha")
entropy=sapply(ldas[[1]], function(x) mean(apply(posterior(x)$topics, 1, 
                                                 function(z) - sum(z * log(z)))))
ae=data.frame(x=c(1:24,1:24),y=c(alpha,entropy),
              type=c(rep('alpha',24),rep('entropy',24)))

png('png/entropy.png',width=720,height=480)
ggplot(ae, aes(x, y, color=type)) + 
  geom_line() + geom_point()+ 
  facet_wrap(~ type, ncol=2)+
  labs(list(title ='', x = "number of topic", y = "alpha and entropy"))
dev.off()
# png('perplextity.png',width=640,height=320)
# par(mfrow=c(1,3),mar=c(2,4,3,1))
# plot(train_perp,type='o',col=2,main='perplexity of training data',ylab='perplexity')
# plot(test_perp,type='o',col=3,main='perplexity of testing data',ylab='perplexity')
# plot(loglik,type='o',col=4,main='loglik of training data',ylab='loglik')
# dev.off()
####
#library(textir)
#summary(simselect <- topics(dtm_pos, K=10+c(-8:10)), nwrd=0)
# => k=2
###########################################################
## k=8 ,文档-主题概率分布  !!!!!!!!!!!!!!!!!!!!!!!!!
model<-LDA(loan.dtm, control = list(alpha=0.015,verbose=1),method='Gibbs', k = 10)
#model=ldas$model[[7]]
###
loan_topic<-posterior(model)$topics
test_topic<-posterior(model,newdata=test_dtm)$topics
png('topic_dist.png',width=1280,height=640)
par(mfrow=c(1,1))
topic.image(loan_topic,main='loan data ')
topic.image(test_topic,main='Testing data ')
dev.off()
#########################################################
####随机抽取10个样本，显示主题分布
prop=prop2df(lda=model,terms.top.num=5,doc.num=10)
propor=prop$lda.prop
png('topic_rnd10.png',width=640,height=480)
qplot(y=proportion,x=topic, fill=doc, position="dodge",data=propor, geom="bar",stat="identity",facets=~doc)+coord_flip()+facet_wrap(~ doc, ncol=5)
dev.off()
prop2=prop2df(lda=model,newdata=test_dtm,terms.top.num=5,doc.num=10)
propor2=prop2$lda.prop
png('png/topic_rnd10_test.png',width=640,height=480)
qplot(y=proportion,x=topic, fill=doc, position="dodge",
      data=propor2, geom="bar",stat="identity",facets=~doc)+
  coord_flip()+
  facet_wrap(~ doc, ncol=5)
dev.off()

####################################
#### 抽取每个主题的前10个词汇，绘制网络图  !!!!!!
top_terms<-terms(model, k=3, threshold=0.002)
g<-topic.graph(top_terms)
g2<-topic.graph(terms(model, k=10, threshold=0.002))
g4<-topic.graph(terms(ldas$model[[4]], k=10, threshold=0.002))
gg<-graph.data.frame(g[,1:2])
gg2<-graph.data.frame(g2[,1:2])
gg4<-graph.data.frame(g4[,1:2])
png('top_terms_graph.png',width=960,height=320)
par(mfrow=c(1,1),mar=c(0,0,2,0))
plot(gg2, layout=layout.fruchterman.reingold,
     vertex.size=10,
     vertex.color='gray90',
     vertex.label.cex=2,
     edge.color=as.integer(g4[,3]),
     edge.arrow.size=0.5,
     main='2 Topics')
plot(gg4, layout=layout.fruchterman.reingold,
     vertex.size=10,
     vertex.color='gray90',
     vertex.label.cex=2,
     edge.color=as.integer(g4[,3]),
     edge.arrow.size=0.5,
     main='5 Topics')
plot(gg, layout=layout.fruchterman.reingold,
     vertex.size=10,
     vertex.color='gray90',
     vertex.label.cex=1,
     edge.color=as.integer(g[,3]),
     edge.arrow.size=0.5,
     main='8 Topics')
dev.off()
#####################################################################
dtm<-df2dtm(cloudAtlas,content='word',word.min=2)
dim(dtm)
k <- 8;SEED <- 2013
CA_TM <-list(VEM = LDA(dtm, k = k, control = list(seed = SEED)),
             VEM_fixed = LDA(dtm, k = k,control = list(estimate.alpha = FALSE, seed = SEED)),
             Gibbs = LDA(dtm, k = k, method = "Gibbs",
                         control = list(seed = SEED, burnin = 1000, thin = 100, iter = 1000)),
             CTM = CTM(dtm, k = k, control = list(seed = SEED, 
                                                  var = list(tol = 10^-4), em = list(tol = 10^-3))))
#save(CA_TM,file='rda/CA_TM.rda')
load('rda/CA_TM.rda')
#### 文档分配给最有可能的话题的概率分布
sapply(loan_TM[1:2], slot, "alpha")
methods <- c("VEM", "VEM_fixed", "Gibbs", "CTM")
DF <- data.frame(posterior = unlist(lapply(loan_TM, 
                                           function(x) apply(posterior(x)$topics, 1, max))),
                 method = factor(rep(methods, each = nrow(posterior(loan_TM$VEM)$topics)), 
                                 methods))
png('maxProb.png',width=720,height=320)
ggplot(DF, aes(x=posterior, fill=method))+
  geom_histogram(binwidth = 0.05)+facet_wrap(~ method,ncol=4) +
  ylab("Frequency")+
  xlab ("Probability of assignment to the most likely topic")
dev.off()

DF2 <- data.frame(topic_class = unlist(lapply(CA_TM, 
                                              function(x) apply(posterior(x)$topics, 1, which.max))),
                  method = factor(rep(methods, each = nrow(posterior(CA_TM$VEM)$topics)), 
                                  methods))

ggplot(DF2, aes(x=topic_class, fill=method))+
  geom_histogram(binwidth = 1)+facet_wrap(~ method,ncol=4)+
  ylab("Frequency")
#xlab ("Probability of assignment to the most likely topic")
vem_class=DF2[1:495,1]
library(slam)
####################################
#### 合并同一主题的词频
vemsum<-DTM_Topic_SUM(dtm,vem_class)
Gibbssum<-DTM_Topic_SUM(dtm,DF2[991:1485,1])
dim(vemsum)
vemsum<-as.matrix(vemsum)
Gibbssum<-as.matrix(Gibbssum)
#########################################
#### 余玄相似度
sim<-cosine(vemsum)
sim2<-cosine(Gibbssum)
png('png/cosine.png',width=950,height=480)
par(mfrow=c(1,2),mar=c(2,2,2,0))
image(sim,xaxt="n", yaxt="n",main="LDA_VEM")
axis(1, at=seq(0,1,length=8),labels=paste('Topic',1:8), tick=F ,line=-.5)
axis(2, at =seq(0,1,length=8),label=paste('Topic',1:8))
image(sim2,xaxt="n", yaxt="n",main="LDA_Gibbs")
axis(1, at=seq(0,1,length=8),labels=paste('Topic',1:8), tick=F ,line=-.5)
axis(2, at =seq(0,1,length=8),label=paste('Topic',1:8))
dev.off()
#################################################################
dtm_pos<-df2dtm(cloudAtlas,content='wordPos',word.min=2)
ldas_pos<-lda.rep(train=dtm_pos,n=25,method='VEM')

entropy=sapply(ldas_pos[[1]], function(x) mean(apply(posterior(x)$topics, 1, 
                                                     function(z) - sum(z * log(z)))))


