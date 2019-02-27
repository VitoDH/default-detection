library(data.table)
library(plyr)
library(e1071)
library(randomForest)
library(parallel)




setwd("~/Documents/Study/ComplexDataAnalysis/Homework/big homework/IJCAI15 Data/data_format1")
user_log <- fread("user_log_format1.csv",header = T)
user_info <- fread("user_info_format1.csv",header = T)
user_info <- na.omit(user_info)
trainSet <- fread("train_format1.csv",header = T);colnames(trainSet)[2]<-"seller_id"
trainSet <- trainSet[1:100000,]
#testSet <- fread("test_format1.csv",header = T);colnames(testSet)[2]<-"seller_id"
#testSet <- testSet[1:1000,]
trainLength <- nrow(trainSet)
testLength <- nrow(testSet)


#-----------------------------------------------------Direct Connection
#将用户行为与训练集
train <- merge(user_log,trainSet,by=c("user_id","seller_id"))
train <- merge(train,user_info,by=c("user_id"))   #由于user_info内包含缺失值，所以合并后训练集的样本会减少
#test <- merge(user_log,testSet,by=c("user_id","seller_id"))#test <- user_log[(user_log$user_id %in% testSet$user_id) & (user_log$seller_id %in% testSet$merchant_id),]


sgbTrain <- split(train,by=c("user_id","seller_id"))          #split group by train
#sgbTest <- split(test,by=c("user_id","seller_id")) 


classifyActionType <- function(tempTable){      #导入要对action type进行分类统计的
  act_type <- as.data.frame(table(tempTable$action_type))
  rownum <- nrow(act_type)
  act_type <- as.numeric(as.matrix(act_type))   
  act_type <- matrix(act_type,nrow = rownum)
  tmp1 <- setdiff(0:3,act_type[,1])      #act_type第一列是action_type种类，第二列是对应的频数
  if(length(tmp1)==0){tmp3 <- act_type   #若包含了所有action，则不用修改，直接赋值
  }else{                                 #若不然，补全数据框，使之包含四种action
    tmp2 <- cbind(tmp1,0)
    tmp3 <- rbind(act_type,tmp2) 
  }
  sortVec <- tmp3[order(tmp3[,1]),][,2]   #重新对一列排序，使之按照点击、加入购物车、购买、喜欢的顺序
  return(sortVec)
}



direct_link <-function(tableTrain){
  directActionVec <- classifyActionType(tableTrain)
  label <- as.numeric(tableTrain[1,8])      #提取标签
  age_range <- as.numeric(tableTrain[1,9])
  gender <-as.numeric(tableTrain[1,10])
  user_id <- as.numeric(tableTrain[1,1])
  seller_id <- as.numeric(tableTrain[1,2])
  directVec <- c(user_id,seller_id,directActionVec,age_range,gender,label)
  return(directVec)
}
#pp<-direct_link(sgbTrain[[5]])
#---------------------begin parallel
cores <- detectCores()-1
cl <- makeCluster(cores)
clusterExport(cl,"classifyActionType")
direct_linkVec <- parLapply(cl,sgbTrain, direct_link)     #得到用户-商家的直接汇总记录
stopCluster(cl)

list2Vec <- unlist(direct_linkVec)
Vec2Mat <- matrix(list2Vec,ncol = 9,byrow = T)
colnames(Vec2Mat) <- c("user_id","seller_id","click","cart","buy","favorite","age_range","gender","label")

#direct_linkVecTest <- lapply(sgbTest, direct_link)
#list2VecTest <- unlist(direct_linkVecTest)
#Vec2MatTest <- matrix(list2VecTest,ncol = 5,byrow = T)
#colnames(Vec2MatTest) <- c("click","cart","buy","favorite","label")


trainMat <- as.data.frame(Vec2Mat)
trainMat$label <- as.factor(trainMat$label)






#--------------------------------------------------Indirect Connection
trainUserId <- trainSet$user_id
trainSellerId <- trainSet$seller_id

clusterSetUser <- user_log[user_log$user_id %in% trainUserId,]
clusterSetUser <- clusterSetUser[order(user_id)]


clusterSetSeller <- user_log[user_log$seller_id %in% trainSellerId,]
clusterSetSeller <- clusterSetSeller[order(seller_id)]


clusterUser <- split(clusterSetUser,by="user_id")          #为用户建立数据库
clusterSeller <- split(clusterSetSeller,by="seller_id")    #为商家建立数据库




BuildFeatVec <- function(tempList){        #------------用于对clusterUser和clusterSeller的lapply
  #-----------------obtain the the vector of catagories 
  userList <- tempList[order(cat_id)]       #对商品种类按数字进行排序
  catSplit <- split(userList,by = "cat_id")      #对某用户根据其浏览过的商品种类来分类
  catAction <- lapply(catSplit,classifyActionType)  #返回某个用户的关于商品种类的基础特征向量(未加权)，形式为列表
  catActionWeightedMean <- unlist(lapply(catAction,weighted.mean,w= c(0.1,0.2,0.3,0.4))) #用户-种类的加权特征向量
  catActionSum <- mean(catActionWeightedMean)    #提取均值作为得分
  #names(catActionWeightedMean) <- paste0(names(catActionWeightedMean),"-cat")
  
  #-----------------obtain the vector of brands
  userList <- tempList[order(brand_id)]       #对商品种类按数字进行排序
  brandSplit <- split(userList,by = "brand_id")
  brandAction <- lapply(brandSplit,classifyActionType)  #返回某个用户的关于商品种类的基础特征向量(未加权)，形式为列表
  brandActionWeightedMean <- unlist(lapply(brandAction,weighted.mean,w= c(0.1,0.2,0.3,0.4))) #用户-种类的加权特征向量
  brandActionSum <- mean(brandActionWeightedMean)
  #names(brandActionWeightedMean) <- paste0(names(brandActionWeightedMean),"-brand")
  
  #UserVec <- c(catActionWeightedMean,brandActionWeightedMean)
  UserVec <- c(catActionSum,brandActionSum)
}

#------------------------提取用户特征
cores <- detectCores()-1
cl <- makeCluster(cores)
clusterExport(cl,"classifyActionType")
clusterExport(cl,"data.table")     #将必要的函数和包导入并行核中
#UserfeatVec <- lapply(clusterUser[1:5],BuildUserfeatVec)

system.time({UserfeatVec <- parLapply(cl,clusterUser,BuildFeatVec)}) 
UserfeatData <- t(as.data.frame(UserfeatVec))
#kc <- kmeans(UserfeatData,5)
UserRowNames <- gsub("[^0-9]", "", rownames(UserfeatData)) 
rownames(UserfeatData) <- UserRowNames 
UserfeatData <- cbind(as.numeric(UserRowNames),UserfeatData)
UserfeatData <- as.data.frame(UserfeatData)
colnames(UserfeatData) <- c("user_id","UserCatScore","UserBrandScore")

#tmp<-BuildUserfeatVec(clusterUser[[1]])
stopCluster(cl)



#-------------------提取商家的特征
cores <- detectCores()-1
cl <- makeCluster(cores)
clusterExport(cl,"classifyActionType")
clusterExport(cl,"data.table")     #将必要的函数和包导入并行核中
#SellerfeatVec <- lapply(clusterSeller[1:5],BuildSellerfeatVec)

system.time({SellerfeatVec <- parLapply(cl,clusterSeller,BuildFeatVec)}) 
SellerfeatData <- t(as.data.frame(SellerfeatVec))
#kc <- kmeans(SellerfeatData,5)
SellerRowNames <- gsub("[^0-9]", "", rownames(SellerfeatData)) 
rownames(SellerfeatData) <- SellerRowNames 
SellerfeatData<-cbind(as.numeric(SellerRowNames),SellerfeatData)
SellerfeatData <- as.data.frame(SellerfeatData)
colnames(SellerfeatData) <- c("seller_id","SellerCatScore","SellerBrandScore")

#tmp<-BuildSellerfeatVec(clusterSeller[[1]])
stopCluster(cl)

FinalMat <- merge(trainMat,SellerfeatData,by="seller_id")
FinalMat <- merge(FinalMat,UserfeatData,by="user_id")
label <- FinalMat$label
FinalMat$label <- NULL
FinalMat <- cbind(FinalMat,label)
write.csv(FinalMat,"trainSet3.csv")




FinalMat1 <- subset(FinalMat,select=-c(user_id,seller_id))
trainMat1 <-FinalMat1[1:8000,]      #切分成训练和测试看效果
testMat1 <- FinalMat1[8001:9854,]   #删掉了有缺失值（年龄，性别）的用户，所以不够10000个


#testMat <- as.data.frame(Vec2MatTest)
#testMat$label <- as.factor(testMat$label)

#--------------svm
x <- subset(trainMat1, select = -label)
y <- trainMat1$label
model <- svm(x, y)
print(model)
summary(model)
x <- subset(testMat1, select = -label)
y <- testMat1$label
pred <- predict(model, x)

# Check accuracy:
table(pred, y)

#--------------randomForest
model.forest <- randomForest(label~.,data = trainMat1)
pre.forest <- predict(model.forest,testMat1)
table(pre.forest,testMat1$label)

