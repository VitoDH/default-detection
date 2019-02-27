#--------将空白处转为缺失值
Blank2NA <- function(x){
  x_out <- rep(0,nrow(x))
  blankIndex <- which(x=="")
  x_out[blankIndex] <- NA
  return(x_out)
}

#----------plot.forest ggplot version
##--point_type 4,15,16,17
ggForest <- function(x){   #x:forest object
  #pd <- position_dodge(5)
  ErrorMat <- x$err.rate
  num_row <- nrow(ErrorMat)
  sample_index <- seq(5,num_row,10)
  ErrorMat <- cbind(1:x$ntree,ErrorMat)
  ErrorMat <- as.data.frame(ErrorMat)
  ErrorMat <- ErrorMat[sample_index,]
  colnames(ErrorMat) <- c("Num_Trees","OOB_Error","0_Error","1_Error")
  melt_ErrorMat <- melt(ErrorMat,id.vars = "Num_Trees")
  colnames(melt_ErrorMat)[3] <- "Error"
  
  ggplot(melt_ErrorMat, aes(x=Num_Trees, y=Error, group = variable)) +
    geom_line(aes(linetype=variable), # 线的类型取决于variable 
              size = 1.5) +       # 粗线 
    #geom_point(aes(shape=variable),   # 点形状取决于variable 
             #  size = 3,position = "jitter") +        # 大型点标记 
    scale_shape_manual(values=c(0,1,2)) +                  # 改变点形状 
    scale_linetype_manual(values=c("longdash","dotted", "twodash"))+ # 改变线类型 
    theme_bw()+ theme(panel.grid = element_blank())+theme(plot.title = element_text(hjust = 0.5))+
    theme(legend.key.size=unit(3,'cm'))+ theme(legend.position="bottom")+ theme(legend.title = element_blank())+
    theme(legend.text = element_text(colour = 'black', angle = 0, size = 18, hjust = 1, vjust = 1, face = 'bold'))+
    theme(axis.title.x= element_text(size=15,  color="black", face= "bold", vjust=0.5, hjust=0.5))+
    theme(axis.title.y= element_text(size=15,  color="black", face= "bold", vjust=0.5, hjust=0.5))+
    theme(axis.text.x= element_text(size=12, color="black", face= "bold", vjust=0.5, hjust=0.5))+
    theme(axis.text.y= element_text(size=12, color="black", face= "bold", vjust=0.5, hjust=0.5))
  
}

#-----compare train-validation performance
ggCompare <- function(row_Vec,train_Vec,validation_Vec,x_axis,type,ylim){
  BasicMat <- data.frame(row_Vec,train_Vec,validation_Vec)
  colnames(BasicMat) <- c(x_axis,paste("Train_",type),paste("Validation_",type))
  melt_BasicMat <- melt(BasicMat,id.vars = x_axis)
  colnames(melt_BasicMat)[3] <- type
  
  ggplot(melt_BasicMat, aes(x=melt_BasicMat[,x_axis], y=melt_BasicMat[,type], group = variable)) +
    geom_line(aes(linetype=variable), # 线的类型取决于variable 
              size = 1.2) +       # 粗线 
    #geom_point(aes(shape=variable),   # 点形状取决于variable 
     #          size = 3) +        # 大型点标记 
    scale_shape_manual(values=c(1,1)) +                  # 改变点形状 
    scale_linetype_manual(values=c("dashed","dotted"))+ # 改变线类型 
    scale_y_continuous(breaks=ylim)+ 
    theme_bw()+ theme(panel.grid = element_blank())+labs(x = x_axis,y = type)+theme(plot.title = element_text(hjust = 0.5))+
    theme(legend.key.size=unit(3,'cm'))+ theme(legend.position="bottom")+ theme(legend.title = element_blank())+
    theme(legend.text = element_text(colour = 'black', angle = 0, size = 18, hjust = 1, vjust = 1, face = 'bold'))+
    theme(axis.title.x= element_text(size=15,  color="black", face= "bold", vjust=0.5, hjust=0.5))+
    theme(axis.title.y= element_text(size=15,  color="black", face= "bold", vjust=0.5, hjust=0.5))+
    theme(axis.text.x= element_text(size=13, color="black", face= "bold", vjust=0.5, hjust=0.5))+
    theme(axis.text.y= element_text(size=13, color="black", face= "bold", vjust=0.5, hjust=0.5))
}

#----------varImp modification
varImpPlot2 <- function(x, sort=TRUE,
                       n.var=min(30, nrow(x$importance)),
                       type=NULL, class=NULL, scale=TRUE, 
                       main=deparse(substitute(x)), ...) {
  if (!inherits(x, "randomForest"))
    stop("This function only works for objects of class `randomForest'")
  imp <- importance(x, class=class, scale=scale, type=type, ...)
  colnames(imp) <- c("0","1","Mean Decrease Accuracy","Mean Decrease Gini")
  ## If there are more than two columns, just use the last two columns.
  if (ncol(imp) > 2) imp <- imp[, -(1:(ncol(imp) - 2))]
  nmeas <- ncol(imp)
  if (nmeas > 1) {
    op <- par(mfrow=c(1, 2), mar=c(4, 5, 4, 1), mgp=c(2, .8, 0),
              oma=c(0, 0, 2, 0), no.readonly=TRUE)
    on.exit(par(op))
  }
  for (i in 1:nmeas) {
    ord <- if (sort) rev(order(imp[,i],
                               decreasing=TRUE)[1:n.var]) else 1:n.var
    xmin <- if (colnames(imp)[i] %in%
                c("IncNodePurity", "MeanDecreaseGini")) 0 else min(imp[ord, i])
    dotchart(imp[ord,i], xlab=colnames(imp)[i], ylab="",
             main=if (nmeas == 1) main else NULL,
             xlim=c(xmin, max(imp[,i])), cex=1.2,cex.lab=1.2,cex.axis=1.2,font=2,font.lab=2,font.axis=2,...)
  }
  if (nmeas > 1) mtext(outer=TRUE, side=3, text="Variable Importance", cex=1.2)
  invisible(imp)
}


#-----------------------------Classification
FScore <- function(x,beta=1){
  precision <- x$precision; recall <- x$recall
  F <- (1+beta^2)*(precision*recall)/(beta^2*precision+recall)
}

#---------------------------将定序变量从中文转为数字
Chinese2Number <- function(input){
  colName <- input[1]
  x <- input[-1]
  x_out <- rep(0,length(x))
  
  x_table <- table(x)
  x_name <- names(x_table)
  x_num <- length(x_name)
  if(""%in%x_name){
    x_num <- x_num - 1
  }
  
  if(x_num == 2){
    x_vec <- (0:1)
  }else{
    x_vec <- (1:x_num)
  }
  emptyIndex <- which(x_name == "")
  if(length(emptyIndex) != 0){
    names(x_vec) <- x_name[-emptyIndex]
  }else{
    names(x_vec) <- x_name
  }
  
  for(i in 1:length(x)){
    if(as.character(x[i]) == ""){
      x_out[i] <- NA
    }else{
      x_out[i] <- as.character(x_vec[as.character(x[i])])
    }
  }
  if(file.exists("transformation.txt")== TRUE){
    write.table("\n",file = "transformation.txt",append = TRUE,row.names = FALSE,col.names = FALSE,quote = FALSE)
  }
  write.table(colName,file = "transformation.txt",append = TRUE,row.names = FALSE,col.names = FALSE,quote = FALSE)
  write.table(cbind(x_vec,names(x_vec)),file = "transformation.txt",append = TRUE,row.names = FALSE,col.names = FALSE,quote = FALSE)
  return(x_out)
}

#---------------检测异常值
outlierKD <- function(dt, var) {
  var_name <- eval(substitute(var),eval(dt))
  tot <- sum(!is.na(var_name))
  na1 <- sum(is.na(var_name))
  m1 <- mean(var_name, na.rm = T)
  par(mfrow=c(2, 2), oma=c(0,0,3,0))
  boxplot(var_name, main="With outliers")
  hist(var_name, main="With outliers", xlab=NA, ylab=NA)
  outlier <- boxplot.stats(var_name)$out
  mo <- mean(outlier)
  var_name <- ifelse(var_name %in% outlier, NA, var_name)
  boxplot(var_name, main="Without outliers")
  hist(var_name, main="Without outliers", xlab=NA, ylab=NA)
  title("Outlier Check", outer=TRUE)
  na2 <- sum(is.na(var_name))
  cat("Outliers identified:", na2 - na1, "\n")
  cat("Propotion (%) of outliers:", round((na2 - na1) / tot*100, 1), "\n")
  cat("Mean of the outliers:", round(mo, 2), "\n")
  m2 <- mean(var_name, na.rm = T)
  cat("Mean without removing outliers:", round(m1, 2), "\n")
  cat("Mean if we remove outliers:", round(m2, 2), "\n")
  response <- readline(prompt="Do you want to remove outliers and to replace with NA? [yes/no]: ")
  if(response == "y" | response == "yes"){
    dt[as.character(substitute(var))] <- invisible(var_name)
    assign(as.character(as.list(match.call())$dt), dt, envir = .GlobalEnv)
    cat("Outliers successfully removed", "\n")
    return(invisible(dt))
  } else{
    cat("Nothing changed", "\n")
    return(invisible(var_name))
  }
}

#-------用中值填补缺失值
ImputeMedianBatch <- function(x){
  return(impute(x,median))
} 

#-------标准化
Normalize <- function(x){
  x_out <- (x - min(x))/(max(x) - min(x))
  return(x_out)
}

#-------找到缺失值
FindNAIndex <- function(x){
  NAIndex <- which(is.na(x)==TRUE)
  return(NAIndex)
}


#------去除文本中的数字
removeNumbers <- function(x) { ret = gsub("[0-9０１２３４５６７８９]","",x) }

#------分词
wordsegment<- function(x) {
  library(Rwordseg)
  segmentCN(x)
}
#-------去除省、市
removeProCity <- function(x){
  x_out <- substring(x,1,nchar(x)-1)
  return(x_out)
}

#-------去除停用词
removeStopWords <- function(x,words) {
  ret = character(0)
  index <- 1
  it_max <- length(x)
  while (index <= it_max) {
    if (length(words[words==x[index]]) <1) ret <- c(ret,x[index])
    index <- index +1
  }
  ret
}