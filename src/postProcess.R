library(topicmodels)
library(tm)
library(igraph)
library(ggplot2)
library(slam)
##########################################
#### 过滤特定字符
remove.chars<-function(x,n=1,pattern='[0-9]'){
  x<-as.character(x)
  x<-unlist(strsplit(x,' '))
  x<-gsub(pattern,'',x)
  x<-x[nchar(x)>n]
  x<-paste(x,collapse=' ')
  names(x)<-NULL
  x
}


#####################################################################
#### 循环计算lda的perplexity
lda.rep<-function(train,validation=NULL,n_topic=100,method='VEM'){
  model<-vector('list')
  train_perp<-validation_perp<-c()
  loglik<-c()
  SEED <- 2010
  n=seq(10,n_topic,10)
  for(i in 1:length(n)){
    cat('---------------- k=',i,' --------------\n')
    model[[i-1]]<-LDA(train, control = list(verbose=0,seed = SEED),method, k = n[i])
    train_perp[i]<-perplexity(model[[i-1]],newdata = train)
    if(!is.null(validation)){
      validation_perp[i]<-perplexity(model[[i-1]],newdata=validation)
    }
    
    loglik[i]<-logLik(model[[i-1]])[1]
  }
  return(list(model=model,
              train_perp=train_perp,
              validation_perp=validation_perp,
              loglik=loglik))
}
#####################################################################
#### 将terms转化为left-right结构的data.frame，可进一步转化为graph.data.frame
topic.graph<-function(x){
  v2g<-function(x){
    nr=length(x)-1
    g<-c()
    nr=1:nr
    for(i in nr){
      g0=cbind(x[i],x[i+1])
      g<-rbind(g,g0)
    }
    g
  }
  ###
  gg<-c()
  if(class(x)=='matrix'){
    nc=1:ncol(x)
    for(i in nc){
      gg0<-v2g(x[,i])
      gg0=cbind(gg0,rep(i,nrow(gg0)))
      gg<-rbind(gg,gg0)
    }
  }
  if(class(x)=='list'){
    nc=1:length(x)
    for(i in nc){   
      gg0<-v2g(x[[i]])
      gg0=cbind(gg0,rep(i,nrow(gg0)))
      gg<-rbind(gg,gg0)
    }
  }
  colnames(gg)<-c('source','target','type')
  return(gg)
}
###################################################
####  随机提取文档的主题比例
prop2df<-function(lda,newdata=NULL,terms.top.num=6,doc.num=8){
  if(is.null(newdata))
    pos=posterior(lda)
  if(!is.null(newdata))
    pos=posterior(lda,newdata)
  pos_term<-pos$terms
  pos_topic<-pos$topics
  top_terms<-terms(lda, k=terms.top.num, threshold=0.002)
  top_names<-apply(top_terms, 2, paste, collapse=".")
  nr=nrow(pos_topic)
  if(doc.num>nr)doc.num=nr
  pt=pos_topic[sample(1:nr,doc.num),]
  prop<-c()
  for(i in 1:doc.num){
    prop<-c(prop,pt[i,])
  }
  nc=ncol(pos_topic)
  propor<-data.frame(topic=rep(top_names,doc.num),
                     proportion=prop,
                     doc=as.factor(rep(1:doc.num,rep(nc,doc.num))) )
  
  pos_topic<-cbind(pos_topic,type=apply(pos_topic,1,which.max))
  return(list(lda.prop=propor,doc.sample=pt,topics=pos_topic))
}
##########################################################################
#### Document-Topic 权重分布图
topic.image<-function(topic,main=NULL,by=10){
  #topic<-t(posterior(lda)$topics)
  topic=t(topic)
  nc=nrow(topic);nr=ncol(topic)
  w=as.integer(as.factor(topic))
  w=w/max(w) 
  by<-floor(nr/by)
  op<-par(xpd=TRUE, mar=c(5.1,4.1,2.1,2))
  image(topic,col=gray(w),xaxt="n", yaxt="n",
        ylab='Document',xlab='Topic',font.lab=3,
        main=paste(main,"Topic-Document Weights" ))
  axis(side=1, at=(1:nc)/nc-1/(2*nc), 
       labels=paste('Topic',1:nc), tick=F ,line=-.5)
  axis(2, at=seq(0,nr,by=by)/nr,label=seq(0,nr,by=by))
  par(op)
}

####################################
#### 合并同一主题的词频
DTM_Topic_SUM<-function(dtm,class){
  #n=length(class)
  m=length(table(class))
  x<-dtm[1:m,]
  row.names(x)<-paste('Topic',1:m,sep="_")
  #x=simple_triplet_zero_matrix(nrow=m,ncol=ncol(dtm))
  for( i in 1:m){
    x[i,]<-col_sums(dtm[class==i,])
  }
  #colnames(x)<-colnames(dtm)  
  x
}
#########################################
#### 余玄相似度
cosine<-function(x){
  .cosine<-function(x,y){
    x<-as.vector(x)
    y<-as.vector(y)
    sum(x*y)/(sqrt(sum(x^2))*sqrt(sum(y^2)))
  }
  nr<-nrow(x)
  simmat<-matrix(nrow=nr,ncol=nr)
  for(i in 1:nr){
    for(j in i:nr){
      simmat[i,j]<-.cosine(x[i,],x[j,])
    }
  }
  return(simmat)
}