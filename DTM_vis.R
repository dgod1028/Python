
dat <- read.csv("d:/study/paper/topwords6.csv",stringsAsFactors = F)
docd <- read.csv("d:/study/paper/docdis6.csv",stringsAsFactors = F)

t <- read.csv("d:/study/paper/t.csv",stringsAsFactors = F)[,2]


top <- list()

for(i in 1:11){
  top[[i]] = dat[,((i-1)*10 +1):((i-1)*10+10)]
  colnames(top[[i]]) = c("Topic 1","prob.","Topic 2","prob.","Topic 3","prob.","Topic 4","prob.","Topic 5","prob.")
}

#sink("d:/study/paper/topword001.txt")
for(i in 11:1){
  if( i %% 2 != 0){
    cat(paste("T",2006+12-i,"\n"));print(top[[i]]);cat("\n")
  }

  }
#sink()

doc2 <- cbind(docd,t)
doc3 <- doc2[,c(2:6,57)]
dd <- list()
ddm <- matrix(0,11,5)
for (i in 1:11){
  dd[[i]] = doc3[doc3$t ==i,] 
  ddm[i,] = colSums(dd[[i]][,1:5])
}
ddmp = ddm
for (i in 1:11){
  ddmp[i,] = ddm[i,]/sum(ddm[i,])
}

png("d:/study/paper/006.png",width=600,height = 300)
plot(2007:2017,ddmp[,1],type="l",col="red",ylim=c(0.1,0.5),main="alpha = 0.1 beta = 0.1 seed = 2 ",lwd=2)
points(2007:2017,ddmp[,1],pch=1)
co = c("red","blue","green","black","orange")
for(i in 2:5){
  lines(2007:2017,ddmp[,i],col=co[i],lwd=2)
  points(2007:2017,ddmp[,i],pch=i,col=co[i])
}
legend("topright",legend = c(1,2,3,4,5),col=co,pch=1:5,lty=1,lwd=2)
dev.off()
