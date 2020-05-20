data <- acs2017_county_data
library(ISLR)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(caTools)
library(glmnet)
library(caTools)
library(dplyr)
dim(data)
sum(complete.cases(data))
apply(data, 2, function(x) any(is.na(x)))
data$ChildPoverty[is.na(data$ChildPoverty)] <- round(mean(data$ChildPoverty, na.rm = TRUE))
apply(data, 2, function(x) any(is.na(x)))
sum(complete.cases(data))
dim(data)

data_std <- data %>% select(4:37)
return_std <- sapply(data_std, function(x) x/sd(x))
return_std_df <- as.data.frame((return_std))
glimpse(data)

n=3220
e              =     rnorm(n, mean = 0, sd = 1)
beta.star      =     c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)

X <- data.matrix(data_std, rownames.force = NA)

y              =      X %*%beta.star + e

n.train        =     floor(0.8*n)
n.test         =     n-n.train
p=34
M              =     100
Rsq.test.rf    =     rep(0,M)  # rf= randomForest
Rsq.train.rf   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  #en = elastic net
Rsq.train.en   =     rep(0,M)
Rsq.test.la    =     rep(0,M)  #la = lasso
Rsq.train.la   =     rep(0,M)
Rsq.test.ri    =     rep(0,M)  #ri = ridge
Rsq.train.ri   =     rep(0,M)
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net
  cv.fit           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  
  
  
  # fit RF and calculate and record the train and test R squares 
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat       =     predict(rf, X.test)
  y.train.hat      =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.en=%.2f| Rsq.train.rf=%.2f,  Rsq.train.en=%.2f| \n", m,  Rsq.test.rf[m], Rsq.test.en[m],  Rsq.train.rf[m], Rsq.train.en[m]))
  
}
plot(cv.fit)




for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit lasso and calculate and record the train and test R squares 
  a=0 # lasso
  cv.fit           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.la[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.la[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  
  
  
  # fit ridge and calculate and record the train and test R squares 
  a=1 # ridge
  cv.fit           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.ri[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.ri[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)   
  
  cat(sprintf("m=%3.f| Rsq.test.la=%.2f,  Rsq.test.ri=%.2f| Rsq.train.la=%.2f,  Rsq.train.ri=%.2f| \n", m,  Rsq.test.la[m], Rsq.test.ri[m],  Rsq.train.la[m], Rsq.train.ri[m]))
  
}


boxplot(Rsq.test.en, Rsq.test.rf, Rsq.test.la,  Rsq.test.ri,
        main = "Boxplots of R-Sq for test data",
        at = c(1,2,3,4),
        names = c("Elastic Net", "R.Forest", "Lasso", "Ridge"),
        col = c("orange","red","royalblue2","green")
)


boxplot(Rsq.train.en, Rsq.train.rf, Rsq.train.la,  Rsq.train.ri,
        main = "Boxplots of R-Sq for train data",
        at = c(1,2,3,4),
        names = c("Elastic Net", "R.Forest", "Lasso", "Ridge"),
        col = c("orange","red","royalblue2","green")
)


1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
importance(rf)



p=34
bootstrapSamples =     5
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)         

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  # fit bs en
  a                =     0.5 # elastic-net
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit$beta)
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}
# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")



# fit rf to the whole data
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit en to the whole data
a=0.5 # elastic-net
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit              =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)


betaS.rf               =     data.frame(c(1:p), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.en               =     data.frame(c(1:p), as.vector(fit$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

b=0
cv.fit.la           =     cv.glmnet(X, y, alpha = b, nfolds = 10)
fit.la = glmnet(X, y, alpha = b, lambda = cv.fit.la$lambda.min)

betaS.la           =     data.frame(c(1:p), as.vector(fit.la$beta), 2*en.bs.sd)
colnames(betaS.la)     =     c( "feature", "value", "err")

c=1
cv.fit.ri           =     cv.glmnet(X, y, alpha = c, nfolds = 10)
fit.ri = glmnet(X, y, alpha = c, lambda = cv.fit.ri$lambda.min)
betaS.ri          =     data.frame(c(1:p), as.vector(fit.ri$beta), 2*en.bs.sd)
colnames(betaS.ri)     =     c( "feature", "value", "err")

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) 


enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

laPlot =  ggplot(betaS.la, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

riPlot =  ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

grid.arrange(rfPlot, laPlot, nrow = 2)
grid.arrange(riPlot, enPlot, nrow = 2)
#order of features that have a strong impact and response

# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.la$feature     =  factor(betaS.la$feature, levels = betaS.la$feature[order(betaS.la$value, decreasing = TRUE)])


rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

laPlot =  ggplot(betaS.la, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

grid.arrange(rfPlot, laPlot, nrow = 2)

#Part c #plot 10 fold cross validation curves
plot(cv.fit,sub = "CV for Elastic Net",cex.sub=1)
plot(cv.fit.la,sub = "CV for Lasso",cex.sub=1)
plot(cv.fit.ri,sub = "CV for Ridge",cex.sub=1)


cv.fit
cv.fit.la
cv.fit.ri


