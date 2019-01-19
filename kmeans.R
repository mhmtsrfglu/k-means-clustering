#Author: Mehmet Serefoglu
#Github-repo: https://github.com/mhmtsrfglu/k-means-clustering.git

library(caret)
library(ggfortify)
library(tidyverse)  # data manipulation
library(factoextra) # clustering algorithms & visualization
library(readr)

myData<-read.csv(file = "k-means-clustering/dataset-census.csv",sep =",") #read Data
myData<-myData[-c(1,2),] #remove first two row

#represent attributes by number label
myData$workclass = factor(myData$workclass,
                         levels = c('Private', 'Self-emp-not-inc', 'Self-emp-inc','Federal-gov','Local-gov','State-gov','Without-pay','Never-worked'),
                         labels = c(1, 2, 3, 4, 5, 6, 7, 8))

myData$education = factor(myData$education,
                          levels = c('Bachelors', 'Some-college', '11th','HS-grad','Prof-school','Assoc-acdm','Assoc-voc','9th','7th-8th','12th','Masters','1st-4th',
                                     '10th','Doctorate','5th-6th','Preschool'),
                          labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16))

myData$marital.status = factor(myData$marital.status,
                          levels = c('Married-civ-spouse', 'Divorced', 'Never-married','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'),
                          labels = c(1, 2, 3, 4, 5, 6, 7))

myData$occupation = factor(myData$occupation,
                          levels = c('Tech-support', 'Craft-repair', 'Other-service','Sales','Exec-managerial','Prof-specialty','Handlers-cleaners','Machine-op-inspct','Adm-clerical','Farming-fishing','Transport-moving','Priv-house-serv',
                                     'Protective-serv','Armed-Forces'),
                          labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))

myData$relationship = factor(myData$relationship,
                           levels = c('Wife', 'Own-child', 'Husband','Not-in-family','Other-relative','Unmarried'),
                           labels = c(1, 2, 3, 4, 5, 6))
myData$race = factor(myData$race,
                             levels = c('White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other','Black'),
                             labels = c(1, 2, 3, 4, 5))
myData$sex = factor(myData$sex,
                     levels = c('Female', 'Male'),
                     labels = c(1, 2))

#convert columns factor to numeric
myData$age<-as.numeric(myData$age)
myData$workclass<-as.numeric(myData$workclass)
myData$fnlwgt<-as.numeric(myData$fnlwgt)
myData$education<-as.numeric(myData$education)
myData$education.num<-as.numeric(myData$education.num)
myData$marital.status<-as.numeric(myData$marital.status)
myData$occupation<-as.numeric(myData$occupation)
myData$race<-as.numeric(myData$race)
myData$relationship<-as.numeric(myData$relationship)
myData$sex<-as.numeric(myData$sex)
myData$capital.gain<-as.numeric(myData$capital.gain)
myData$capital.loss<-as.numeric(myData$capital.loss)
myData$hours.per.week<-as.numeric(myData$hours.per.week)

cleanData <- myData[complete.cases(myData), ] #remove NA rows
rangeStd <- function(x) {(x-min(x))/(max(x)-min(x))} #standardize function 
stdData <- as.data.frame(apply(cleanData,2, rangeStd)) #normalize cleaned data
summary(stdData)

#remove redundant columns, there are too many zeros when we normalize data so I removed these columns
stdData$capital.gain<-NULL
stdData$capital.loss<-NULL
stdData$marital.status<-NULL
stdData$workclass<-NULL
stdData$race<-NULL

summary(stdData)

analys.pca <- princomp(stdData) #application of PCA

print(analys.pca)

summary(analys.pca)

transformed <- predict(analys.pca, stdData) #application of PCA attributes
summary(transformed)

#k-means clustering application.
set.seed(123)

kmeans1<-kmeans(transformed,2)
cleanData$cluster.kvalue.2<-kmeans1$cluster
autoplot(kmeans1, data = transformed)

kmeans2<-kmeans(transformed,3)
cleanData$cluster.kvalue.3<-kmeans2$cluster
autoplot(kmeans2, data = transformed)

kmeans3<-kmeans(transformed,4)
cleanData$cluster.kvalue.4<-kmeans3$cluster
autoplot(kmeans3, data = transformed)

kmeans4<-kmeans(transformed,5)
cleanData$cluster.kvalue.5<-kmeans4$cluster
autoplot(kmeans4, data = transformed)

kmeans5<-kmeans(transformed,6)
cleanData$cluster.kvalue.6<-kmeans5$cluster
autoplot(kmeans5, data = transformed)

#Elbow Method for finding the optimal number of clusters
# Compute and plot wss for k = 1 to k = 6
k.max <- 1:6

wss <- sapply(k.max, 
              function(k){kmeans(stdData, k, nstart=25,iter.max = 15 )$tot.withinss})
wss
plot(k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


#rename attributes with orginal names
cleanData$workclass = factor(cleanData$workclass,
                          levels = c(1, 2, 3, 4, 5, 6, 7, 8),
                          labels = c('Private', 'Self-emp-not-inc', 'Self-emp-inc','Federal-gov','Local-gov','State-gov','Without-pay','Never-worked'))
cleanData$education = factor(cleanData$education,
                          levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16) ,
                          labels = c('Bachelors', 'Some-college', '11th','HS-grad','Prof-school','Assoc-acdm','Assoc-voc','9th','7th-8th','12th','Masters','1st-4th',
                                     '10th','Doctorate','5th-6th','Preschool'))

cleanData$marital.status = factor(cleanData$marital.status,
                               levels = c(1, 2, 3, 4, 5, 6, 7),
                               labels = c('Married-civ-spouse', 'Divorced', 'Never-married','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'))

cleanData$occupation = factor(cleanData$occupation,
                           levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                           labels = c('Tech-support', 'Craft-repair', 'Other-service','Sales','Exec-managerial','Prof-specialty','Handlers-cleaners','Machine-op-inspct','Adm-clerical','Farming-fishing','Transport-moving','Priv-house-serv',
                                      'Protective-serv','Armed-Forces'))

cleanData$relationship = factor(cleanData$relationship,
                             levels = c(1, 2, 3, 4, 5, 6),
                             labels = c('Wife', 'Own-child', 'Husband','Not-in-family','Other-relative','Unmarried'))
cleanData$race = factor(cleanData$race,
                     levels = c(1, 2, 3, 4, 5),
                     labels = c('White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other','Black'))
cleanData$sex = factor(cleanData$sex,
                    levels = c(1, 2),
                    labels = c('Female', 'Male'))

write.csv(cleanData,file = "dataset-census-with-clusters.csv") #writing datas into file with clusters



