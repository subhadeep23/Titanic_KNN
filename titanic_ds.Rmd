---
title: "KNN"
author: "Subhadeep Majumder"
date: "07-09-2021"
output:
  pdf_document: default
  
  
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
# Creating A KNN Model On The Titanic Dataset

![Sinking of the Titanic](titanic.jpg)

## Importing The Required Libraries
### Well, first we import all the libraries required to develop the model:
```{r echo=TRUE,message=FALSE,warning=FALSE}
library(gt)
library(class)
library(caret)
library(GGally)

```
## Importing the given Dataset
### Then we import the Given Dataset "titanic.csv" into the project:
```{r echo=TRUE}

titanic_ds<-read.csv('titanic_ds.csv',stringsAsFactors = FALSE)
```

## Pre-processing The Data to Increase its Quality
### In this stage of Data Analysis, we transform the structure and type of the data to make it suitable for the analysis that is to follow.
### First, we change the categorical data of Sex(Male,Female) to a Numerical Form Sex(1,0): 
```{r echo=TRUE}
titanic_ds$Sex<-ifelse(titanic_ds$Sex== "male" ,1,0)
```

### Second, we change the Categorical data of Embarked(Q,S,C) to Embarked(0,1,2):
```{r echo=TRUE}
 titanic_ds$Embarked[titanic_ds$Embarked=="Q"]<-0
 titanic_ds$Embarked[titanic_ds$Embarked=="S"]<-1
 titanic_ds$Embarked[titanic_ds$Embarked=="C"]<-2
```

### Now let's take a look at our imported dataset:
```{r echo=TRUE}
gt_preview(titanic_ds)

```
### We noticed that there are some columns that won't contribute to building the KNN model like the PassengerID and Name column. So we remove them and Preview the clean data:
```{r echo=TRUE}
titanic_clean<-titanic_ds[,c(2,3,5,6,7,8,10,12)]
gt_preview(titanic_clean)
```
### In the next step, we check if there are any missing values in the dataset:
```{r echo=TRUE}
sum(is.na(titanic_clean))

```

### As the attribute of interest based on which we need to classify the data is "Survived" where (Survived,Non-Survived):(1,0), we change the type of the Survival Data as Factors:
```{r echo=TRUE}
titanic_clean$Survived<-as.factor(titanic_clean$Survived)
str(titanic_clean)
```

### Before we change the type of the data, let's first locate the missing values and impute them:
```{r echo=TRUE}
sum(is.na(titanic_clean$Age)) #86 missing values in Age column
sum(is.na(titanic_clean$Pclass))
sum(is.na(titanic_clean$Sex))
sum(is.na(titanic_clean$SibSp))
sum(is.na(titanic_clean$Fare))# 1 missing value in fare column

```

### We impute the missing values in Age column with Median and that of Fare column with the Mode:
```{r echo=TRUE}
getmode <- function(mode_fare) {
   uniqv <- unique(mode_fare)
   uniqv[which.max(tabulate(match(mode_fare, uniqv)))]
}
mode_fare<-titanic_clean$Fare

titanic_clean$Age[is.na(titanic_clean$Age)]<-median(titanic_clean$Age,na.rm = TRUE)
titanic_clean$Fare[is.na(titanic_clean$Fare)]<-getmode(mode_fare)
getmode(mode_fare)
```

### Then we coerce all the columns in the dataset into numeric data type:
```{r echo=TRUE}
titanic_clean$Embarked<-as.numeric(titanic_clean$Embarked)
titanic_clean$Pclass<-as.numeric(titanic_clean$Pclass)
titanic_clean$SibSp<-as.numeric(titanic_clean$SibSp)
titanic_clean$Parch<-as.numeric(titanic_clean$Parch)
titanic_clean$Fare<-as.numeric(titanic_clean$Fare)
str(titanic_clean)
```

### We preview the final processed dataset again:
```{r echo=TRUE}
gt_preview(titanic_clean)
```

## Now we Start the Analysis Stage.

### Well first we normalize the data set using Z-scores so that there are no biases in the data due to difference in location and scale:

```{r echo=TRUE,warning=FALSE}
normlz <-function(x) {return(x-as.numeric(mean(x)))/as.numeric(sd(x)) }
titanic_norm<-as.data.frame(lapply(titanic_clean[,2:8], normlz))
summary(titanic_norm)
```

### This plot observes the correlation between the different attributes: 
```{r echo=TRUE,warning=FALSE}
ggpairs(titanic_clean,columns=2:8,mapping =aes(color=Survived))
```

### Then we split the Data frame into the Training Dataset and Testing Dataset:

```{r echo=TRUE}
titanic_train<-titanic_norm[1:293,1:7]
titanic_test<-titanic_norm[294:418,1:7]
titanic_train_labels<-as.array(titanic_clean[1:293,1]) 
titanic_test_labels<-as.array(titanic_clean[294:418,1])
```


```{r include=FALSE}
set.seed(456)

```
## Finding the Optimal Number Of Neighbours
### Here we create a loop to find the Percentage of Accuracy for each k from 1 to 25:

```{r echo=TRUE, warning=FALSE}

i=1
k.optm=1
for (i in 1:25){
 knn.pred <- knn(train=titanic_train, test=titanic_test, cl=titanic_train_labels, k=i)
 k.optm[i] <- 100 * sum(titanic_test_labels == knn.pred)/NROW(titanic_test_labels)
 k=i
 cat(k,'=',k.optm[i],'
')

}

```

### Then we draw the accuracy plot and determine the value of k for which we have the highest Accuracy:
```{r echo=TRUE, warning=FALSE}
acc_plot<-plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")

```


### Thus, from the graph we see that the model has the best accuracy of 72.8% for K=20 neighbours.


