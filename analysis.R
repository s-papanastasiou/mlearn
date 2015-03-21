#Processing in R
#Import data as a data frame

#The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

#how you used cross validation, 
#what you think the expected out of sample error is, 
#and why you made the choices you did. 
#You will also use your prediction model to predict 20 different test cases.


#Before loading the csv file we replace "#DIV/0!" with NA. These are obviously values that indicate an error and thus missing data
rdata <- read.csv("pml-training.csv", header=TRUE, sep=",")

#Interesting observations:
#---------------------------------#
#rdata$amplitude_yaw_forearm has a 322 0s and 19000 NAs out of 19622 entries. It is useless and we delete it (139)
which(colnames(rdata)=="amplitude_yaw_forearm")
#rdata$skewness_yaw_forearm has 19622 entries all NA. It is useless and we delete it (130)
which(colnames(rdata)=="skewness_yaw_forearm")
#rdata$kurtosis_yaw_forearm has 19622 entries all NA. It is useless and we delete it (127)
which(colnames(rdata)=="kurtosis_yaw_forearm")
#rdata$amplitude_yaw_dumbbell has 19221 NAs and rest (out of 19622) is 0 (101)
which(colnames(rdata)=="amplitude_yaw_dumbbell")
#rdata$skewness_yaw_dumbbell is all NAs and we delete it (92)
which(colnames(rdata)=="skewness_yaw_dumbbell")
#rdata$kurtosis_yaw_dumbbell is all NAs and we delete it (89)
which(colnames(rdata)=="kurtosis_yaw_dumbbell")
#rdata$amplitude_yaw_belt has 19226 NA and rest is 0 (26)
which(colnames(rdata)=="amplitude_yaw_belt")
#rdata$skewness_yaw_belt has all NAs (17)
which(colnames(rdata)=="skewness_yaw_belt")
#rdata$kurtosis_yaw_belt has all NAs (14)
which(colnames(rdata)=="kurtosis_yaw_belt")

#State the above columns to delete
cols_delete = c("amplitude_yaw_forearm", "skewness_yaw_forearm", "kurtosis_yaw_forearm", "amplitude_yaw_dumbbell", "skewness_yaw_dumbbell", "kurtosis_yaw_dumbbell", "amplitude_yaw_belt", "skewness_yaw_belt","kurtosis_yaw_belt")
rdata_clean <- rdata[, !(colnames(rdata) %in% cols_delete)]

#Remove data with excessively missing values. These are unrepresentative of anything, i.e. there are too few to be a meaningful sample
rdata_clean_m <- rdata_clean[,colSums(is.na(rdata_clean))< 19000]
#The above reduces the predictors which go down from 151 to 60
#Column cvtd_timestamp is not considered a time series variable, but is instead considered a factor. There may be something there, i.e. potential for exploitation but we ignore for now, due to time restrictions
# To find type of columns you can do sapply(my.data, typeof) and verify all is as expected

#We will split our data into separate training and testing sets. The provided "testing" set used later on is not enough, we need to make sure that we do not overfit and we gain some confidence in the generality of the model.
library(caret)
# Do a 60-40 split (rule of thumb)
inTrain <- createDataPartition(y=rdata_clean_m$classe, p=0.60, list=FALSE)
training <- rdata_clean_m[inTrain,]
testing <- rdata_clean_m[-inTrain,]
#Delete old and useless variables to save memory
rm(rdata)
rm(rdata_clean)
rm(rdata_clean_m)
#Can do some further manual analysis by discovering correlating variables
#Find only numeric columns
#nums <- sapply(training, is.numeric)
#numFeatures <- training[,nums]
#Find columns to remove
#removeCols <- findCorrelation(cov(numFeatures))
#trueTrain <- training[-removeCols,]

#We will not do the above but instead use the PCA option to do principal components analysis
#and combine correlating variables into fewer components
#The following trainControl does k-fold cross validation with k=5. We try to ensure that we do not overfit the model and get respectable computation requirements.
ctrl <-  trainControl(method="repeatedcv", number=5) #Cross validation with k=5
#Reduce the number of trees produced due to hardware limitations and produce a random forest which will hopefully create a good classification
modelFit <- train(classe~., data = training, method="rf", ntree=100, trControl=ctrl, preProcess="pca",prox=TRUE)

#Test how well we did on the testing data
predictions <- predict(modelFit, newdata=testing)
confusionMatrix(predictions, testing$classe)

#Confusion Matrix and Statistics
#
#          Reference
#Prediction    A    B    C    D    E
#         A 2217   32    1    0    0
#         B   10 1475   33    0    0
#         C    5   11 1329   34    0
#         D    0    0    5 1244   15
#         E    0    0    0    8 1427
#
#Overall Statistics
                                          
#               Accuracy : 0.9804          
#                 95% CI : (0.9771, 0.9833)
#    No Information Rate : 0.2845          
#    P-Value [Acc > NIR] : < 2.2e-16       
#                                          
#                  Kappa : 0.9752          
# Mcnemar's Test P-Value : NA              

#Statistics by Class:

#                     Class: A Class: B Class: C Class: D Class: E
#Sensitivity            0.9933   0.9717   0.9715   0.9673   0.9896
#Specificity            0.9941   0.9932   0.9923   0.9970   0.9988
#Pos Pred Value         0.9853   0.9717   0.9637   0.9842   0.9944
#Neg Pred Value         0.9973   0.9932   0.9940   0.9936   0.9977
#Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
#Detection Rate         0.2826   0.1880   0.1694   0.1586   0.1819
#Detection Prevalence   0.2868   0.1935   0.1758   0.1611   0.1829
#Balanced Accuracy      0.9937   0.9824   0.9819   0.9821   0.9942

#Load real test data for the exercise
tdata <- read.csv("pml-testing.csv", header=TRUE, sep=",")
predictions <- predict(modelFit, newdata=tdata)

