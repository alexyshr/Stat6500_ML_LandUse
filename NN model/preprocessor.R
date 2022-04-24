# preprocessor fetches the data and separate the different coarseness levels

## 1- load the data files
trainData <- read.csv("training.csv")
testData <- read.csv("testing.csv")

## 2- check for missing or NaN values
#check for missing data in both data sets
sum(is.na(trainData)); sum(is.na(testData))
# check for NaN/infinite data, since all attributes are numerical, we check with is.infinite()
sum(!sapply(trainData[,!1], is.finite)); sum(!sapply(testData[,!1], is.finite))
#no missing or NaN data

## 3- Change to class to categorical, and normalize variables
# convert class from "char" into a factor
trainData$class <- as.factor(trainData$class)
testData$class <- as.factor(testData$class)

#normalize variables 
norm_vals <- function(x) {(x-min(x))/(max(x)-min(x))}
trainData[2:148] <- lapply(trainData[2:148], norm_vals)
testData[2:148] <- lapply(testData[2:148], norm_vals)

# create a list of variables to be deleted 
# omitted variables: Brdindx, Shpindx, compact, round, Mean_G, Mean_R, Mean_NIR, SD_R, SD_NIR
partialList <- c(2,4,6,7,8,9,10,12,13) #deletions will be in the loop

## 4- separate the data into 7 coarseness levels (2 sets: full and partial (EDA result))
#create a list of dataframes
trainingList <- list(); testingList <- list()
parTrainingList<- list(); parTestingList <- list()

#create new dataframe for each coarseness level

for( i in 1:7){
  #first column index of the selected coarseness
  begin = (i-1)*21+2
  #last column index
  ending = begin+20
  # add to the list
  trainingList[[i]] <- trainData[,c(1,begin:ending)]
  #name the coarseness level
  names(trainingList)[i] <-  paste("trainingData", i*20, sep = "")
  testingList[[i]] <- testData[,c(1,begin:ending)]
  #name the coarseness level
  names(testingList)[i] <-  paste("testingData", i*20, sep = "")
  #Create and name the partial datasets
  parTrainingList[[i]] <- trainingList[[i]][,-partialList]
  names(parTrainingList)[i] <-  paste("parTrainingData", i*20, sep = "")
  parTestingList[[i]] <- testingList[[i]][,-partialList]
  names(parTestingList)[i] <-  paste("parTestingData", i*20, sep = "")
}

## 5- clean the work space
rm(begin, ending, i, trainData, testData, partialList, norm_vals)
# Now all variables in the work space will called into "NN_Processor.Rmd"