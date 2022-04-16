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

## 3- Change to categorical data
# convert class from "char" into a factor
trainData$class <- as.factor(trainData$class)
testData$class <- as.factor(testData$class)


## 4- separate the data into 7 coarseness levels
#create a list of dataframes
trainingList <- list(); testingList <- list(); name <- list()

#create new dataframe for each coarseness level

for( i in 1:7){
  #set the name to data + coarseness value
  name[1] <- paste("trainingData", i*20, sep = "")
  name[2] <- paste("testingData", i*20, sep = "")
  #first column index of the selected coarseness
  begin = (i-1)*21+2
  #last column index
  ending = begin+20
  #assign the first column and the columns between first and last
  assign(name[[1]], trainData[,c(1,begin:ending)])
  assign(name[[2]], testData[,c(1,begin:ending)])
  # add to the list
  trainingList[[i]] <- trainData[,c(1,begin:ending)]
  testingList[[i]] <- testData[,c(1,begin:ending)]
}

## 5- clean the work space
rm(name); rm(begin); rm(ending); rm(i); rm(trainData); rm(testData)

# Now all variables in the work space will called into "NN_Processor.Rmd"