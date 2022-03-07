#STAT 6500 project: Urban and Land Cover
#
# explanatory data analysis
#
#Part 1: Loading and cleaning data
# notes:
# 1-there are 21 measurements
# 2- these measurements are repeated 7 times for each coarsness level
# 3- Resulting in 147 attributes in total

library(corrplot)

#Load and view Training data
trainData <- read.csv("training.csv")
summary(trainData)

#check for missing data 
sum(is.na(trainData))
# check for NaN/infinte data, since all attributes are numerical, we check with is.infinite()
sum(!sapply(trainData[,!1], is.finite))
#no missing or NaN data

# convert class from "char" into a factor
trainData$class <- as.factor(trainData$class)

#Pre-processing complete, data is clean and ready

## Part 2: Investigation 

# A first start would be to investigate each coarseness level
#create a list of dataframes
trainingList <- list()

#create new dataframe for each coarseness level

for( i in 1:7){
  #set the name to data + coarseness value
  name<- paste("data", i*20, sep = "")
  #first column index of the selected coarseness
  begin = (i-1)*21+2
  #last column index
  ending = begin+20
  #assign the first column and the columns between first and last
  assign(name, trainData[,c(1,begin:ending)])
  # add to the list
  trainingList[[i]] <- trainData[,c(1,begin:ending)]
}

# check collinearity among repeated variables

corList <- list()
#looping through all repeated variables
for(i in 2:22){
  #creating a dataframe with coarseness 20, and attribute i
  df <- trainingList[[1]][,i]
  for(j in 2:7){
    #adding the same attribute from a different coarseness level
    df <- cbind(df,trainingList[[j]][,i] )
    colnames(df)[j]<- paste(j*20)
  }
  colnames(df)[1]<- "20"
  #adding a correlation matrix to the list
  corList[[i]] <-  cor(df)
  #plotting 21 correlation matrices
  corrplot.mixed(corList[[i]], upper = 'circle', lower = 'number', title = 
                   colnames(data20[i]), line = -4)
}

# notes, from the correlation plots:
# 1- in all coarseness cases, 120 and 140 are very highly correlated
# 2- across all attributes, correlation among consecutive coarseness level is highly correlated
# 3- Bright, Mean_G, Mean_R, Mean_NIR, and NVDI are extremely correlated across all coarseness levels

#hypothesis, perhaps modelling the data using a single coarseness level is sufficient
# or using coarseness levels 20 and 140 as they are they least correlated

# checking the correlation among different attributes within the same coarseness
corList2 <- list()
for(i in 1:7){
  corList2[[i]] <- cor(trainingList[[i]][,2:22])

  corrplot.mixed(corList2[[i]], upper = 'circle', lower = 'number', title = paste("Coarseness Level", i*20, sep=" "))
}

# notes:
#1- brdindx and shpindx are highly correlated across all coarseness levels
# 2- bright is highly correlated with mean_G, mean_R, and mean_NIR
#3- SD_G, SD_R, SD_NIR are highly correlated among each other
# 4- rect is inversely correlated with shpindx, compact, round, and brdindx

#conclusion:
# 1- classification can be attempted with only one coarsensss level, or levels 20 and 140
# 2- Brdindx, Shpindx, compact, and round can be omitted from the model since rect is highly correlated with all of them
# 3- Mean_G, Mean_R, and Mean_NIR can be omitted from the model since bright is highly correlated with all of them
# 4- SD_G, SD_R, or SD_NIR should be used in the model; whichever produces the best results