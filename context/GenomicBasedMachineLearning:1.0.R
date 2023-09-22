#!/usr/bin/env Rscript

#### system specifications ####
# Architecture:                    x86_64
# CPU op-mode(s):                  32-bit, 64-bit
# CPU(s):                          8
# Thread(s) per core:              2
# Core(s) per socket:              4
# Model name:                      Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz
# R:                               4.3.0
# RStudio:                         2022.02.3 Build 492

# install libssl-dev and libcurl4-openssl-dev Ubuntu 20.04 libraries required for the benchmarkme R library
## sudo apt-get update && apt-get install -y libssl-dev
## sudo apt-get update && apt-get install -y libcurl4-openssl-dev

# set process limits (e.g. "--ulimit stack=100000000" as a Docker argument or "ulimit -s 100000000" through a Linux shell) for high performance computing users to prevent "Error: segfault from C stack overflow"

# set the maximum vector heap size (e.g. "-e R_MAX_VSIZE=900G" as a Docker argument or "touch ~/.Renviron | echo R_MAX_VSIZE=900GB > ~/.Renviron" through a Linux shell) for high performance computing users according to available memory ("free -g -h -t" through Linux shell) to prevent "Error: protect(): protection stack overflow"

# skip lines related to installation of libraries because there are supposed to be already installed
skip_instalation <- scan(what="character", quiet = TRUE)
# install libraries
install.packages("remotes") # version 2.4.2
require(remotes)
install_version("optparse", version = "1.7.3", repos = "https://cloud.r-project.org")
install_version("caret", version = "6.0-94", repos = "https://cloud.r-project.org")
install_version("doParallel", version = "1.0.17", repos = "https://cloud.r-project.org")
install_version("benchmarkme", version = "1.0.8", repos = "https://cloud.r-project.org")
install_version("data.table", version = "1.14.8", repos = "https://cloud.r-project.org")
install_version("dplyr", version = "1.1.2", repos = "https://cloud.r-project.org")
install_version("janitor", version = "2.1.0", repos = "https://cloud.r-project.org")
install_version("randomForest", version = "4.7-1.1", repos = "https://cloud.r-project.org")
install_version("gbm", version = "2.1.8.1", repos = "https://cloud.r-project.org")
install_version("ranger", version = "0.15.1", repos = "https://cloud.r-project.org")
install_version("xgboost", version = "1.7.5.1", repos = "https://cloud.r-project.org")
install_version("MLeval", version = "0.3", repos = "https://cloud.r-project.org")
install_version("MLmetrics", version = "1.1.1", repos = "https://cloud.r-project.org")

# load packages avoiding warning messages
suppressPackageStartupMessages(library(optparse)) # version 1.7.3
suppressPackageStartupMessages(library(caret)) # version 6.0-92
suppressPackageStartupMessages(library(doParallel)) # version 1.0.17
suppressPackageStartupMessages(library(benchmarkme)) # version 1.0.8
suppressPackageStartupMessages(library(data.table)) # version 1.14.2
suppressPackageStartupMessages(library(dplyr)) # version 1.0.9
suppressPackageStartupMessages(library(janitor)) # version 2.1.0
suppressPackageStartupMessages(library(randomForest)) # version 4.7-1.1
suppressPackageStartupMessages(library(gbm)) # version 2.1.8.1
suppressPackageStartupMessages(library(ranger)) # version 0.14.1
suppressPackageStartupMessages(library(xgboost)) # version 1.6.0.1
suppressPackageStartupMessages(library(MLeval)) # version 0.3
suppressPackageStartupMessages(library(MLmetrics)) # version 1.1.1

# clean environment
rm(list=ls())

# clean graphical device
graphics.off()

# set a limit on the number of nested expressions together with maximum size of the pointer protection stack ("Rscript --max-ppsize=500000") to prevent "Error: protect(): protection stack overflow"
options(expressions=500000)

# use English language
invisible(capture.output(Sys.setlocale("LC_TIME", "C")))

# keep in mind start time
start.time <- Sys.time()

# keep in mind start time as human readable
start.time.readable <- format(Sys.time(), "%X %a %b %d %Y")

# identify available CPUs (benchmarkme)
allCPUs <- get_cpu()$no_of_cores

# create a opt list (optparse) that contains all the arguments sorted by order of appearance in option_list and which can be called by their names (e.g. opt$input)
option_list = list(
  make_option(c("-g", "--goal"), type="character", default=NULL, 
              help="Perform prediction and estimate accuracy from training and testing datasets through the holdout method combined with the repeated k-fold cross-validation method if the tested phenotypes are known (i.e. 'research') or perform prediction and estimate accuracy from the training dataset through the holdout method combined with the repeated k-fold cross-validation method if the tested phenotypes are unknown (i.e. 'surveillance'). [MANDATORY]", metavar="character"),
  make_option(c("-m", "--mutations"), type="character", default=NULL, 
              help="Input mutation file with an absolute or relative path (tab-separated values). First column: mutations (header: whatever). Other columns: profiles of binary (e.g. presence/absence of genes or kmers) or categorical (e.g. profiles of alleles or variants) mutations for each sample (header: sample identifiers identical to the phenotype input file). [MANDATORY]", metavar="character"),
  make_option(c("-i", "--phenotype"), type="character", default=NULL, 
              help="Input phenotype file with an absolute or relative path (tab-separated values). First column: sample identifiers identical to the mutation input file (header: 'sample'). Second column: categorical phenotype (header: 'phenotype'). Third column: 'training' or 'testing' dataset (header: 'dataset'). [MANDATORY]", metavar="character"),
  make_option(c("-c", "--cpu"), type="integer", default=allCPUs, 
              help="Number of central processing units (CPUs). [OPTIONAL, default = all]", metavar="integer"),
  make_option(c("-d", "--dataset"), type="character", default="random", 
              help="Perform random (i.e. 'random') or manual (i.e. 'manual') splitting of training and testing datasets dedicated to the accuracy estimation through the holdout method combined with the repeated k-fold cross-validation method. [OPTIONAL, default = %default]", metavar="character"),
  make_option(c("-s", "--splitting"), type="numeric", default=80, 
              help="Proportion (%) defining the amount of training samples during random splitting of training and testing datasets through the holdout method (e.g. 50, 60, 70, 80 or 90%). [OPTIONAL, default = %default]", metavar="numeric"),
  make_option(c("-v", "--variances"), type="logical", default=TRUE, 
              help="Removal of near zero-variance descriptors from the training dataset. [OPTIONAL, default = %default]", metavar="logical"),
  make_option(c("-r", "--ratio"), type="numeric", default=19, 
              help="Frequency of the most prevalent value divided by the frequency of the second most frequent value below which descriptors will be considered as near zero-variance descriptors and discarded from the training dataset (a.k.a. frequency ratio: freqCut argument of the nearZeroVar function from the caret R library). [OPTIONAL, default = %default]", metavar="numeric"),
  make_option(c("-u", "--unique"), type="numeric", default=10, 
              help="Hundred times the number of unique values divided by the total number of samples above which descriptors will be considered as near zero-variance descriptors and discarded from the training dataset (a.k.a. percent of unique values: uniqueCut argument of the nearZeroVar function from the caret R library). [OPTIONAL, default = %default]", metavar="numeric"),
  make_option(c("-k", "--fold"), type="numeric", default=5, 
              help="Value defining k-1 groups of samples used to train against one group of validation through the k-fold cross-validation method (e.g. 2.0‑, 2.5-, 3.3-, 5.0- or 10-fold cross‑validations). [OPTIONAL, default = %default]", metavar="numeric"),
  make_option(c("-e", "--repetition"), type="integer", default=10,  
              help="Number of repetition of the k-fold cross-validation method. [OPTIONAL, default = %default]", metavar="integer"),
  make_option(c("-t", "--tuning"), type="integer", default=10, 
              help="Maximal value of the main parameter to considered for the model tuning. Ten incremental tenth of the maximal value of the main parameter will be used for the model training. [OPTIONAL, default = %default]", metavar="integer"),
  make_option(c("-f", "--fit"), type="character", default="xgb", 
              help="Perform boosted logistic regression (i.e. 'blr'), extremely randomized trees (i.e. 'ert'), random forest (i.e. 'rf'), stochastic gradient boosting (i.e. 'sgb'), support vector machine (i.e. 'svm') or extreme gradient boosting (i.e. 'xgb') models. [OPTIONAL, default = %default]", metavar="character"),
  make_option(c("-o", "--prefix"), type="character", default="output_", 
              help="Absolute or relative output path with or without output file prefix. [OPTIONAL, default = %default]", metavar="character"),
  make_option(c("-b", "--backup"), type="logical", default=FALSE, 
              help="Save an external representation of R objects (i.e. saved_data.RData) and a short-cut of the current workspace (i.e. saved_images.RData). [OPTIONAL, default = %default]", metavar="logical")
); 

# parse the opt list and get opt list as arguments (optparse)
opt_parser <- OptionParser(option_list=option_list);
opt <- suppressWarnings(parse_args(opt_parser));

# prepare a global message for help
help1 <- "Help: Rscript GenomicBasedMachineLearning:1.0.R -h"
help2 <- "Help: Rscript GenomicBasedMachineLearning:1.0.R --help"

# management of mandatory arguments
## arguments -g/--goal, -m/--mutations and -i/--phenotype
if (((is.null(opt$goal)) || (is.null(opt$mutations)) || (is.null(opt$phenotype))) == TRUE){
  cat("\n", 'Version: 1.0', "\n")
  cat("\n", 'Please, provide at least one goal (i.e. mandatory argument -g) and two input files (i.e. mandatory arguments -m and -i) and potentially other optional arguments:', "\n")
  cat("\n", 'Example 1: Rscript --max-ppsize=500000 GenomicBasedMachineLearning:1.0.R -g research -m Alleles-100-samples.tsv -i PhenotypeDataset-100-samples.tsv', "\n")
  cat("\n", 'Example 2: Rscript --max-ppsize=500000 GenomicBasedMachineLearning:1.0.R -g research -m Genes-100-samples.tsv -i PhenotypeDataset-100-samples.tsv -c 6 -d manual -v FALSE -f svm -o MyOutput_', "\n")
  cat("\n", 'Example 3: Rscript --max-ppsize=500000 GenomicBasedMachineLearning:1.0.R --goal research --mutations Alleles-100-samples.tsv --phenotype PhenotypeDataset-100-samples.tsv --cpu 6 --dataset manual --variances FALSE --fit svm --prefix MyOutput_', "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -g/--goal
if (opt$goal != "research" & opt$goal != "surveillance") {
  cat("\n", "The argument -g/--goal must be a character (i.e. 'research' or 'surveillance')", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}

# management of optional arguments
## argument -c/--cpu
if (!grepl("\\D",opt$cpu) == FALSE) {
  cat("\n", "The argument -c/--cpu (number of central processing units) must be an integer (NB: a number with a decimal will return the rounded down integer)", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
if (((opt$cpu <= 0) || (opt$cpu > allCPUs)) == TRUE) {
  cat("\n", "The argument -c/--cpu (number of central processing units) must be an integer (NB: a number with a decimal will return the rounded down integer) between 1 and the maximum of available CPUs (i.e.", allCPUs, "in your case)", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -d/--dataframe
if (opt$dataset != "random" & opt$dataset != "manual") {
  cat("\n", "The argument -d/--dataframe must be a character (i.e. 'ramdom' or 'manual')", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -s/--splitting
if (is.numeric(opt$splitting) == FALSE) {
  cat("\n", "The argument -s/--splitting must be a positive whole number or a positive number with a decimal between 0 and 100", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
if (((opt$splitting <= 0) || (opt$splitting >= 100)) == TRUE) {
  cat("\n", "The argument -s/--splitting must be a positive whole number or a positive number with a decimal between 0 and 100", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -v/--variances
if (((isTRUE(opt$variances)) || (isFALSE(opt$variances))) == FALSE){
  cat("\n", "The argument -v/--variances must be logical (TRUE or FALSE)", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -r/--ratio
if (is.numeric(opt$ratio) == FALSE) {
  cat("\n", "The argument -r/--ratio must be a positive whole number or a positive number with a decimal", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
if ((opt$ratio <= 0) == TRUE) {
  cat("\n", "The argument -r/--ratio must be a positive whole number or a positive number with a decimal", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -u/--unique
if (is.numeric(opt$unique) == FALSE) {
  cat("\n", "The argument -u/--unique must be a positive whole number or a positive number with a decimal", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
if ((opt$unique <= 0) == TRUE) {
  cat("\n", "The argument -u/--unique must be a positive whole number or a positive number with a decimal", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -k/--fold
if (is.numeric(opt$fold) == FALSE) {
  cat("\n", "The argument -k/--fold must be a positive whole number or a positive number with a decimal between 0 and 100", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
if ((opt$fold < 2) == TRUE) {
  cat("\n", "The argument -k/--fold must be a positive whole number or a positive number with a decimal equal or higher than 2", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -e/--repetition
if (!grepl("\\D",opt$repetition) == FALSE) {
  cat("\n", "The argument -e/--repetition must be a positive integer (NB: a number with a decimal will return the rounded down integer)", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
if ((opt$repetition <= 0) == TRUE) {
  cat("\n", "The argument -e/--repetition must be a positive integer (NB: a number with a decimal will return the rounded down integer)", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -t/--tuning
if (!grepl("\\D",opt$tuning) == FALSE) {
  cat("\n", "The argument -t/--tuning must be a positive integer (NB: a number with a decimal will return the rounded down integer)", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
if ((opt$tuning <= 0) == TRUE) {
  cat("\n", "The argument -t/--tuning must be a positive integer (NB: a number with a decimal will return the rounded down integer)", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -f/--fit
if (opt$fit != "rf" & opt$fit != "svm" & opt$fit != "blr" & opt$fit != "sgb" & opt$fit != "xgb" & opt$fit != "ert") {
  cat("\n", "The argument -f/--fit must be a character (i.e. 'rf', 'svm', 'blr', 'sgb', 'xgb' or 'ert')", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## argument -b/--backup
if (((isTRUE(opt$backup)) || (isFALSE(opt$backup))) == FALSE){
  cat("\n", "The argument -r/--backup must be logical (TRUE or FALSE)", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}

# management of threads
## set desired CPUs and print
setDTthreads(threads = opt$cpu, restore_after_fork = TRUE, throttle = 1024)
cat("\n", "Used CPUs:", opt$cpu, ".... Please wait","\n")
## create a computing cluster and register
cluster <- makePSOCKcluster(opt$cpu)
registerDoParallel(cluster)

# step control
step1.time <- Sys.time()
step1.taken <- difftime(step1.time, start.time, units="secs")
cat(" Step 1/14 completed: checking of arguments approx. ", ceiling(step1.taken), " second(s)", "\n", sep = "")

# read the dataframe of mutations preventing read.table to add "X." as prefix of ID samples starting with number or special character
data_mutations <- read.table(opt$mutations, dec = ".", header=TRUE, sep = "\t", check.names = FALSE)

# step control
step2.time <- Sys.time()
step2.taken <- difftime(step2.time, step1.time, units="secs")
cat(" Step 2/14 completed: reading of input mutations approx. ", ceiling(step2.taken), " second(s)", "\n", sep = "")

# transform the dataframe of mutations
## replace by "mutation" the first variable
names(data_mutations)[1] <- "mutation"
## transpose dataframe (data.table)
trans_data_mutations <- transpose(data_mutations, keep.names = "sample", make.names = "mutation")

# step control
step3.time <- Sys.time()
step3.taken <- difftime(step3.time, step2.time, units="secs")
cat(" Step 3/14 completed: transposition of input mutations approx. ", ceiling(step3.taken), " second(s)", "\n", sep = "")

# add the phenotype into the dataframe of mutations
## read the phenotype/dataset file
PhenotypeDataset <- read.table(opt$phenotype, dec = ".", header=TRUE, sep = "\t")
## retrieve only the phenotype
CategoricalPhenotype <- PhenotypeDataset
CategoricalPhenotype <- subset(CategoricalPhenotype, select = -dataset)
## step control
step4.time <- Sys.time()
step4.taken <- difftime(step4.time, step3.time, units="secs")
cat(" Step 4/14 completed: reading of input phenotypes approx. ", ceiling(step4.taken), " second(s)", "\n", sep = "")
## test equality of sample identifiers from the input mutation and phenotype files
### sample identifiers from the input mutation file
IDs.mutations <- sort(trans_data_mutations$sample, decreasing=FALSE)
### sample identifiers from the input phenotype file
IDs.phenotypes <- sort(CategoricalPhenotype$sample, decreasing=FALSE)
### test equality
if (identical(IDs.mutations, IDs.phenotypes) == FALSE) {
  cat("\n", "The sample identifiers from the input mutation and phenotype files must be identical", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}
## joint (dplyr)
trans_data_mutations_catpheno <- suppressWarnings(left_join(trans_data_mutations, CategoricalPhenotype, by = "sample", keep = FALSE))
## keep in mind the amount of initial descriptors
Initial <- ncol(trans_data_mutations_catpheno) - 2

# step control
step5.time <- Sys.time()
step5.taken <- difftime(step5.time, step4.time, units="secs")
cat(" Step 5/14 completed: mapping of input mutations and phenotypes approx. ", ceiling(step5.taken), " second(s)", "\n", sep = "")

# transform all descriptors as factor and the phenotype as factor
## transform variables (mutations are considered as factor to manage presence/absence of genes or kmers, as well as profiles of alleles or variants)
col <- ncol(trans_data_mutations_catpheno)
trans_data_mutations_catpheno[2:(col-1)] <- lapply(trans_data_mutations_catpheno[2:(col-1)], FUN = function(y){as.factor(y)})
trans_data_mutations_catpheno$phenotype <- as.factor(trans_data_mutations_catpheno$phenotype)

# preparation of the training dataset
## make a copy of the training dataset
data <- trans_data_mutations_catpheno
## management of NAs
### replace missing data encoded "" with NA
data[ data == "" ] <- NA
### remove the descriptors harboring NAs (i.e. missing data)
data <- data[ , colSums(is.na(data)) == 0]
### keep in mind the amount of remaining descriptors
AfterNA <- ncol(data) - 2
## management of constant descriptors
### remove constant descriptors (janitor)
data <- remove_constant(data, na.rm = FALSE, quiet = TRUE)
### keep in mind the amount of remaining descriptors
AfterCD <- ncol(data) - 2

# step control
step6.time <- Sys.time()
step6.taken <- difftime(step6.time, step5.time, units="secs")
cat(" Step 6/14 completed: dataset preparation approx. ", ceiling(step6.taken), " second(s)", "\n", sep = "")

# perform dataset splitting according to random splitting or manual splitting
if (opt$dataset == "manual"){
  ## split according to the information provided by the user
  ### retrieve only the datasets related to training or testing with sample IDs
  TrainingTesting <- PhenotypeDataset
  TrainingTesting <- subset(TrainingTesting, select = -phenotype)
  ### retrieve only the sample identifiers of the training dataset
  IDtraining <- subset(TrainingTesting, dataset=="training")
  ### retrieve only the sample identifiers of the testing dataset
  IDtesting <- subset(TrainingTesting, dataset=="testing")
  ### get the training dataframe
  trainID <- data[data$sample %in% IDtraining$sample,]
  ### get the testing dataframe
  testID <- data[data$sample %in% IDtesting$sample,]
} else {
  ## split randomly into training and testing datasets to estimate accuracy through the holdout method
  ### calculate absolute proportion
  split_holdout_method <- opt$splitting / 100
  ### get randomly the training matrix (caret)
  inTrain <- createDataPartition(trans_data_mutations_catpheno$phenotype, p=split_holdout_method, list=FALSE)
  ### get the training dataframe
  trainID <- data[inTrain,]
  ### get the testing dataframe
  testID <- data[-inTrain,]
}

# step control
step7.time <- Sys.time()
step7.taken <- difftime(step7.time, step6.time, units="secs")
cat(" Step 7/14 completed: dataset splitting approx. ", ceiling(step7.taken), " second(s)", "\n", sep = "")

# remove the sample column because it must not be included into training or prediction
## remove the sample identifiers (ID) from the training dataframe
train <- subset(trainID, select = -sample)
## remove the sample identifiers (ID) from the testing dataframe
test <- subset(testID, select = -sample)

# control that at least 3 samples are present in each phenotype
## count training samples for each phenotype 
count.table <- as.data.frame(table(train$phenotype))
count.vector <- count.table$Freq
## test if any phenotype presents less than 3 samples
if (any(count.vector<3) == TRUE) {
  cat("\n", "Each phenotype of the training dataset has to harbor at least 3 samples", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}

# remove the phenotype column because it must not be included into preprocessing steps before training
## keep in mind the variable of interest because it is not a descriptors
trainingphenotype <- train$phenotype
## remove the variable of interest which is not a descriptors
train <- subset(train, select = -phenotype)

# make a copy of the training dataset
filteredtrain <- train

# step control
step8.time <- Sys.time()
step8.taken <- difftime(step8.time, step7.time, units="secs")
cat(" Step 8/14 completed: control of training sample amount approx. ", ceiling(step8.taken), " second(s)", "\n", sep = "")

# remove near zero-variance descriptors to prevent crash of susceptible machine learning models (optional)
if (isTRUE(opt$variances)){
  ## add near zero-variance descriptors in a interger vector
  nzv <- nearZeroVar(filteredtrain, freqCut = opt$ratio, uniqueCut = opt$unique, foreach = TRUE, allowParallel = TRUE)
  ## remove near zero-variance descriptors
  filteredtrain <- filteredtrain[, -nzv]
  ## keep in mind remaining descriptors
  AfterNZ <- ncol(filteredtrain)
} else {
  AfterNZ <- NA
}

# test if the filteredtrain dataframe is empty
if (is.null(dim(filteredtrain)) == TRUE) {
  cat("\n", "The ML model cannot be trained because all descriptors has been discarded from the training dataset", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}

# test if the filteredtrain dataframe harbors only one descriptor
if (ncol(filteredtrain) <= 1) {
  cat("\n", "The ML model cannot be trained because it remains only one descriptor from the training dataset", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}

# step control
step9.time <- Sys.time()
step9.taken <- difftime(step9.time, step8.time, units="secs")
cat(" Step 9/14 completed: removal of near zero-variance descriptors approx. ", ceiling(step9.taken), " second(s)", "\n", sep = "")

# prepare machine learning (ML) settings
# put back the variable of interest
filteredtrain <- cbind(filteredtrain, trainingphenotype)
# rename the variable of interest
names(filteredtrain)[names(filteredtrain) == 'trainingphenotype'] <- 'phenotype'
## drop potential useless levels
filteredtrain$phenotype <- droplevels(filteredtrain$phenotype)
## get the k value for the k-fold cross-validation
split_CV_method <- opt$fold
## set cross-validation adding repeated k-fold cross-validation during training (i.e. number of random variables collected at each split in normal equal square number columns)
setCrossValidation <- trainControl(
  method = "repeatedcv",
  number = split_CV_method, ## 5-fold CV per default (i.e. corresponding to default splitting of 80% for the training dataset: (5-1)/5*100)
  repeats = opt$repetition, ## repeated 10 times per default
  search = "random", ## randomly
  classProbs = TRUE, ## necessary for ROC
  summaryFunction = multiClassSummary, ## necessary for ROC
  savePredictions = TRUE,
  allowParallel = TRUE)

# train the model (caret library) using ROC metrics (MLmetrics library)
## ML based on Random Forest
if (opt$fit == "rf") {
  ### set the model grid
  setModelGrid <- expand.grid(.mtry = seq(from = opt$tuning/10, to = opt$tuning, by = (opt$tuning/10))) 
  ### run the train function to build ML model
  fitModel <- suppressWarnings(train(
    phenotype ~ .,
    data = filteredtrain,
    method = "rf",
    metric = "ROC",
    tuneGrid = setModelGrid,
    trControl = setCrossValidation,
    verbose = FALSE,
    quiet = TRUE))
## ML based on support vector machine
} else if (opt$fit == "svm") {
  ### set the model grid
  setModelGrid <- expand.grid(cost = seq(from = opt$tuning/10, to = opt$tuning, by = (opt$tuning/10)))
  ### run the train function to build ML model
  fitModel <- suppressWarnings(train(
    phenotype ~ .,
    data = filteredtrain,
    method = "svmLinear2",
    metric = "ROC",
    tuneGrid = setModelGrid,
    trControl = setCrossValidation,
    verbose = FALSE,
    quiet = TRUE))
## ML based on boosted logistic regression
} else if (opt$fit == "blr") {
  ### set the model grid
  setModelGrid <- expand.grid(nIter = seq(from = opt$tuning/10, to = opt$tuning, by = (opt$tuning/10))) # cannot start at 0
  ### run the train function to build ML model
  fitModel <- suppressWarnings(train(
    phenotype ~ .,
    data = filteredtrain,
    method = "LogitBoost",
    metric = "ROC",
    tuneGrid = setModelGrid,
    trControl = setCrossValidation,
    verbose = FALSE,
    quiet = TRUE))
## ML based on stochastic gradient boosting
} else if (opt$fit == "sgb") {
  ### set the model grid
  setModelGrid <- expand.grid(n.trees = 20,
                              interaction.depth = seq(from = opt$tuning/10, to = opt$tuning, by = (opt$tuning/10)),
                              shrinkage = 0.01,
                              n.minobsinnode = 3)
  ### run the train function to build ML model
  fitModel <- suppressWarnings(train(
    phenotype ~ .,
    data = filteredtrain,
    method = "gbm",
    metric = "ROC",
    tuneGrid = setModelGrid,
    trControl = setCrossValidation,
    verbose = FALSE))
## ML based on extreme gradient boosting
} else if (opt$fit == "xgb") {
  ## set the model grid
  setModelGrid <- expand.grid(nrounds = seq(from = opt$tuning/10, to = opt$tuning, by = (opt$tuning/10)),
                              eta = 0.3,
                              gamma = 0,
                              max_depth = 6, 
                              subsample = 1,
                              min_child_weight = 1,
                              colsample_bytree = 1)
  ### run the train function to build ML model
  fitModel <- suppressWarnings(train(
    phenotype ~ .,
    data = filteredtrain,
    method = "xgbTree",
    metric = "ROC",
    tuneGrid = setModelGrid,
    trControl = setCrossValidation,
    verbose = FALSE))
## ML based on extremely randomized trees
} else if (opt$fit == "ert") {
  ### set the model grid
  setModelGrid <- expand.grid(.mtry= seq(from = opt$tuning/10, to = opt$tuning, by = (opt$tuning/10)),
                              .splitrule = "extratrees",
                              .min.node.size = 1)
  ### run the train function to build ML model
  fitModel <- suppressWarnings(train(
    phenotype ~ .,
    data = filteredtrain,
    method = "ranger",
    metric = "ROC",
    tuneGrid = setModelGrid,
    trControl = setCrossValidation,
    verbose = FALSE))
} else {
  cat("\n", "Unexpected error", "\n")
  cat("\n", help1, "\n")
  cat("\n", help2, "\n", "\n")
  stop()
}

# step control
step10.time <- Sys.time()
step10.taken <- difftime(step10.time, step9.time, units="secs")
cat(" Step 10/14 completed: model training approx. ", ceiling(step10.taken), " second(s)", "\n", sep = "")

# output predictors
fitmodel.predictors <- as.data.frame(predictors(fitModel))
write.table(fitmodel.predictors, file = paste(opt$prefix, "model_predictors.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = FALSE, col.names = FALSE)

# output metrics_training_dataset_metrics files
## retrieve metrics across tuning parameters
metrics.parameters <- data.frame(fitModel$results)
## manage digits
metrics.parameters <- format(metrics.parameters, digits=3, nsmall=3)
## output in a tsv file
write.table(metrics.parameters, file = paste(opt$prefix, "metrics_training_dataset_tuning_parameters.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE)
## retrieve metrics across resampling
metrics.resampling <- data.frame(fitModel$resample)
## manage digits
metrics.resampling <- format(metrics.resampling, digits=3, nsmall=3)
## output in a tsv file
write.table(metrics.resampling, file = paste(opt$prefix, "metrics_training_dataset_resampling.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE)
## output calibration (cc), precision recall (pr), precision recall gain (prg), and receiver operating characteristic (roc) curves with the MLeval library
### prevent Rplots.pdf from being generated
if(!interactive()) pdf(NULL)
### open a pdf file
pdf(file = paste(opt$prefix, "metrics_training_dataset_curves.pdf", sep = ""))
### create output figures
invisible(evalm(fitModel, plots = c("cc", "pr","prg", "r"), silent = TRUE, dlinecol = "blue", percent = 95))
### close graphical device
invisible(dev.off())
## extract other performance metrics corresponding to the operating point with the maximal informedness based on the MLeval library
performance <- evalm(fitModel, plots = c("cc", "pr","prg", "r"), silent = TRUE, dlinecol = "blue", percent = 95)
performance.informedness <- as.data.frame(performance$optres)
colnames(performance.informedness) <- c("score", "CI")
write.table(performance.informedness, file = paste(opt$prefix, "metrics_training_dataset_model_performance.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = TRUE, col.names = NA)

# step control
step11.time <- Sys.time()
step11.taken <- difftime(step11.time, step10.time, units="secs")
cat(" Step 11/14 completed: writting of training output approx. ", ceiling(step11.taken), " second(s)", "\n", sep = "")

## perform prediction and estimate accuracy from training dataset
### perform prediction with the built model and without expected phenotype
col.filteredtrain <- which(colnames(filteredtrain)=="phenotype")
prediction.train <- suppressWarnings(predict(fitModel, filteredtrain[,-col.filteredtrain]))
### build confusion matrix and prediction statistics
accuracy.train <- suppressWarnings(confusionMatrix(prediction.train, filteredtrain$phenotype, mode = "prec_recall"))
### retrieve sample IDs from the train dataframe
sample.train <- trainID$sample
### retrieve the expected phenotype
expectation.train <- filteredtrain$phenotype
### combine predicted phenotype
results.train <- data.frame(sample.train, expectation.train, prediction.train)
### extract accuracy metrics
confusion.table.train <- as.table(accuracy.train, what = "table")
write.table(confusion.table.train, file = paste(opt$prefix, "metrics_training_dataset_confusion_matrix.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = TRUE, col.names = NA)
confusion.overall.train <- as.matrix(accuracy.train, what = "overall")
confusion.overall.train <- format(confusion.overall.train, digits=3, nsmall=3)
write.table(confusion.overall.train, file = paste(opt$prefix, "metrics_training_dataset_accuracy_overall.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = TRUE, col.names = FALSE)
confusion.classes.train <- as.matrix(accuracy.train, what = "classes")
confusion.classes.train <- format(confusion.classes.train, digits=3, nsmall=3)
write.table(confusion.classes.train, file = paste(opt$prefix, "metrics_training_dataset_accuracy_classes.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = TRUE, col.names = NA)
# output prediction
write.table(results.train, file = paste(opt$prefix, "prediction_training_dataset.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE)

# perform prediction and estimate accuracy from the testing dataset if the tested phenotypes are known (i.e. 'research') or perform prediction if the tested phenotypes are unknown (i.e. 'surveillance')
if (opt$goal == "research"){
  ## perform prediction with the built model and without expected phenotype
  col.test <- which(colnames(test)=="phenotype")
  prediction.test <- suppressWarnings(predict(fitModel, test[,-col.test]))
  ## build a confusion matrix and statistics of prediction
  accuracy.test <- suppressWarnings(confusionMatrix(prediction.test, test$phenotype, mode = "prec_recall"))
  ## retrieve sample IDs from the test dataframe
  sample.test <- testID$sample
  ## retrieve the expected phenotype
  expectation.test <- test$phenotype
  ## combine predicted phenotype
  results.test <- data.frame(sample.test, expectation.test, prediction.test)
  ## extract accuracy metrics
  confusion.table.test <- as.table(accuracy.test, what = "table")
  write.table(confusion.table.test, file = paste(opt$prefix, "metrics_testing_dataset_confusion_matrix.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = TRUE, col.names = NA)
  confusion.overall.test <- as.matrix(accuracy.test, what = "overall")
  confusion.overall.test <- format(confusion.overall.test, digits=3, nsmall=3)
  write.table(confusion.overall.test, file = paste(opt$prefix, "metrics_testing_dataset_accuracy_overall.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = TRUE, col.names = FALSE)
  confusion.classes.test <- as.matrix(accuracy.test, what = "classes")
  confusion.classes.test <- format(confusion.classes.test, digits=3, nsmall=3)
  write.table(confusion.classes.test, file = paste(opt$prefix, "metrics_testing_dataset_accuracy_classes.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = TRUE, col.names = NA)
  ## output prediction
  write.table(results.test, file = paste(opt$prefix, "prediction_testing_dataset.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE)
} else {
  ## perform prediction with unknown phenotype from the testing dataset
  col.test <- which(colnames(test)=="phenotype")
  prediction.test <- suppressWarnings(predict(fitModel, test[,-col]))
  ### retrieve sample IDs from the test dataframe
  sample.test <- testID$sample
  length(sample.test)
  ### retrieve phenotype
  phenotype.train <- filteredtrain$phenotype
  length(phenotype.train)
  ### retrieve unknown phenotype
  expectation.test <- rep("unknown", length(sample.test))
  length(expectation.test)
  ### combine predicted phenotype
  results.test <- data.frame(sample.test, expectation.test, prediction.test)
  ## output prediction
  write.table(results.test, file = paste(opt$prefix, "prediction_testing_dataset.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE)
}

# step control
step12.time <- Sys.time()
step12.taken <- difftime(step12.time, step11.time, units="secs")
cat(" Step 12/14 completed: model testing approx. ", ceiling(step12.taken), " second(s)", "\n", sep = "")

# output the input phenotype/dataset file
if (opt$goal == "research"){
  ## output the input phenotype/dataset file for the research goal
  ### retrieve sample IDs after splitting and training
  sample.train <- trainID$sample
  length(sample.train)
  sample.test <- testID$sample
  length(sample.test)
  ### retrieve phenotype
  phenotype.train <- filteredtrain$phenotype
  length(phenotype.train)
  phenotype.test <- test$phenotype
  length(phenotype.test)
  ### merge sample IDs and phenotypes from the training dataset
  summary_phenotypes.train <- data.frame(sample.train, phenotype.train)
  summary_phenotypes.train.final <- summary_phenotypes.train
  summary_phenotypes.train.final$dataset <- "training"
  dim(summary_phenotypes.train.final)
  ### merge sample IDs and phenotypes from the testing dataset
  summary_phenotypes.test <- data.frame(sample.test, phenotype.test)
  summary_phenotypes.test.final <- summary_phenotypes.test
  summary_phenotypes.test.final$dataset <- "testing"
  dim(summary_phenotypes.test.final)
  ### merge vertically dataframes
  colnames(summary_phenotypes.train.final) <- c("sample", "phenotype", "dataset")
  colnames(summary_phenotypes.test.final) <- c("sample", "phenotype", "dataset")
  summary_phenotypes_final <- rbind(summary_phenotypes.train.final, summary_phenotypes.test.final)
  summary_input_phenotypes <- summary_phenotypes_final[order(summary_phenotypes_final$sample), ]
  ### output
  write.table(summary_input_phenotypes, file = paste(opt$prefix, "summary_input_phenotypes.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE)
} else {
  ## output the input phenotype/dataset file for the surveillance goal
  ### retrieve sample IDs after splitting and training
  sample.train <- trainID$sample
  length(sample.train)
  sample.test <- testID$sample
  length(sample.test)
  ### retrieve phenotype
  phenotype.train <- filteredtrain$phenotype
  length(phenotype.train)
  phenotype.test <- rep("unknown", length(sample.test))
  length(phenotype.test)
  ### merge sample IDs and phenotypes from the training dataset
  summary_phenotypes.train <- data.frame(sample.train, phenotype.train)
  summary_phenotypes.train.final <- summary_phenotypes.train
  summary_phenotypes.train.final$dataset <- "training"
  dim(summary_phenotypes.train.final)
  ### merge sample IDs and phenotypes from the testing dataset
  summary_phenotypes.test <- data.frame(sample.test, phenotype.test)
  summary_phenotypes.test.final <- summary_phenotypes.test
  summary_phenotypes.test.final$dataset <- "testing"
  dim(summary_phenotypes.test.final)
  ### merge vertically dataframes
  colnames(summary_phenotypes.train.final) <- c("sample", "phenotype", "dataset")
  colnames(summary_phenotypes.test.final) <- c("sample", "phenotype", "dataset")
  summary_phenotypes_final <- rbind(summary_phenotypes.train.final, summary_phenotypes.test.final)
  summary_input_phenotypes <- summary_phenotypes_final[order(summary_phenotypes_final$sample), ]
  ### output
  write.table(summary_input_phenotypes, file = paste(opt$prefix, "summary_input_phenotypes.tsv", sep = ""), append = FALSE, quote = FALSE, sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE)
}

# step control
step13.time <- Sys.time()
step13.taken <- difftime(step13.time, step12.time, units="secs")
cat(" Step 13/14 completed: writting of testing output approx. ", ceiling(step13.taken), " second(s)", "\n", sep = "")

# output RData
if (isTRUE(opt$backup)){
  ## to load with load("output_saved_data.RData")
  save(list = ls(), file = paste(opt$prefix, "saved_data.RData", sep = ""))
  ## to load with load("output_saved_images.RData")
  save.image(file = paste(opt$prefix, "saved_images.RData", sep = ""))
}

# step control
step14.time <- Sys.time()
step14.taken <- difftime(step14.time, step13.time, units="secs")
cat(" Step 14/14 completed: writting of R objects and current workspace approx. ", ceiling(step14.taken), " second(s)", "\n", sep = "")

# keep in mind end time
end.time <- Sys.time()

# calculate execution time
time.taken <- difftime(end.time, start.time, units="secs")

# export output log
## prepare overall statistics output
### from training
confusion.overall.train.modif <- as.data.frame(confusion.overall.train)
names(confusion.overall.train.modif)[1] <- "Values"
confusion.overall.train.modif$Values <- as.numeric(confusion.overall.train.modif$Values)
confusion.overall.train.modif$Values <- prettyNum(confusion.overall.train.modif$Values, digits = 3)
### from testing
confusion.overall.test.modif <- as.data.frame(confusion.overall.test)
names(confusion.overall.test.modif)[1] <- "Values"
confusion.overall.test.modif$Values <- as.numeric(confusion.overall.test.modif$Values)
confusion.overall.test.modif$Values <- prettyNum(confusion.overall.test.modif$Values, digits = 3)
## prepare statistics output per classes
### from training
confusion.classes.train.modif <- as.data.frame(confusion.classes.train)
confusion.classes.train.modif[1:(ncol(confusion.classes.train.modif))] <- lapply(confusion.classes.train.modif[1:(ncol(confusion.classes.train.modif))], FUN = function(y){as.numeric(y)})
confusion.classes.train.modif[1:(ncol(confusion.classes.train.modif))] <- lapply(confusion.classes.train.modif[1:(ncol(confusion.classes.train.modif))], FUN = function(y){prettyNum(y, digits = 3)})
### from testing
confusion.classes.test.modif <- as.data.frame(confusion.classes.test)
confusion.classes.test.modif[1:(ncol(confusion.classes.test.modif))] <- lapply(confusion.classes.test.modif[1:(ncol(confusion.classes.test.modif))], FUN = function(y){as.numeric(y)})
confusion.classes.test.modif[1:(ncol(confusion.classes.test.modif))] <- lapply(confusion.classes.test.modif[1:(ncol(confusion.classes.test.modif))], FUN = function(y){prettyNum(y, digits = 3)})
## output in a summary_workflow.txt file
sink(paste(opt$prefix, "summary_workflow.txt", sep = ""))
## print
cat("\n", "###########################################################")
cat("\n", "####################### Information #######################")
cat("\n", "###########################################################", "\n")
cat("\n", "Running start:", start.time.readable, "\n")
cat("\n", "Running time (seconds):", time.taken, "\n")
cat("\n", " Machine Learning outcomes: ", opt$prefix,"\n", sep = "")
cat("\n", "Developped by Nicolas Radomski since July 2022", "\n")
cat("\n", "###########################################################")
cat("\n", "######################### Versions ########################")
cat("\n", "###########################################################", "\n")
cat("\n", "GenomicBasedMachineLearning: 1.0", "\n")
cat("\n", "R:", strsplit(version[['version.string']], ' ')[[1]][3], "\n")
cat("\n", "remotes:", getNamespaceVersion("remotes"), "\n")
cat("\n", "optparse:", getNamespaceVersion("optparse"), "\n")
cat("\n", "caret:", getNamespaceVersion("caret"), "\n")
cat("\n", "doParallel:", getNamespaceVersion("doParallel"), "\n")
cat("\n", "benchmarkme:", getNamespaceVersion("benchmarkme"), "\n")
cat("\n", "data.table:", getNamespaceVersion("data.table"), "\n")
cat("\n", "dplyr:", getNamespaceVersion("dplyr"), "\n")
cat("\n", "janitor:", getNamespaceVersion("janitor"), "\n")
cat("\n", "randomForest:", getNamespaceVersion("randomForest"), "\n")
cat("\n", "gbm:", getNamespaceVersion("gbm"), "\n")
cat("\n", "ranger:", getNamespaceVersion("ranger"), "\n")
cat("\n", "xgboost:", getNamespaceVersion("xgboost"), "\n")
cat("\n", "MLeval:", getNamespaceVersion("MLeval"), "\n")
cat("\n", "MLmetrics:", getNamespaceVersion("MLmetrics"), "\n")
cat("\n", "###########################################################")
cat("\n", "######################## References #######################")
cat("\n", "###########################################################", "\n")
cat("\n", "Please cite as: Pierluigi Castelli, Andrea De Ruvo, Andrea Bucciacchio, Nicola D'Alterio, Cesare Camma, Adriano Di Pasquale and Nicolas Radomski (2023) Harmonization of supervised machine learning practices for efficient source attribution of Listeria monocytogenes based on genomic data. 2023, BMC Genomics, 24(560):1-19, https://doi.org/10.1186/s12864-023-09667-w", "\n")
cat("\n", "GitHub: https://github.com/Nicolas-Radomski/GenomicBasedMachineLearning", "\n")
cat("\n", "Docker: https://hub.docker.com/r/nicolasradomski/genomicbasedmachinelearning", "\n")
cat("\n", "###########################################################")
cat("\n", "######################### Setting #########################")
cat("\n", "###########################################################", "\n")
if (opt$goal == "research"){
  cat("\n", " Goal: Perform prediction and estimate accuracy from the training and testing datasets through the holdout method combined with the repeated k-fold cross-validation method if the tested phenotypes are known (i.e. ", opt$goal, ")", "\n", sep = "")
} else {
  cat("\n", " Goal: Perform prediction and estimate accuracy from the training dataset through the holdout method combined with the repeated k-fold cross-validation method if the tested phenotypes are unknown (i.e. ", opt$goal, ")", "\n", sep = "")
}
cat("\n", "Input mutation file path:", opt$mutations, "\n")
cat("\n", "Input phenotype file path:", opt$phenotype, "\n")
cat("\n", "Number of central processing units:", opt$cpu, "\n")
cat("\n", "Splitting of training and testing dataset:", opt$dataset, "\n")
if (opt$dataset == "manual"){
  cat("\n", "Proportion of training samples during random splitting between training and testing dataset (%):", "NA", "\n")
} else {
  cat("\n", "Proportion of training samples during random splitting between training and testing dataset (%):", opt$splitting, "\n")
}
cat("\n", "Removal of near zero-variance descriptors in the training dataset:", opt$variances, "\n")
if (isTRUE(opt$variances)){
  cat("\n", "Frequency ratio below which descriptors will be considered as near zero-variance descriptors:", opt$ratio, "\n")
  cat("\n", "Percent of unique values above which descriptors will be considered as near zero-variance descriptors:", opt$unique, "\n")
} else {
  cat("\n", "Frequency ratio below which descriptors will be considered as near zero-variance descriptors:", "NA", "\n")
  cat("\n", "Percent of unique values above which descriptors will be considered as near zero-variance descriptors:", "NA", "\n")
}
cat("\n", " k-fold cross-validation: ", round(split_CV_method, digits=3), "\n", sep = "")
cat("\n", " Repetition of the k-fold cross-validation : ", opt$repetition, "\n", sep = "")
cat("\n", "Maximal value of the main parameter:", opt$tuning, "\n")
cat("\n", "Machine Learning model:", opt$fit, "\n")
cat("\n", "Output prefix:", opt$prefix, "\n")
cat("\n", "External representation and current workspace:", opt$backup, "\n")
cat("\n", "###########################################################")
cat("\n", "####################### Output files ######################")
cat("\n", "###########################################################", "\n")
if (opt$goal == "research"){
  cat("\n",
      paste(opt$prefix, "model_predictors.tsv", sep = ""), "\n", 
      paste(opt$prefix, "metrics_training_dataset_tuning_parameters.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_resampling.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_curves.pdf", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_performance.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_confusion_matrix.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_accuracy_overall.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_accuracy_classes.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_testing_dataset_confusion_matrix.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_testing_dataset_accuracy_overall.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_testing_dataset_accuracy_classes.tsv", sep = ""), "\n",
      paste(opt$prefix, "prediction_training_dataset.tsv", sep = ""), "\n",      
      paste(opt$prefix, "prediction_testing_dataset.tsv", sep = ""), "\n",
      paste(opt$prefix, "summary_workflow.txt", sep = ""), "\n",
      paste(opt$prefix, "summary_input_phenotypes.tsv", sep = ""), "\n")
} else {
  cat("\n",
      paste(opt$prefix, "model_predictors.tsv", sep = ""), "\n", 
      paste(opt$prefix, "metrics_training_dataset_tuning_parameters.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_resampling.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_curves.pdf", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_performance.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_confusion_matrix.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_accuracy_overall.tsv", sep = ""), "\n",
      paste(opt$prefix, "metrics_training_dataset_accuracy_classes.tsv", sep = ""), "\n",
      paste(opt$prefix, "prediction_training_dataset.tsv", sep = ""), "\n",
      paste(opt$prefix, "prediction_testing_dataset.tsv", sep = ""), "\n",
      paste(opt$prefix, "summary_workflow.txt", sep = ""), "\n",
      paste(opt$prefix, "summary_input_phenotypes.tsv", sep = ""), "\n")
}
if (isTRUE(opt$backup)){
  cat("", paste(opt$prefix, "saved_data.RData", sep = ""), "\n")
  cat("", paste(opt$prefix, "saved_images.RData", sep = ""), "\n")
}
cat("\n", "###########################################################")
cat("\n", "##################### Dataset splitting ###################")
cat("\n", "###########################################################", "\n")
cat("\n", "Samples included into the training dataset:", nrow(trainID), "\n")
cat("\n", "Samples included into the testing dataset:", nrow(testID), "\n")
cat("\n", "###########################################################")
cat("\n", "####### Remaining descriptors before model training #######")
cat("\n", "###########################################################", "\n")
cat("\n", "Initially:", Initial, "\n")
cat("\n", "After removal of missing values (NAs):", AfterNA, "\n")
cat("\n", "After removal of constant descriptors:", AfterCD, "\n")
cat("\n", "After removal of near zero-variance descriptors:", AfterNZ, "\n")
cat("\n", "###########################################################")
cat("\n", "##################### Model fitting #######################")
cat("\n", "###########################################################", "\n")
cat("\n", "Tuning parameters:", fitModel$metric, "\n")
cat("\n", "Resampling:", nrow(fitModel$resample), "\n")
cat("\n", "Model fitting metric:", fitModel$metric, "\n")
cat("\n", "Best tuning parameters:", colnames(fitModel$bestTune), "=", fitModel$bestTune[1,1], "\n")
cat("\n", "Levels of selected predictors:",length(predictors(fitModel)), "\n")
cat("\n", "###########################################################")
cat("\n", "################## Performance metrics ####################")
cat("\n", "################ from the model training ##################")
cat("\n", "###########################################################", "\n", "\n")
performance.informedness
cat("\n", "###########################################################")
cat("\n", "#################### Model accuracy #######################")
cat("\n", "################ from the training dataset ################")
cat("\n", "###########################################################", "\n", "\n")
cat("Confusion Matrix:", "\n")
confusion.table.train
cat("\n", "Overall Statistics:", "\n", sep = "")
confusion.overall.train.modif
cat("\n", "Statistics by Class:", "\n", sep = "")
confusion.classes.train.modif
cat("\n", "###########################################################")
cat("\n", "#################### Model accuracy #######################")
cat("\n", "################ from the testing dataset #################")
cat("\n", "###########################################################", "\n", "\n")
cat("Confusion Matrix:", "\n")
if (opt$goal == "research"){
  confusion.table.test
} else {
  cat(" NA", "\n")
}
cat("\n", "Overall Statistics:", "\n", sep = "")
if (opt$goal == "research"){
  confusion.overall.test.modif
} else {
  cat(" NA", "\n")
}
cat("\n", "Statistics by Class:", "\n", sep = "")
if (opt$goal == "research"){
  confusion.classes.test.modif
} else {
  cat(" NA", "\n")
}
cat("\n", "###########################################################")
cat("\n", "####################### Prediction ########################")
cat("\n", "################ from the training dataset ################")
cat("\n", "###########################################################", "\n", "\n")
print(results.train, row.names = FALSE)
cat("\n", "###########################################################")
cat("\n", "####################### Prediction ########################")
cat("\n", "################ from the testing dataset #################")
cat("\n", "###########################################################", "\n", "\n")
print(results.test, row.names = FALSE)

## close workflow.txt
sink()

## stop the computing cluster
stopCluster(cluster)

# add messages
cat("\n", "Running time (seconds):", time.taken, "\n")
cat("\n", " Machine Learning outcomes are ready: ", opt$prefix,"\n", sep = "")
cat("\n", "Developped by Nicolas Radomski since July 2022", "\n")
cat("\n", "Please cite as: Pierluigi Castelli, Andrea De Ruvo, Andrea Bucciacchio, Nicola D'Alterio, Cesare Camma, Adriano Di Pasquale and Nicolas Radomski (2023) Harmonization of supervised machine learning practices for efficient source attribution of Listeria monocytogenes based on genomic data. 2023, BMC Genomics, 24(560):1-19, https://doi.org/10.1186/s12864-023-09667-w", "\n", "\n")
