# Usage
The repository GenomicBasedMachineLearning provides a R script called GenomicBasedMachineLearning:1.0.R to performe Machine Learning prediction of categorical phenotype (e.g. source attribution of miroorganisms) based on binary (e.g. presence/absence profiles of accessory genes or accessory kmers) or categorical (e.g. profiles of core alleles or core variants) genomic data.
# Dependencies
The R script GenomicBasedMachineLearning:1.0.R was prepared and tested with the R version 4.3.0.
- require(remotes) # version 2.4.2
- library(optparse) # version 1.7.3
- library(caret) # version 6.0-94
- library(doParallel) # version 1.0.17
- library(benchmarkme) # version 1.0.8
- library(data.table) # version 1.14.8
- library(dplyr) # version 1.1.2
- library(janitor) # version 2.1.0
- library(randomForest) # version 4.7-1.1
- library(gbm) # version 2.1.8.1
- library(ranger) # version 0.15.1
- library(xgboost) # version 1.7.5.1
- library(MLeval) # version 0.3
- library(MLmetrics) # version 1.1.1
# Expected Input
## 1/ Input mutation file with potential empty cells for missing data (tsv file)
### Example of binary genomic data (e.g. presence/absence profiles of accessory genes: Genes-100-samples.tsv)
```
Gene		2015.TE.14784.1.19.1	2016.TE.28410.1.48.1	2016.TE.3350.1.18.1	2016.TE.4440.1.41.1	2018.TE.15762.1.12.1
group_2951	absence			presence		presence		presence		presence
rpmC		absence			presence		presence		presence		presence
group_2936	absence			absence			absence			presence		presence
tuf		presence		absence			absence			absence			absence
infA		presence		presence					presence		presence
secY		presence		presence		presence		presence		presence
rpsF		presence					presence		presence		presence
fus		presence		presence		presence		presence		presence
rplF		presence		presence		presence		presence		presence
```
### Example of categorical genomic data (e.g. profiles of alleles: Alleles-100-samples.tsv) with L and A standing for locus and alleles, respectively
```
Locus	2015.TE.14784.1.19.1	2016.TE.28410.1.48.1	2016.TE.3350.1.18.1	2016.TE.4440.1.41.1	2018.TE.15762.1.12.1
L1	A4			A2			A2			A2			A2
L2	A4			A2						A2								
L3	A5			A3			A4			A2			A2
L4	A2			A3			A4			A5			A6
L5	A5			A5			A5			A5			A2
L6	A2			A2			A2			A2			A2
L7	A2			A2			A2			A2			A2
L8	A2			A2			A2			A2			A2
L9	A2			A2			A2			A2			A2
L10	A2			A2			A2			A2			A2
```
### Examples of bash commands to prepare the expected mutation file (tsv file)

#### from a Rtab file (e.g. Panaroo output)
```
cat Genes.Rtab > Genes.tsv
```
#### from a vcf file (e.g. snippy-core output)
```
grep -v "##" core.vcf | sed "s@\t@_@" | sed 's@#CHROM_POS@chromosome_position@' | cut --complement -f2,3,4,5,6,7,8 > intermediate-core.tsv
cat <(head -n 1 intermediate-core.tsv) \
    <(sed -E 's@(\t)([0-9])@\1v\2@g' <(grep -v "chromosome_position" intermediate-core.tsv)) \
    >> SNPs.tsv
```
#### from a cgMLST file (e.g. chewBBACA output)
```
col="$(head -n 1 results_alleles.tsv | wc -w)"
for i in $(seq 1 $col); do
    awk '{ print $'$i' }' results_alleles.tsv | paste -s -d "\t"
done >> transposed-results_alleles.tsv
cat <(head -n 1 transposed-results_alleles.tsv) \
    <(grep -v "FILE" transposed-results_alleles.tsv | tr "\t" ',' | sed 's/NIPH//g' | sed 's/NIPHEM//g' | tr ',' "\t") \
    > Alleles.tsv
```
#### from a kmer file (e.g. kmtricks output)
```
cut -d$' ' -f1 list_files-fastqgz.txt > IDs-fastqgz.lst
cat pa_matrix.txt | tr ' ' '\t' | grep -v 'info' > pa_matrix.tsv
col="$(head -1 IDs-fastqgz.lst | wc -w)"
for i in $(seq 1 $col); do
    awk '{ print $'$i' }' IDs-fastqgz.lst | paste -s -d "\t"
done >> transposed-IDs-fastqgz.lst
cat transposed-IDs-fastqgz.lst | tr '\t' ',' | sed 's@^@kmers,@' | tr ',' '\t' > transposed-IDs-fastqgz.tsv
cat transposed-IDs-fastqgz.tsv pa_matrix.tsv > kmers.tsv
```
## 2/ Input phenotype file with or without empty dataset column corresponding to the "--splitting manual" argument (tsv file)
```
sample				phenotype	dataset
2015.TE.14784.1.19.1		pig		training
2016.TE.28410.1.48.1		pig		training
2016.TE.3350.1.18.1		pig		training
2016.TE.4440.1.41.1		pig		training
2018.TE.15762.1.12.1		poultry		training
2019.TE.1226.1.3.1		poultry		training
2019.TE.1367.1.9.1		poultry		training
2019.TE.602.1.8.1		cow		testing
2020.TE.156004.1.4.1		cow		training
```
# Options
```
Usage: GenomicBasedMachineLearning:1.0.R [options]
Options:
	-g CHARACTER, --goal=CHARACTER
		Perform prediction and estimate accuracy from training and testing datasets through the holdout method combined with the repeated k-fold cross-validation method if the tested phenotypes are known (i.e. 'research') or perform prediction and estimate accuracy from the training dataset through the holdout method combined with the repeated k-fold cross-validation method if the tested phenotypes are unknown (i.e. 'surveillance'). [MANDATORY]
	-m CHARACTER, --mutations=CHARACTER
		Input mutation file with an absolute or relative path (tab-separated values). First column: mutations (header: whatever). Other columns: profiles of binary (e.g. presence/absence of genes or kmers) or categorical (e.g. profiles of alleles or variants) mutations for each sample (header: sample identifiers identical to the phenotype input file). [MANDATORY]
	-i CHARACTER, --phenotype=CHARACTER
		Input phenotype file with an absolute or relative path (tab-separated values). First column: sample identifiers identical to the mutation input file (header: 'sample'). Second column: categorical phenotype (header: 'phenotype'). Third column: 'training' or 'testing' dataset (header: 'dataset'). [MANDATORY]
	-c INTEGER, --cpu=INTEGER
		Number of central processing units (CPUs). [OPTIONAL, default = all]
	-d CHARACTER, --dataset=CHARACTER
		Perform random (i.e. 'random') or manual (i.e. 'manual') splitting of training and testing datasets dedicated to the accuracy estimation through the holdout method combined with the repeated k-fold cross-validation method. [OPTIONAL, default = random]
	-s NUMERIC, --splitting=NUMERIC
		Proportion (%) defining the amount of training samples during random splitting of training and testing datasets through the holdout method (e.g. 50, 60, 70, 80 or 90%). [OPTIONAL, default = 80]
	-v LOGICAL, --variances=LOGICAL
		Removal of near zero-variance descriptors from the training dataset. [OPTIONAL, default = TRUE]
	-r NUMERIC, --ratio=NUMERIC
		Frequency of the most prevalent value divided by the frequency of the second most frequent value below which descriptors will be considered as near zero-variance descriptors and discarded from the training dataset (a.k.a. frequency ratio: freqCut argument of the nearZeroVar function from the caret R library). [OPTIONAL, default = 19]
	-u NUMERIC, --unique=NUMERIC
		Hundred times the number of unique values divided by the total number of samples above which descriptors will be considered as near zero-variance descriptors and discarded from the training dataset (a.k.a. percent of unique values: uniqueCut argument of the nearZeroVar function from the caret R library). [OPTIONAL, default = 10]
	-k NUMERIC, --fold=NUMERIC
		Value defining k-1 groups of samples used to train against one group of validation through the k-fold cross-validation method (e.g. 2.0‑, 2.5-, 3.3-, 5.0- or 10-fold cross‑validations). [OPTIONAL, default = 5]
	-e INTEGER, --repetition=INTEGER
		Number of repetition of the k-fold cross-validation method. [OPTIONAL, default = 10]
	-t INTEGER, --tuning=INTEGER
		Maximal value of the main parameter to considered for the model tuning. Ten incremental tenth of the maximal value of the main parameter will be used for the model training. [OPTIONAL, default = 10]
	-f CHARACTER, --fit=CHARACTER
		Perform boosted logistic regression (i.e. 'blr'), extremely randomized trees (i.e. 'ert'), random forest (i.e. 'rf'), stochastic gradient boosting (i.e. 'sgb'), support vector machine (i.e. 'svm') or extreme gradient boosting (i.e. 'xgb') models. [OPTIONAL, default = xgb]
	-o CHARACTER, --prefix=CHARACTER
		Absolute or relative output path with or without output file prefix. [OPTIONAL, default = output_]
	-b LOGICAL, --backup=LOGICAL
		Save an external representation of R objects (i.e. saved_data.RData) and a short-cut of the current workspace (i.e. saved_images.RData). [OPTIONAL, default = FALSE]
	-h, --help
		Show this help message and exit
```
# Unpack GitHub repository and move inside
```
git clone https://github.com/Nicolas-Radomski/GenomicBasedMachineLearning.git
cd GenomicBasedMachineLearning
```
# Install R libraries and launch with Rscript
## 1/ Install R (Ubuntu 20.04 LTS Focal Fossa)
### Install additional Ubuntu libraries
```
sudo apt-get update \
    && apt-get install -y \
    libssl-dev \
    libcurl4-openssl-dev
```
### Install specific R version
```
export R_VERSION=4.3.0
apt install -y --no-install-recommends \
  r-base-core=${R_VERSION} \
  r-base-html=${R_VERSION} \
  r-doc-html=${R_VERSION} \
  r-base-dev=${R_VERSION}
```
### Check installed R version
```
R --version
```
## 2/ Install R libraries
```
R
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
quit()
```
## 3/ Launch with Rscript and different input files and/or options
### Call usage
```
Rscript GenomicBasedMachineLearning:1.0.R
```
### Call help
```
Rscript GenomicBasedMachineLearning:1.0.R -h
```
### Command examples
```
Rscript --max-ppsize=500000 GenomicBasedMachineLearning:1.0.R -g research -m Genes-100-samples.tsv -i PhenotypeDataset-100-samples.tsv -c 6 -d manual -v FALSE -f svm -o test_Rscript_genes_
```
```
Rscript --max-ppsize=500000 GenomicBasedMachineLearning:1.0.R --goal research --mutations Alleles-100-samples.tsv --phenotype PhenotypeDataset-100-samples.tsv --cpu 6 --dataset manual --variances FALSE --fit svm --prefix test_Rscript_alleles_
```
# Install Docker image and launch with Docker
## 1/ Install Docker
### Switch from user to administrator
```
sudo su
```
### Install Docker through snap
```
snap install docker
```
### Switch from administrator to user
```
exit
```
### Create a docker group called docker
```
sudo groupadd docker
```
### Add your user to the docker group
```
sudo usermod -aG docker n.radomski
```
### Activate the modifications of groups
```
newgrp docker
```
### Check the proper installation
```
docker run hello-world
```
## 2/ Pull Docker image from Docker Hub
```
docker pull nicolasradomski/genomicbasedmachinelearning:1.0
```
## 3/ Launch with Docker and different input files and options
### Call usage
```
docker run --name nicolas --rm -u `id -u`:`id -g` nicolasradomski/genomicbasedmachinelearning:1.0
```
### Call help
```
docker run --name nicolas --rm -u `id -u`:`id -g` nicolasradomski/genomicbasedmachinelearning:1.0 -h
```
### Command examples

```
docker run --name nicolas --rm -v $(pwd):/wk -w /wk --ulimit stack=100000000 -e R_MAX_VSIZE=25G nicolasradomski/genomicbasedmachinelearning:1.0 -g research -m Genes-100-samples.tsv -i PhenotypeDataset-100-samples.tsv -c 6 -d manual -v FALSE -f svm -o test_Docker_genes_
```
```
docker run --name nicolas --rm -v $(pwd):/wk -w /wk --ulimit stack=100000000 -e R_MAX_VSIZE=25G nicolasradomski/genomicbasedmachinelearning:1.0 --goal research --mutations Alleles-100-samples.tsv --phenotype PhenotypeDataset-100-samples.tsv --cpu 6 --dataset manual --variances FALSE --fit svm --prefix test_Docker_alleles_
```
# Expected output
- model_predictors.tsv
- metrics_training_dataset_tuning_parameters.tsv
- metrics_training_dataset_resampling.tsv
- metrics_training_dataset_curves.pdf
- metrics_training_dataset_performance.tsv
- metrics_training_dataset_confusion_matrix.tsv
- metrics_training_dataset_accuracy_overall.tsv
- metrics_training_dataset_accuracy_classes.tsv
- metrics_testing_dataset_confusion_matrix.tsv
- metrics_testing_dataset_accuracy_overall.tsv
- metrics_testing_dataset_accuracy_classes.tsv
- prediction_training_dataset.tsv
- prediction_testing_dataset.tsv
- summary_workflow.txt
- summary_input_phenotypes.tsv
- test_saved_data.RData
- test_saved_images.RData
# Illustration
![workflow figure](https://github.com/Nicolas-Radomski/GenomicBasedMachineLearning/blob/main/illustration.png)
# Reference
- Pierluigi Castelli, Andrea De Ruvo, Andrea Bucciacchio, Nicola D'Alterio, Cesare Camma, Adriano Di Pasquale and Nicolas Radomski (2023) Harmonization of supervised machine learning practices for efficient source attribution of Listeria monocytogenes based on genomic data. 2023, BMC Genomics, 24(560):1-19, https://doi.org/10.1186/s12864-023-09667-w
- Docker Hub: https://hub.docker.com/r/nicolasradomski/genomicbasedmachinelearning
- Additional statistical analyses: https://github.com/PCas95/GenomicBasedMachineLearning
- Python users: https://github.com/Nicolas-Radomski/GenomicBasedClassification
# Acknowledgment
The GENPAT-IZSAM Staff for our discussions aiming at managing arguments and building Docker images
# Author
Nicolas Radomski
