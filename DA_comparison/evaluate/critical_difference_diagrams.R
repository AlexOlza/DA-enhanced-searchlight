#!/usr/bin/Rscript
r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
library('data.table')
library(tidyr)
library(dplyr)
if (!require("devtools")) {
  install.packages("devtools")
}

if (!require("BiocManager")) install.packages("BiocManager") 
if (!require("Rgraphviz")) BiocManager::install("Rgraphviz")

if (!require("scmamp")) devtools::install_github("b0rxa/scmamp")
if (!require("dplyr")) install.packages("dplyr")

library('scmamp')


source("aux_CD_diagrams.R", encoding = "UTF-8")
dataset = c('ds001246')#commandArgs(trailingOnly=TRUE)
if (dataset[1]=='own'){
  allresultspath <- '../../results/DA_comparison/allresults_DA_comparison.csv' 
  savefigpath <-'../../figures/CD_diagrams/'
  comparison_fname <- '../../figures/CD_diagrams/table_CD_diagrams.csv'
  region_name <- 'allregions'
} else{
  region_name <- 'VC'
  allresultspath <- '../../results/DA_comparison/ds001246_allpresent_oversampled/allresults_DA_comparison.csv' 
  savefigpath <-'../../figures/CD_diagrams/ds001246_allpresent_oversampled/'
  comparison_fname <- '../../figures/CD_diagrams/ds001246_allpresent_oversampled/table_CD_diagrams.csv'
  
}

if (! dir.exists(savefigpath)) dir.create(file.path(savefigpath),recursive = TRUE)
col='X100'
allresults <- read.csv(allresultspath,header=TRUE)
allresults <- allresults[allresults$Domain=='imagery',]
allresults <- allresults[rowSums(allresults == -1) == 0, ]
subjects<-unique(allresults$Subject)
methods<-unique(allresults$Method)

# 2) CONCATENATE NT
i<-1
cols <- names(allresults[grepl("X[0-9]", names(allresults))])

#cols=c('X100')
results <- c()
for (s in subjects){
  for (m in methods){
    for (col in cols){
    dfcol <- as.data.frame(allresults[(allresults$Subject==s) & (allresults$Method==m),col])
    names(dfcol) <- 'bacc'
    dfcol$Subject <- s
    dfcol$Method <- m
    dfcol$column <- col
    results <- rbind(results, dfcol )
    }
    
  }
}
results <- as.data.frame(results)
results[results$Method=='Finetuning','Method'] <- 'FT'
results[results$Method=='DeepCORAL','Method'] <- 'DCORAL'
pr <-analysis_agg_Nt_CONCAT(results,savefigpath)
comparison <- compare_algorithms(pr)
write.csv(comparison,comparison_fname)
i<-1
results <- c()
for (s in subjects){
  for (m in methods){
    for (col in cols){
    dfcol <- as.data.frame(allresults[(allresults$Subject==s) & (allresults$Method==m),col])
    names(dfcol) <- 'bacc'
    dfcol$Subject <- s
    dfcol$Method <- m
    dfcol$column <- col
    results <- rbind(results, dfcol )
    }
    
  }
}
results <- as.data.frame(results)

results[results$Method=='FineTuning','Method'] <- 'FT'
results[results$Method=='DeepCORAL','Method'] <- 'DCORAL'
#pr2<- analysis_CONCAT(results, region_name,'',savefigpath)


i<-1
for (s in sort(subjects)){
  title_s <- paste('Subject',i,sep=' ')
  analysis_agg_Nt_CONCAT_individual_plots(results[results$Subject==s,],savefigpath, title_s)
  i<-i+1
  }
