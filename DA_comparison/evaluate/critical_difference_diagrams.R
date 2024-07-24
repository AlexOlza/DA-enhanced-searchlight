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

allresultspath <- '../../results/DA_comparison/allresults_DA_comparison.csv' 
savefigpath <-'../../figures/CD_diagrams/'
if (! dir.exists(savefigpath)) dir.create(file.path(savefigpath))
col='X100'
allresults <- read.csv(allresultspath,header=TRUE)
allresults <- allresults[allresults$Domain=='imagery',]

subjects<-unique(allresults$Subject)
methods<-unique(allresults$Method)


# 2) CONCATENATE NT
i<-1
cols <- names(allresults[grepl("X[0-9]", names(allresults))])
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

pr <-analysis_agg_Nt_CONCAT(results)
comp <- comparison.table(pr)
write.csv(comp,'../../figures/CD_diagrams/table_CD_diagrams.csv')
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

pr<- analysis_CONCAT(results, 'allregions','',savefigpath)


