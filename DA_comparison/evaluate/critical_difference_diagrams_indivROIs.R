#!/usr/bin/Rscript
r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
if (!require("devtools")) {
  install.packages("devtools")
  
}
library("devtools")
if (!require("BiocManager")) install.packages("BiocManager") 
if (!require("Rgraphviz")) BiocManager::install("Rgraphviz")

if (!require("scmamp")) devtools::install_github("b0rxa/scmamp")

if (!require('data.table')) install.packages('data.table')
#if (!require('tidyr')) install.packages('tidyr')
library('data.table')
library(tidyr)
library(dplyr)
library('scmamp')


source("aux_CD_diagrams.R", encoding = "UTF-8")

allresultspath <- '../../results/RTLC/allresults_individualrois.csv' 
savefigpath <-'../../figures/CD_diagrams/'
col='X100'
allresults <- read.csv(allresultspath,header=TRUE)
#allresults$Method <- paste(allresults$Method , allresults$Region,sep='-')

if (! dir.exists(savefigpath)) dir.create(file.path(savefigpath),recursive = TRUE)


# 1) AGGREGATE OVER SUBJECT
#library(dplyr)

subjects<-unique(allresults$Subject)
methods<-unique(allresults$Method)
regions <- unique(allresults$Region)



i<-1
cols <- names(allresults[grepl("X[0-9]", names(allresults))])
domains <-unique(allresults$Domain)
results <- c()
for (s in subjects){
  print(s)
  for (m in methods){
    for (r in regions){
      for (d in domains){
        for (col in cols){
          dfcol <- as.data.frame(allresults[(allresults$Subject==s) & (allresults$Method==m) & (allresults$Domain==d) & (allresults$Region==r),col])
          names(dfcol) <- 'bacc'
          dfcol$Subject <- s
          dfcol$Method <- m
          dfcol$Region <- r
          dfcol$column <- col
          dfcol$Domain <- d
          results <- rbind(results, dfcol )
        }
        }
    }
    
  }
}
results <- as.data.frame(results)


praggagg <- analysis_CONCAT(results[(results$Method=='BASELINE') & (results$Domain=='perception'),],'ALL_BASE_PERC', title='BASELINE - PERCEPTION',savefigpath=savefigpath)
praggagg <- analysis_CONCAT(results[(results$Method=='BASELINE') & (results$Domain=='imagery'),],'ALL_BASE_IMAG', title='BASELINE - IMAGERY',savefigpath=savefigpath)
praggagg <- analysis_CONCAT(results[(results$Method=='RTLC') & (results$Domain=='imagery'),],'ALL_RTLC_IMAG', title='RTLC - IMAGERY',savefigpath=savefigpath)
