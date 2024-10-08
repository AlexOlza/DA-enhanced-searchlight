
plotRanking <-function (pvalues, summary, alpha = 0.05, cex = 0.75, decreasing = FALSE) 
{
    opar <- par(mai = c(0, 0, 0, 0), mgp = c(0, 0, 0))
    on.exit(par(opar))
    k <- length(summary)
    if (decreasing) {
        summary <- k - summary + 1
        decreasing = FALSE
    }
    if (is.matrix(summary)) {
        if (ncol(summary) == 1) {
            summary <- summary[, 1]
        }
        else {
            summary <- summary[1, ]
        }
    }
    #if (!all(sort(colnames(pvalues)) %in% sort(names(summary))) | 
    #    !all(sort(names(summary)) %in% sort(colnames(pvalues)))) {
    #    stop("The column names of 'pvalues' and the names of 'summary' have to contain the same names")
    #}
    control <- NULL
    if (nrow(pvalues) == 1) {
        control <- colnames(pvalues)[is.na(pvalues)]
    }
    o <- order(summary, decreasing = decreasing)
    summary <- summary[o]
    if (nrow(pvalues) > 1) {
        pvalues <- pvalues[names(summary), names(summary)]
    }
    else {
        pvalues <- matrix(pvalues[1, names(summary)], nrow = 1)
        colnames(pvalues) <- names(summary)
    }
    lp <- round(k/2)
    left.algs <- summary[1:lp]
    right.algs <- summary[(lp + 1):k]
    max.rows <- ceiling(k/2)
    char.size <- 0.001
    line.spacing <- 0.25
    to.join <- c()
    m <- floor(min(summary))
    M <- ceiling(max(summary))
    max.char <- max(sapply(names(as.data.frame(pvalues)), FUN = nchar))
    text.width <- (max.char + 4) * char.size
    w <- (M - m) + 2 * text.width
    h.up <- line.spacing
    h.down <- (max.rows + 2.25) * line.spacing
    tick.h <- 0.25 * line.spacing
    label.displacement <- 0.25
    line.displacement <- 0.025
    plot(0, 0, type = "n", xlim = c(m - w/(M - m), M + w/(M - 
        m)), ylim = c(-h.down, h.up), xaxt = "n", yaxt = "n", 
        xlab = "", ylab = "", bty = "n")
    lines(c(m, M), c(0, 0))
    dk <- sapply(m:M, FUN = function(x) {
        lines(c(x, x), c(0, tick.h))
        text(x, 3 * tick.h, labels = x, cex = cex)
    })
    dk <- sapply(1:length(left.algs), FUN = function(x) {
        line.h <- -line.spacing * (x + 2)
        name <- names(left.algs)[x]
        if (!is.null(control) && name == control) {
            font = 4
        }
        else {
            font = 1
        }
        text(x = m - label.displacement, y = line.h, labels = name, 
            cex = cex, adj = 1, font = font)
        lines(c(m - label.displacement * 0.75, left.algs[x]), 
            c(line.h, line.h))
        lines(c(left.algs[x], left.algs[x]), c(line.h, 0))
    })
    dk <- sapply(1:length(right.algs), FUN = function(x) {
        line.h <- -line.spacing * (x + 2)
        name <- names(right.algs)[x]
        if (!is.null(control) && name == control) {
            font = 4
        }
        else {
            font = 1
        }
        text(x = M + label.displacement, y = line.h, labels = name, 
            cex = cex, adj = 0, font = font)
        lines(c(M + label.displacement * 0.75, right.algs[x]), 
            c(line.h, line.h))
        lines(c(right.algs[x], right.algs[x]), c(line.h, 0))
    })
    if (nrow(pvalues) == 1) {
        to.join <- summary[which(is.na(pvalues) | pvalues > alpha)]
        if (length(to.join) > 1) {
            lines(c(min(to.join), max(to.join)), c(0, 0), lwd = 3)
        }
    }
    else {
        getInterval <- function(x) {
            ls <- which(pvalues[x, ] > alpha)
            ls <- ls[ls > x]
            res <- NULL
            if (length(ls) > 0) {
                res <- c(as.numeric(summary[x]), as.numeric(summary[max(ls)]))
            }
            return(res)
        }
        intervals <- mapply(1:(k - 1), FUN = getInterval)
        if (is.matrix(intervals)) {
            aux <- t(intervals)
        }
        else {
            aux <- do.call(rbind, intervals)
        }
        if (length(aux) > 0) {
            to.join <- aux[1, ]
            if (nrow(aux) > 1) {
                for (r in 2:nrow(aux)) {
                  if (aux[r - 1, 2] < aux[r, 2]) {
                    to.join <- rbind(to.join, aux[r, ])
                  }
                }
            }
            row <- c(1)
            if (!is.matrix(to.join)) {
                to.join <- t(as.matrix(to.join))
            }
            nlines <- dim(to.join)[1]
            for (r in 1:nlines) {
                id <- which(to.join[r, 1] > to.join[, 2])
                if (length(id) == 0) {
                  row <- c(row, tail(row, 1) + 1)
                }
                else {
                  row <- c(row, min(row[id]))
                }
            }
            step <- max(row)/2
            dk <- sapply(1:nlines, FUN = function(x) {
                y <- -line.spacing * (0.5 + row[x]/step)
                lines(c(to.join[x, 1] - line.displacement, to.join[x, 
                  2] + line.displacement), c(y, y), lwd = 3)
            })
        }
    }
    return(list('groups' = to.join))
}
compare_algorithms <- function(pr) {
  # Get the names of the algorithms from the first ranking
  algorithms <- base::colnames(pr$rankings[[1]])
  
  # Initialize the result matrix
  result <- matrix(0, nrow = length(algorithms), ncol = length(algorithms))
  base::rownames(result) <- algorithms
  base::colnames(result) <- algorithms
  
  # Loop through each ranking and corresponding lines
  for (k in seq_along(pr$rankings)) {
    ranking_k <- pr$rankings[[k]] # matrix of rankings
    lines_k <- pr$lines[[k]]      # matrix of intervals
    
    # Compare each pair of algorithms i and j
    for (i in seq_along(algorithms)) {
      for (j in seq_along(algorithms)) {
        if (i != j) {
          x <- ranking_k[1, i]  # Value for algorithm i
          y <- ranking_k[1, j]  # Value for algorithm j
          
          # Check if x < y
          if (x < y) {
            # Check if both x and y are not in any of the intervals in lines_k
            in_interval <- FALSE
            for (interval in 1:nrow(lines_k)) {
              lower <- min(lines_k[interval, 1], lines_k[interval, 2])
              upper <- max(lines_k[interval, 1], lines_k[interval, 2])
              if (x >= lower && x <= upper && y >= lower && y <= upper) {
                in_interval <- TRUE
                break
              }
            }
            
            # If not in an interval, count this as a valid comparison
            if (!in_interval) {
              result[i, j] <- result[i, j] + 1
            }
          }
        }
      }
    }
  }
  
  # Convert result matrix to dataframe for easier manipulation
  df_result <- as.data.frame(result)
  
  # Add a column for the sum of each row
  df_result$Total <- rowSums(df_result)
  
  # Add a row for the sum of each column (including the RowSum column)
  col_sums <- colSums(df_result)
  df_result <- rbind(df_result, Total = col_sums)
  
  return(df_result)
}


analysis_CONCAT <- function(results,region,title, savefigpath){
  print('COMPLETELY AGGREGATED ANALYSIS')
  
  df <- results
  if (length(unique(df$Method))>1){ df$Method <- paste(df$Method,df$Region,sep=' ')
  }
  else {
    df$Method <- df$Region
  }
  ranking <- list()
  horizontal.lines <- list()
  fname <-paste(savefigpath,'completely_Aggregated_CONCAT_',region,'.png',sep='')
  #png(filename=fname, width=900*4, height=900*5,res=300)
  
  widedf <- data.frame(matrix(NA, ncol=1, nrow=nrow(df)/length(unique(df$Method))))[-1]
  for (m in unique(df$Method)){
    widedf[,m]<-df[df$Method==m,'bacc']
  }
    # FREQUENTIST Critical Difference diagrams
    # STEP 1) Assess whether all algorithms' performances come from the same distribution (H0). 
    # In case of rejection, there is at least one algorithm that performs differently to others and we will do a post test 
    
    ftest<- friedmanTest(widedf)
    print(ftest)
    if (ftest$p.value<=0.05 || is.na(ftest$p.value)) {
      #Friedman test significant, procceeding with Nemenyi post-hoc test for pairwise comparisons
      res <- postHocTest(widedf, test = "aligned ranks", correct = 'shaffer', use.rank=TRUE)
   
      pv.adj <- res$corrected.pval
     
      png(fname,width = 1200, height=600, res=300)
      par(mar = c(2,2,4,1))
      #plotCD(widedf,alpha=0.05, title = s)
      horizontal.lines <-plotRanking(pv.adj, res$summary, cex = 0.6)
      mtext(side=3, line=2.3, adj=0.5, cex=0.7, title)
      dev.off()
      ranking <- list(res$summary)  
    }
    else print('Non significant test')
    return(list('rankings'=ranking,'lines'=horizontal.lines))
  }
analysis_agg_Nt_CONCAT <- function(results){
  print('ANALYSIS AGGREGATED OVER NT')
  savefigpath <-'../../figures/CD_diagrams/'
  subjects<-unique(results$Subject)
  df <- results
  ranking <- list()
  horizontal.lines <- list()
  fname <-paste(savefigpath,'allSubjects','_concat.png',sep='')
  png(filename=fname, width=900*4, height=900*5,res=300)
  layout(matrix(1:length(subjects),nrow=ceiling(length(subjects)/2),ncol=2,byrow=TRUE))
  i=1
  for (s in subjects){
    df_subj <- df[df$Subject==s, c('bacc','Method')]
    widedf <- data.frame(matrix(NA, ncol=1, nrow=nrow(df_subj)/length(unique(df_subj$Method))))[-1]
    for (m in sort(unique(df_subj$Method))){
      widedf[,m]<-df_subj[df_subj$Method==m,'bacc']
    }
    
    # FREQUENTIST Critical Difference diagrams
    # STEP 1) Assess whether all algorithms' performances come from the same distribution (H0). 
    # In case of rejection, there is at least one algorithm that performs differently to others and we will do a post test # nolint
    
    ftest<- friedmanTest(widedf)
    if (ftest$p.value<=0.05) {
      #print('Friedman test significant, procceeding with Nemenyi post-hoc test for pairwise comparisons')
      res <- postHocTest(widedf, test = "aligned ranks", correct = 'shaffer', use.rank=TRUE)
      #pv.matrix <- postHocTest(data=widedf, control=NULL)
      pv.adj <- res$corrected.pval
      r <- rankMatrix(widedf)
      r.means <- colMeans(r)
      #drawAlgorithmGraph(pvalue.matrix = pv.adj, mean.value = r.means)
      ordering <- order(summarizeData(widedf))
      
      #plotPvalues(pv.adj,alg.order=ordering)+ggtitle(s)#+theme(aspect.ratio=1,plot.margin = margin(0,0,0,0))
      #ggsave(paste(savefigpath,s,'_pvalues.png',sep=''),width=12,height=10)
      
      #png(paste(savefigpath,s,'.png',sep=''))
      #plotCD(widedf,alpha=0.05, title = s)
      horizontal.lines[s] <- plotRanking(pv.adj, res$summary, cex = 0.9)
      title(paste('Subj.',i,sep=''), cex=0.5)
      i=i+1
    
    ranking[s] <- list(res$summary)  
    }
    else print(paste(c('Non-significant for ',s),sep=' '))
  }
 
  dev.off()
  return(list('rankings'=ranking,'lines'=horizontal.lines))
}
