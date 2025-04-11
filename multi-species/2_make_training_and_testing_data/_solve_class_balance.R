#- class-balanced chromosome holdout

#- This function gets a set of chromosomes
#  that (approximately) house a specified fraction of
#  overall examples. It also (approximately) keeps
#  the class balance (binary assumed)

#- The chromosome stat file has a row for each chromosome
# and two columns, 'pos' and 'neg' containing the number
# of positive and negative examples on that chromosome, respectively.

remove_set <- function(chr_stat_file, frac = 0.1) {
    vpc <- read.table(file = chr_stat_file, sep = "\t", header = TRUE)

    # Get names and indices
    chr_names   <- vpc$Chromosome
    n_chroms    <- nrow(vpc)

    # Get the weights for the positive and negative classes
    w_pos       <- vpc[,"Positive"] / sum(vpc[,"Positive"])
    w_neg       <- vpc[,"Negative"] / sum(vpc[,"Negative"])

    # Calculate the linear program
    f.obj <- w_pos - abs(w_pos - w_neg)
    f.con <- t(w_pos)
    f.dir <- c("<=")
    f.rhs <- c(frac)

    sol  <- lpSolve::lp("max", f.obj, f.con, f.dir, f.rhs, binary.vec = 1:n_chroms)

    # And get the indicies
    inds <- which(sol$solution == 1)

    # Now return the rows names based on the indices
    chr_list <- chr_names[inds]

    # Now remove the chrs in chr_list and save the new file
    vpc_filtered <- vpc[!(vpc[,1] %in% chr_list),]
    
    write.table(
        vpc_filtered,
        file = chr_stat_file,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE,
        col.names = TRUE
    )

    rr <- list(
        chrs     = chr_list,
        frac_pos = sum(w_pos[inds]),
        frac_neg = sum(w_neg[inds])
    )
    
    return(jsonlite::toJSON(rr$chrs))
}

# Get arguments from the command line
args <- commandArgs(trailingOnly = TRUE)

# Convert arguments to numeric if needed
chr_stat_file <- args[1]

# Call the function and print the result to standard output
output <- remove_set(chr_stat_file=chr_stat_file)
cat(output)