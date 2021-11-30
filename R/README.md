# scRECOCE for R code

## Installation

You can install `scRECODE` on R with:

``` r
devtools::install_github("yusuke-imoto-lab/scRECODE/R")
```


## Example
For the single cell data *X* (rows:genes, columns:cells), we can apply scRECODE as follows. 


``` r
library(scRECODE)

X.RECODE <- scRECODE(X)
```
