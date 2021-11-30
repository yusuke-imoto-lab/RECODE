# scRECOCE_R
scRECODE for R

## Installation

You can install the alpha version of `scRECODE_R` with:

``` r
devtools::install_github("yusuke-imoto-lab/scRECODE_R")
```


## Example
For the single cell data *X* (rows:genes, columns:cells), we can apply scRECODE as follows. 


``` r
library(scRECODE)

X.RECODE <- scRECODE(X)
```
