# RECODE - Resolution of curse of dimensionality

## Python code

### Installation

To install RECODE package, use `pip` as follows:

```
$ pip install screcode
```

### Documentation

[Tutorials and API reference](https://yusuke-imoto-lab.github.io/RECODE/index.html)


### Requirements
* Python3
* numpy
* scipy
* scikit-learn

## R code

### Installation

You can install `RECODE` on R with:

``` r
devtools::install_github("yusuke-imoto-lab/RECODE/R")
```


### Example
For the single-cell sequeincing data *X* (rows: genes/epigenomes, columns: cells), we can apply RECODE as follows. 


``` r
library(RECODE)

X.RECODE <- RECODE(X)
```

In the [Seurat](https://satijalab.org/seurat/) analysis, we can apply RECODE to `SeuratObject`  and set it as default, as follows:

``` r
library(RECODE)
library(Matrix)

data <- as.matrix(seurat[["RNA"]]@counts)
data_RECODE <- RECODE(data)
seurat[["RECODE"]] <- CreateAssayObject(counts = Matrix(data_RECODE, sparse = TRUE))
DefaultAssay(seurat) <- "RECODE"
```