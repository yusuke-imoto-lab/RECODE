# RECOCE for R code

Remark: The current version of the R code is not fast because of the lower speed of the PCA algorithm on R. Therefore, we recommend using the python code for large-scale data.

## Installation

You can install `RECODE` on R with:

``` r
devtools::install_github("yusuke-imoto-lab/RECODE/R")
```


### Tutorials
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

Below is a more detailed analysis:

[Tutorial (Run)](https://yusukeimoto.github.io/images/RECODE_R_Tutorials/Run_RECODE_on_R_tutorial.html)

[Tutorial (Run,QC,Clustering,Annotating)](https://yusukeimoto.github.io/images/RECODE_R_Tutorials/Run_RECODE_on_R_example.html)

[Tutorial (Python calling)](https://yusukeimoto.github.io/images/RECODE_R_Tutorials/Run_RECODE_on_R_tutorial3_reticulate.html)