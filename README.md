# RECODE - Resolution of the curse of dimensionality

<div style="text-align:left"><img style="width:100%; height: auto" src="https://github.com/yusuke-imoto-lab/RECODE/blob/main/images/RECODE_procedure.jpg"/></div>

The preprint is available [here](https://doi.org/10.1101/2022.05.02.490246). 
The license gives permission for personal, academic, or educational use. Any commercial use is strictly prohibited. Please contact imoto.yusuke.4e\<at\>kyoto-u.ac.jp for licensing terms for any commercial use.

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

Remark: The current version of the R code is not fast because of the lower speed of the PCA algorithm on R. Therefore, we recommend using the python code for large-scale data.

### Installation

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

For a detailed analysis, please see below:

[Tutorial (Run)](https://yusukeimoto.github.io/images/RECODE_R_Tutorials/Run_RECODE_on_R_tutorial.html)

[Tutorial (Run, QC, Clustering, Annotating etc.)](https://yusukeimoto.github.io/images/RECODE_R_Tutorials/Run_RECODE_on_R_example.html)


## Application

[RECODE application](https://github.com/yusuke-imoto-lab/GUI-RECODE/releases/tag/v1.1.1)

Windows (exe) and MAC OS (dmg) applications are avairable.

<div style="text-align:left"><img style="width:100%; height: auto" src="https://github.com/yusuke-imoto-lab/RECODE/blob/main/images/RECODE_GUI.jpg"/></div>