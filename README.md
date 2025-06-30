# RECODE - Resolution of the curse of dimensionality

<div style="text-align:left"><img style="width:50%; height: auto" src="https://github.com/yusuke-imoto-lab/RECODE/blob/main/images/Logo_RECODE.jpg"/></div>
<div style="text-align:left"><img style="width:100%; height: auto" src="https://github.com/yusuke-imoto-lab/RECODE/blob/main/images/RECODE_procedure.jpg"/></div>


Resolution of the curse of dimensionality (RECODE) is a noise reduction method for single-cell sequencing data based on high-dimensional statistics.

- [Y. Imoto, T. Nakamura, et al. Resolution of the curse of dimensionality in single-cell RNA sequencing data analysis, *Life Science Alliance*, 2022](https://dx.doi.org/10.26508/lsa.202201591). 

- [Y. Imoto. Comprehensive Noise Reduction in Single-Cell Data with the RECODE Platform, *bioRxiv*, 2024](https://doi.org/10.1101/2024.04.18.590054). 

- [Y. Imoto. Accurate highly variable gene selection using RECODE in scRNA-seq data analysis, *bioRxiv*, 2025](https://doi.org/10.1101/2025.06.23.661026). 

The license gives permission for personal, academic, or educational use. Any commercial use is strictly prohibited. Please contact [imoto.yusuke.4e@kyoto-u.ac.jp](mailto:imoto.yusuke.4e@kyoto-u.ac.jp) for licensing terms for any commercial use.

---

## Table of Contents

* [Overview](#overview)
* [Python code](#python-code)
* [R code](#r-code)
* [R code (Python calling)](#r-code-python-calling)
* [Desktop Application](#desktop-application)
* [License](#license)
* [Citation](#citation)
* [Contact](#contact)

---

## Overview

- Input is single-cell sequencing data (count matrix) $X \in \mathbb{Z}_{\geq 0}^{n\times d}$, where $n$ is the number of sample, $d$ is the number of features. For exmple, for scRNA-seq data, $n$ and $d$ correspond to the number of cells and genes, respectively. 
- Compute the denoised data $X \in \mathbb{R}_{\geq 0}^{n\times d}$ with the same scale with $X$.
- Compute the applicability of RECODE, classified *strongly applicable*, *weekly applicable*, and *inapplicable*, denoting the level of accuracy of noise reduction.


---

## Python code


### Installation

To install RECODE package, use `pip` as follows:

```
$ pip install screcode
```

PyPi downloads (by PePy)

[![Week](https://static.pepy.tech/personalized-badge/screcode?period=week&units=international_system&left_color=black&right_color=green&left_text=Week)](https://pepy.tech/project/screcode)
[![Month](https://static.pepy.tech/personalized-badge/screcode?period=month&units=international_system&left_color=black&right_color=yellow&left_text=Month)](https://pepy.tech/project/screcode)
[![Total](https://static.pepy.tech/personalized-badge/screcode?period=total&units=international_system&left_color=black&right_color=orange&left_text=Total)](https://pepy.tech/project/screcode)

### Documentation and Tutorial

[Tutorials and API reference](https://yusuke-imoto-lab.github.io/RECODE/index.html)


### Requirements
* Python3
* numpy
* scipy
* scikit-learn

---

## R code

Remark: The current version of the R code is not fast because of the lower speed of the PCA algorithm on R. So, we recommend using the Python code or R code with Python calling (below) for large-scale data.

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

---

## R code (Python calling)

### Installation

After installing remotes (`install.packages("remotes")`), you can install "recodeinstaller" with the following command:

``` r
remotes::install_github("yusuke-imoto-lab/recodeinstaller")
```

Then, the following command installs the Python version of RECODE.

``` r
library(recodeinstaller)

install_screcode()
```
Regarding the detail of installer, please see [recodeinstaller](https://github.com/yusuke-imoto-lab/recodeinstaller). 


### Tutorials
For the single-cell sequeincing data *X* (rows: genes/epigenomes, columns: cells), we can apply RECODE as follows. 


``` r
library(reticulate)

source("recodeloader/load_recodeenv.R")

plt <- reticulate::import(module="matplotlib.pyplot")
screcode <- reticulate::import(module="screcode.screcode")
X.RECODE<-screcode$RECODE(X)

```
Below is a more detailed analysis:

[Tutorial (Python calling)](https://yusukeimoto.github.io/images/RECODE_R_Tutorials/Run_RECODE_on_R_tutorial3_reticulate-recodeinstaller.html)

---

## Desktop Application

[Installation and Tutorials](https://github.com/yusuke-imoto-lab/GUI-RECODE#desktop-application-of-recode)

Windows (exe) and MAC OS (dmg) applications are avairable.

<div style="text-align:left"><img style="width:100%; height: auto" src="https://github.com/yusuke-imoto-lab/RECODE/blob/main/images/RECODE_GUI.jpg"/></div>

---

## License

MIT © 2022 Yusuke Imoto

The license gives permission for personal, academic, or educational use. Any commercial use is strictly prohibited. Please contact [imoto.yusuke.4e@kyoto-u.ac.jp](mailto:imoto.yusuke.4e@kyoto-u.ac.jp) for licensing terms for any commercial use.

---

## Citation

- Y. Imoto, T. Nakamura, et al. Resolution of the curse of dimensionality in single-cell RNA sequencing data analysis, *Life Science Alliance*, 2022. 

- Y. Imoto. Comprehensive Noise Reduction in Single-Cell Data with the RECODE Platform, *bioRxiv*, 2024. 

- Y. Imoto. Accurate highly variable gene selection using RECODE in scRNA-seq data analysis, *bioRxiv*, 2025. 

#### BibTex

```bibtex
@article{Imoto2022RECODE,
   author = {Imoto, Yusuke and Nakamura, Tomonori and Escolar, Emerson G and Yoshiwaki, Michio and Kojima, Yoji and Yabuta, Yukihiro and Katou, Yoshitaka and Yamamoto, Takuya and Hiraoka, Yasuaki and Saitou, Mitinori},
   title = {Resolution of the curse of dimensionality in single-cell RNA sequencing data analysis},
   journal = {Life Sci Alliance},
   volume = {5},
   number = {12},
   DOI = {10.26508/lsa.202201591},
   year = {2022},
   type = {Journal Article}
}
```

```bibtex
@article{Imoto2024iRECODE,
    author = {Imoto, Yusuke},
   title = {Comprehensive Noise Reduction in Single-Cell Data with the RECODE Platform},
   journal = {bioRxiv},
   DOI = {10.1101/2024.04.18.590054},
   year = {2024},
   type = {Journal Article}
}
```



---

## Contact

* **Yusuke Imoto**
* Email: [imoto.yusuke.4e@kyoto-u.ac.jp](mailto:imoto.yusuke.4e@kyoto-u.ac.jp)
* GitHub: [yusuke-imoto-lab/RECODE](https://github.com/yusuke-imoto-lab/RECODE)