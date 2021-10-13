scRNA-seq data - 10X chromium HDF5 file
========

We show an exmaple for scRNA-seq data produced by 10X Chromium. 
We use sample `10k Human PBMCs, 3' v3.1, Chromium Controller` (11,485 cells and 36,601 genes) in `10X Genomics Datasets <https://www.10xgenomics.com/jp/resources/datasets>`_.  
The test data is directly avairable from `Feature / cell matrix HDF5 (filtered)` in `here <https://www.10xgenomics.com/jp/resources/datasets/10k-human-pbmcs-3-v3-1-chromium-controller-3-1-high>`_ (need register).


We use ``scanpy<https://scanpy.readthedocs.io/en/stable/>``_ to read/write 10X HDF5 file. 
Import  ``numpy``, ``scipy``, and ``scanpy`` in addlition to ``screcode``. 

.. code-block:: python

	import screcode
	import numpy as np
	import scipy
	import scanpy


Imput data from HDF5 (\*\*\*.h5) file. 

.. code-block:: python

	input_filename = '10k_PBMC_3p_nextgem_Chromium_Controller_filtered_feature_bc_matrix.h5'
	anndata = scanpy.read_10x_h5(input_filename)

Apply scRECODE. 

.. code-block:: python

	screc = screcode.scRECODE()
	data_scRECODE = screc.fit_transform(anndata.X.toarray())

.. parsed-literal::

	start scRECODE
	end scRECODE
	log: {'#significant genes': 15789, '#non-significant genes': 9322, '#silent genes': 11490, 'ell': 165, 'Elapsed_time': '54.8484[sec]'}
	
Write the denoised data as HDF5 file. The output filename is 

.. code-block:: python

	anndata_scRECODE = anndata
	anndata_scRECODE.X = scipy.sparse.csc_matrix(data_scRECODE)
	output_filename = '10k_PBMC_3p_nextgem_Chromium_Controller_filtered_feature_bc_matrix_scRECODE.h5'
	anndata_scRECODE.write(output_filename)

Check applicability. 

.. code-block:: python

	screc.check_applicability()


.. parsed-literal::

	applicabity: (A) Strong applicable

.. image:: ../image/Example_10X_RNA_applicability.svg
	

Show scatter plots of mean vs variance before and after scRECODE. 	

.. code-block:: python

	screc.compare_mean_variance_log()

.. image:: ../image/Example_10X_RNA_mean_var_log.svg

Show noise variance for each gene (sorted by mean expresion level). 

.. code-block:: python

	screc.plot_noise_variance()

.. image:: ../image/Example_10X_RNA_noise_variance.svg

Show the variance after noise-variance-stabilizing normalization. 

.. code-block:: python

	screc.plot_normalization()

.. image:: ../image/Example_10X_RNA_noise_normalization.svg

Check log. 

.. code-block:: python

	screc.log
	

.. parsed-literal::

	{'#significant genes': 11628,
	 '#non-significant genes': 8189,
	 '#silent genes': 16784,
	 'ell': 34,
	 'Elapsed_time': '10.13[sec]',
	 'Applicability': '(A) Strong applicable',
	 "Rate of '0 < normalized variance < 0.9'": '0%',
	 'Peak density of normalized variance': 1.0013721697775515}
