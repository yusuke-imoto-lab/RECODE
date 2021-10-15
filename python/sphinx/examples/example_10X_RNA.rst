scRNA-seq data - 10X chromium HDF5 file
========

We show an exmaple for scRNA-seq data produced by 10X Chromium. 
We use sample `10k Human PBMCs, 3' v3.1, Chromium Controller` (11,485 cells and 36,601 genes) in `10X Genomics Datasets <https://www.10xgenomics.com/jp/resources/datasets>`_.  
The test data is directly avairable from `Feature / cell matrix HDF5 (filtered)` in `here <https://www.10xgenomics.com/jp/resources/datasets/10k-human-pbmcs-3-v3-1-chromium-controller-3-1-high>`_ (need register).


We use `scanpy <https://scanpy.readthedocs.io/en/stable/>`_ to read/write 10X HDF5 file (\*\*\*.h5). 
Import  ``numpy``, ``scipy``, and ``scanpy`` in addlition to ``screcode``. 

.. code-block:: python

	import screcode
	import numpy as np
	import scipy
	import scanpy


Imput data from HDF5 file (\*\*\*.h5). 

.. code-block:: python

	input_filename = '10k_PBMC_3p_nextgem_Chromium_Controller_filtered_feature_bc_matrix.h5'
	anndata = scanpy.read_10x_h5(input_filename)

Apply scRECODE. 

.. code-block:: python

	recode = screcode.scRECODE()
	data_scRECODE = recode.fit_transform(anndata.X.toarray())

.. parsed-literal::

	start scRECODE for scRNA-seq
	end scRECODE for scRNA-seq
	log: {'#significant genes': 15789, '#non-significant genes': 9322, '#silent genes': 11490, 'ell': 165, 'Elapsed_time': '54.8484[sec]'}
	
Write the denoised data as HDF5 file. 

.. code-block:: python

	adata_scRECODE = adata.copy()
	adata_scRECODE.X = scipy.sparse.csc_matrix(data_scRECODE)
	adata_scRECODE.var['noise_variance'] = screc.noise_variance
	adata_scRECODE.var['normalized_variance'] = screc.normalized_variance
	adata_scRECODE.var['significance'] = screc.significance
	adata_scRECODE.var_names_make_unique()
	output_filename = '10k_PBMC_3p_nextgem_Chromium_Controller_filtered_feature_bc_matrix_scRECODE.h5'
	adata_scRECODE.write(output_filename)

Check applicability. 

.. code-block:: python

	recode.check_applicability()


.. parsed-literal::

	applicabity: (A) Strong applicable

.. image:: ../image/Example_10X_RNA_applicability.png
	

Show scatter plots of mean vs variance before and after scRECODE. 	

.. code-block:: python

	recode.plot_mean_variance()

.. image:: ../image/Example_10X_RNA_mean_var_log_Original.png

.. image:: ../image/Example_10X_RNA_mean_var_log_scRECODE.png

Show noise variance for genes which are sorted by mean expresion level. 

.. code-block:: python

	recode.plot_noise_variance()

.. image:: ../image/Example_10X_RNA_noise_variance.png

Show the variance after noise-variance-stabilizing normalization. 

.. code-block:: python

	recode.plot_normalization()

.. image:: ../image/Example_10X_RNA_noise_normalization.png

Check the log. 

.. code-block:: python

	recode.log
	

.. parsed-literal::

	{'#significant genes': 15789,
	 '#non-significant genes': 9322,
	 '#silent genes': 11490,
	 'ell': 165,
	 'Elapsed_time': '56.8615[sec]',
	 'Applicability': '(A) Strong applicable',
	 'Rate of 0 < normalized variance < 0.9': '0%',
	 'Peak density of normalized variance': 1.016056101497848}


Show the gene rank given by the normalizedd variance. 

.. code-block:: python
	 
	import pandas as pd
	n_show_genes = 10
	idx = np.argsort(recode.normalized_variance)[::-1]
	generank = pd.DataFrame({'gene':adata.var.index[idx],
                         'normalized_variance':recode.normalized_variance[idx],
                         'significance':recode.significance[idx]},
                        index=np.arange(len(adata.var.index))+1)
	generank.head(n_show_genes)
	 
.. raw:: html

	<div>
  <style scoped>
      .dataframe tbody tr th:only-of-type {
          vertical-align: middle;
      }
  
      .dataframe tbody tr th {
          vertical-align: top;
      }
  
      .dataframe thead th {
          text-align: right;
      }
  </style>
	<table border="1" class="dataframe">
		<thead>
		  <tr style="text-align: right;">
		    <th></th>
		    <th>gene</th>
		    <th>normalized_variance</th>
		    <th>significance</th>
		  </tr>
		</thead>
		<tbody>
		  <tr>
		    <th>1</th>
		    <td>IGKC</td>
		    <td>476.251373</td>
		    <td>significant</td>
		  </tr>
		  <tr>
		    <th>2</th>
		    <td>IGLC3</td>
		    <td>337.377136</td>
		    <td>significant</td>
		  </tr>
		  <tr>
		    <th>3</th>
		    <td>IGHA1</td>
		    <td>315.810333</td>
		    <td>significant</td>
		  </tr>
		  <tr>
		    <th>4</th>
		    <td>IGLC2</td>
		    <td>250.899536</td>
		    <td>significant</td>
		  </tr>
		  <tr>
		    <th>5</th>
		    <td>IGHG1</td>
		    <td>209.024307</td>
		    <td>significant</td>
		  </tr>
		  <tr>
		    <th>6</th>
		    <td>IGLC1</td>
		    <td>197.974701</td>
		    <td>significant</td>
		  </tr>
		  <tr>
		    <th>7</th>
		    <td>S100A9</td>
		    <td>144.979065</td>
		    <td>significant</td>
		  </tr>
		  <tr>
		    <th>8</th>
		    <td>IGHG2</td>
		    <td>123.463943</td>
		    <td>significant</td>
		  </tr>
		  <tr>
		    <th>9</th>
		    <td>MALAT1</td>
		    <td>98.790283</td>
		    <td>significant</td>
		  </tr>
		  <tr>
		    <th>10</th>
		    <td>S100A8</td>
		    <td>75.027397</td>
		    <td>significant</td>
		  </tr>
		</tbody>
	</table>
	</div>


