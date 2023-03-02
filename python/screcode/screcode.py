import anndata
import adjustText
import datetime
import harmonypy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import sklearn.decomposition
import scipy.sparse
import seaborn as sns
import time
import warnings



class RECODE():
	def __init__(
		self,
		fast_algorithm = True,
		fast_algorithm_ell_ub = 1000,
		seq_target = 'RNA',
		version = 1,
		decimals = 5,
		verbose = True
		):
		""" 
		RECODE (Resolution of curse of dimensionality in single-cell data analysis). A noise reduction method for single-cell sequencing data. 
		
		Parameters
		----------
		fast_algorithm : boolean, default=True
			If True, the fast algorithm is conducted. The upper bound of parameter :math:`\ell` is set in ``fast_algorithm_ell_ub``.
		
		fast_algorithm_ell_ub : int, default=1000
			Upper bound of parameter :math:`\ell` for the fast algorithm. Must be of range [1,:math:`\infity`).
		
		seq_target : {'RNA','ATAC'}, default='RNA'
			Sequencing target. If 'ATAC', the preprocessing (odd-even stabilization) will be performed before the regular algorithm. 

		version : int default='1'
			Version of RECODE. 
		
		verbose : boolean, default=True
			If False, all running messages are not displayed.

		decimals : int default='5'
			Number of decimals for round processed matrices.
		
		Attributes
		----------
		cv_ : ndarray of shape (n_features,)
			Coefficient of variation of features (genes/peaks).
			
		log_ : dict
			Running log.
		
		noise_variance_ : ndarray of shape (n_features,)
			Noise variances of features (genes/peaks).
		
		normalized_variance_ : ndarray of shape (n_features,)
			Variances of features (genes/peaks).
		
		significance_ : ndarray of shape (n_features,)
			Significance (significant/non-significant/silent) of features (genes/peaks).
		"""
		self.fast_algorithm = fast_algorithm
		self.fast_algorithm_ell_ub = fast_algorithm_ell_ub
		self.seq_target = seq_target
		self.version = version
		self.decimals = decimals
		self.verbose = verbose
		self.unit,self.Unit = 'gene','Gene'
		if seq_target == 'ATAC':
			self.unit,self.Unit = 'peak','Peak'
		self.log_ = {}
		self.log_['seq_target'] = self.seq_target
		self.fit_idx = False

	def _check_datatype(
		self,
		X
	):
		if type(X) == anndata._core.anndata.AnnData:
			if scipy.sparse.issparse(X.X):
				return X.X.toarray()
			elif type(X.X) == np.ndarray:
				return X.X
			else:
				raise TypeError("Data type error: ndarray or anndata is available.")
		elif scipy.sparse.issparse(X):
			warnings.warn('RECODE does not support sparse input. The input and output are transformed as regular matricies. ')
			return X.toarray()
		elif type(X) == np.ndarray:
			return X
		else:
			raise TypeError("Data type error: ndarray or anndata is available.")
	

	def _noise_variance_stabilizing_normalization(
		self,
		X
	):
		"""
		Apply the noise-variance-stabilizing normalization to X.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Data matrix
		"""
		d = X.shape[1]
		if d == self.d_all:
			X_ = X[:,self.idx_nonsilent]
		elif d == self.d_nonsilent:
			X_ = X
		else:
			raise TypeError("Dimension of data is not correct.")
		
		## scaled X
		self.X_nUMI = np.sum(X_,axis=1)
		X_scaled = (X_.T/self.X_nUMI).T
		## normalization
		X_norm = (X_scaled-self.X_scaled_mean)/np.sqrt(self.noise_var)

		if d == self.d_all:
			X_norm_ = np.zeros(X.shape,dtype=float)
			X_norm_[:,self.idx_nonsilent] = X_norm
			return X_norm_
		elif d == self.d_nonsilent:
			return X_norm
		
	
	def _inv_noise_variance_stabilizing_normalization(
		self,
		X
	):
		"""
		Apply the inverce transformation of noise-variance-stabilizing normalization to X. 
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Data matrix
		"""
		X_norm_inv_temp = X*np.sqrt(self.noise_var)+self.X_scaled_mean
		X_norm_inv = (X_norm_inv_temp.T*self.X_nUMI).T
		return X_norm_inv
	
	def _ATAC_preprocessing(self,X):
		"""
		Preprocessing of original ATAC-seq data (odd-even stabilization data). 
		
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Data matrix (scATAC-seq data)
		
		Returns
		-------
		X_new : ndarray of shape (n_samples, n_features)
			Preprecessed data matrix.
		"""
		X_new = (X+1)//2
		return X_new
	
	def fit(self,X):
		"""
		Fit the model to X. (Determine the transformation.)
		
		Parameters
		----------
		X : ndarray or anndata of shape (n_samples, n_features).
			single-cell sequencing data matrix (row:cell, culumn:gene/peak).

		"""	
		X_mat = self._check_datatype(X)
		self.idx_nonsilent = np.sum(X_mat,axis=0) > 0
		self.X_temp = X_mat[:,self.idx_nonsilent]
		if self.seq_target == 'ATAC':
			self.X_temp = self._ATAC_preprocessing(self.X_temp)
		
		X_nUMI = np.sum(self.X_temp,axis=1)
		X_scaled = (self.X_temp.T/X_nUMI).T
		X_scaled_mean = np.mean(X_scaled,axis=0)
		noise_var = np.mean(self.X_temp.T/np.sum(self.X_temp,axis=1)/np.sum(self.X_temp,axis=1),axis=1)
		noise_var[noise_var==0] = 1
		X_norm = (X_scaled-X_scaled_mean)/np.sqrt(noise_var)
		X_norm_var = np.var(X_norm,axis=0)
		recode_ = RECODE_core(variance_estimate=False,fast_algorithm=self.fast_algorithm,fast_algorithm_ell_ub=self.fast_algorithm_ell_ub,version=self.version)
		recode_.fit(X_norm)

		# self.X_fit = X_mat
		self.n_all = X_mat.shape[0]
		self.d_all = X_mat.shape[1]
		self.d_nonsilent = sum(self.idx_nonsilent)
		self.noise_var = noise_var
		self.recode_ = recode_
		self.X_norm_var = X_norm_var
		self.X_fit_nUMI = X_nUMI
		self.X_scaled_mean = X_scaled_mean
		self.idx_sig = self.X_norm_var > 1
		self.idx_nonsig = self.idx_sig==False
		self.log_['#significant %ss' % self.unit] = sum(self.idx_sig)
		self.log_['#non-significant %ss' % self.unit] = sum(self.idx_nonsig)
		self.log_['#silent %ss' % self.unit] = X.shape[1] - sum(self.idx_sig) - sum(self.idx_nonsig)
		self.fit_idx = True
	
	def transform(
		self,
		X
	):
		"""
		Transform X into RECODE-denoised data.

		Parameters
		----------
		X : ndarray or anndata of shape (n_samples, n_features)
			Single-cell sequencing data matrix (row:cell, culumn:gene/peak).

		Returns
		-------
		X_new : ndarray of shape (n_samples, n_features)
			RECODE-denoised data matrix.
		"""
		X_mat = self._check_datatype(X)
		if self.fit_idx == False:
			raise TypeError("Run fit before transform.")
		if X_mat.shape[1] != self.d_all:
			raise TypeError("RECODE requires the same dimension as that of fitted data.")
		X_ = X_mat[:,self.idx_nonsilent]
		X_norm = self._noise_variance_stabilizing_normalization(X_)
		X_norm_RECODE = self.recode_.transform(X_norm)
		X_RECODE = np.zeros(X_mat.shape,dtype=float)
		X_RECODE[:,self.idx_nonsilent] = self._inv_noise_variance_stabilizing_normalization(X_norm_RECODE)
		X_RECODE = np.where(X_RECODE>0,X_RECODE,0)
		self.log_['#silent %ss' % self.unit] = sum(np.sum(X_mat,axis=0)==0)
		self.log_['ell'] = self.recode_.ell
		if self.recode_.ell == self.fast_algorithm_ell_ub:
			warnings.warn("Acceleration error: the value of ell may not be optimal. Set 'fast_algorithm=False' or larger fast_algorithm_ell_ub.\n"
			"Ex. X_new = screcode.RECODE(fast_algorithm=False).fit_transform(X)")
		self.X_trans = np.round(X_mat,decimals=self.decimals)
		self.X_RECODE = X_RECODE
		self.noise_variance_ = np.zeros(X_mat.shape[1],dtype=float)
		self.noise_variance_[self.idx_nonsilent] =  self.noise_var
		self.normalized_variance_ = np.zeros(X_mat.shape[1],dtype=float)
		self.normalized_variance_[self.idx_nonsilent] =  self.X_norm_var
		
		X_RECODE_ss = (np.median(np.sum(X_RECODE[:,self.idx_nonsilent],axis=1))*X_RECODE[:,self.idx_nonsilent].T/np.sum(X_RECODE[:,self.idx_nonsilent],axis=1)).T
		self.cv_ = np.zeros(X.shape[1],dtype=float)
		self.cv_[self.idx_nonsilent] =  np.std(X_RECODE_ss,axis=0)/np.mean(X_RECODE_ss,axis=0)
		
		self.significance_ = np.empty(X.shape[1],dtype=object)
		self.significance_[self.normalized_variance_==0] = 'silent'
		self.significance_[self.normalized_variance_>0] = 'non-significant'
		self.significance_[self.normalized_variance_>1] = 'significant'

		if type(X) == anndata._core.anndata.AnnData:
			X_out = anndata.AnnData.copy(X)
			X_out.obsm['RECODE'] = X_RECODE
			X_out.var['noise_variance_RECODE'] = self.noise_variance_
			X_out.var['normalized_variance_RECODE'] = self.normalized_variance_
			X_out.var['significance_RECODE'] = self.significance_
		else:
			X_out = X_RECODE

		return X_out
	
	def fit_transform(self,X):
		"""
		Fit the model with X and transform X into RECODE-denoised data.

		Parameters
		----------
		X : ndarray/anndata of shape (n_samples, n_features)
			Tranceforming single-cell sequencing data matrix (row:cell, culumn:gene/peak).

		Returns
		-------
		X_new : ndarray/anndata (the same format as input)
			Denoised data matrix.
		"""
		start_time = datetime.datetime.now()
		if self.verbose:
			print('start RECODE for sc%s-seq' % self.seq_target)
		self.fit(X)
		X_RECODE = self.transform(X)
		end_time = datetime.datetime.now()
		elapsed_time = end_time - start_time
		hours, remainder = divmod(elapsed_time.seconds, 3600)
		minutes, seconds = divmod(remainder, 60)
		milliseconds = int(elapsed_time.microseconds / 1000)
		self.elapsed_time =  f"{hours}h {minutes}m {seconds}s"
		self.log_['Elapsed time'] = f"{hours}h {minutes}m {seconds}s {milliseconds:03}ms"
		if self.verbose:
			print('end RECODE for sc%s-seq' % self.seq_target)
			print('log:',self.log_)
		return X_RECODE
	
	def transform_integration(
			self,
			X,
			meta_data,
			vers_use = 'batch',
		):
		"""
		Transform X into RECODE-denoised data.

		Parameters
		----------
		X : ndarray or anndata of shape (n_samples, n_features)
			Single-cell sequencing data matrix (row:cell, culumn:gene/peak).
		
		meta_data : DataFrame (n_samples, *)

		Returns
		-------
		X_new : ndarray of shape (n_samples, n_features)
			RECODE-denoised data matrix.
		"""
		X_mat = self._check_datatype(X)
		if self.fit_idx == False:
			raise TypeError("Run fit before transform.")
		if X_mat.shape[1] != self.d_all:
			raise TypeError("RECODE requires the same dimension as that of fitted data.")
		X_ = X_mat[:,self.idx_nonsilent]
		X_norm = self._noise_variance_stabilizing_normalization(X_)
		X_norm_RECODE = self.recode_.transform(X_norm)
		self.harmony = harmonypy.run_harmony(X_norm_RECODE,meta_data,vers_use,verbose=False)
		X_norm_RECODE_merge = self.harmony.Z_corr.T
		X_RECODE = np.zeros(X_mat.shape,dtype=float)
		X_RECODE[:,self.idx_nonsilent] = self._inv_noise_variance_stabilizing_normalization(X_norm_RECODE_merge)
		X_RECODE = np.where(X_RECODE>0,X_RECODE,0)
		self.log_['#silent %ss' % self.unit] = sum(np.sum(X_mat,axis=0)==0)
		self.log_['ell'] = self.recode_.ell
		if self.recode_.ell == self.fast_algorithm_ell_ub:
			warnings.warn("Acceleration error: the value of ell may not be optimal. Set 'fast_algorithm=False' or larger fast_algorithm_ell_ub.\n"
			"Ex. X_new = screcode.RECODE(fast_algorithm=False).fit_transform(X)")
		self.X_trans = np.round(X_mat,decimals=self.decimals)
		self.X_RECODE = X_RECODE
		self.noise_variance_ = np.zeros(X_mat.shape[1],dtype=float)
		self.noise_variance_[self.idx_nonsilent] =  self.noise_var
		self.normalized_variance_ = np.zeros(X_mat.shape[1],dtype=float)
		self.normalized_variance_[self.idx_nonsilent] =  self.X_norm_var
		
		X_RECODE_ss = (np.median(np.sum(X_RECODE[:,self.idx_nonsilent],axis=1))*X_RECODE[:,self.idx_nonsilent].T/np.sum(X_RECODE[:,self.idx_nonsilent],axis=1)).T
		self.cv_ = np.zeros(X.shape[1],dtype=float)
		self.cv_[self.idx_nonsilent] =  np.std(X_RECODE_ss,axis=0)/np.mean(X_RECODE_ss,axis=0)
		
		self.significance_ = np.empty(X.shape[1],dtype=object)
		self.significance_[self.normalized_variance_==0] = 'silent'
		self.significance_[self.normalized_variance_>0] = 'non-significant'
		self.significance_[self.normalized_variance_>1] = 'significant'

		if type(X) == anndata._core.anndata.AnnData:
			X_out = anndata.AnnData.copy(X)
			X_out.obsm['RECODE'] = X_RECODE
			X_out.var['noise_variance_RECODE'] = self.noise_variance_
			X_out.var['normalized_variance_RECODE'] = self.normalized_variance_
			X_out.var['significance_RECODE'] = self.significance_
		else:
			X_out = X_RECODE

		return X_out

	def fit_transform_integration(
			self,
			X,
			meta_data,
			vers_use = 'batch',
		):
		"""
		Fit the model with X and transform X into RECODE-denoised data.

		Parameters
		----------
		X : ndarray/anndata of shape (n_samples, n_features)
			Tranceforming single-cell sequencing data matrix (row:cell, culumn:gene/peak).

		Returns
		-------
		X_new : ndarray/anndata (the same format as input)
			Denoised data matrix.
		"""
		start_time = datetime.datetime.now()
		if self.verbose:
			print('start RECODE for sc%s-seq' % self.seq_target)
		self.fit(X)
		X_RECODE = self.transform_integration(X,meta_data,vers_use)
		end_time = datetime.datetime.now()
		elapsed_time = end_time - start_time
		hours, remainder = divmod(elapsed_time.seconds, 3600)
		minutes, seconds = divmod(remainder, 60)
		milliseconds = int(elapsed_time.microseconds / 1000)
		self.elapsed_time =  f"{hours}h {minutes}m {seconds}s"
		self.log_['Elapsed time'] = f"{hours}h {minutes}m {seconds}s {milliseconds:03}ms"
		if self.verbose:
			print('end RECODE for sc%s-seq' % self.seq_target)
			print('log:',self.log_)
		return X_RECODE

	def check_applicability(
		self,
		title = '',
		figsize=(10,5),
		ps = 2,
		save = False,
		save_filename = 'check_applicability',
		save_format = 'png',
		dpi = None,
		show = True
	):
		"""
		Check the applicability of RECODE. 
		Before using this function, you have to conduct ``fit(X)`` or ``fit_transform(X)`` for the target data matrix ``X``. 
		
		Parameters
		----------
		title : str, default=''
			Figure title.
		
		figsize : 2-tuple of floats, default=(10,5)
			Figure dimension ``(width, height)`` in inches.
		
		ps : float, default=10,
			Point size. 
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default= 'check_applicability',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		X_scaled =(self.X_temp.T/np.sum(self.X_temp,axis=1)).T
		X_norm = self._noise_variance_stabilizing_normalization(self.X_temp)
		norm_var = np.var(X_norm,axis=0,ddof=1)
		x,y = np.mean(X_scaled,axis=0),norm_var
		idx_nonsig, idx_sig = y <= 1, y > 1
		fig = plt.figure(figsize=figsize)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		spec = matplotlib.gridspec.GridSpec(ncols=2, nrows=1,width_ratios=[4, 1],wspace=0.)
		ax0 = fig.add_subplot(spec[0])
		ax0.scatter(x[idx_sig],y[idx_sig],color='b',s=ps,label='significant %s' % self.unit,zorder=2)
		ax0.scatter(x[idx_nonsig],y[idx_nonsig],color='r',s=ps,label='non-significant %s' % self.unit,zorder=3)
		ax0.axhline(1,color='gray',ls='--',lw=2,zorder=1)
		ax0.set_xscale('log')
		ax0.set_yscale('log')
		ax0.set_title(title,fontsize=14)
		ax0.set_xlabel('Mean of scaled data',fontsize=14)
		ax0.set_ylabel('NVSN variance',fontsize=14)
		ax0.legend(loc='upper left',borderaxespad=0,fontsize=14,markerscale=2).get_frame().set_alpha(0)
		ylim = ax0.set_ylim()
		ax1 = fig.add_subplot(spec[1])
		sns.kdeplot(y=np.log10(norm_var[norm_var>0]), color='k',shade=True,ax=ax1)
		ax1.axhline(0,c='gray',ls='--',lw=2,zorder=1)
		ax1.axvline(0,c='k',ls='-',lw=1,zorder=1)
		ax1.set_ylim(np.log10(ax0.set_ylim()))
		ax1.tick_params(labelbottom=True,labelleft=False,bottom=True)
		ax1.set_xlabel('Density',fontsize=14)
		ax1.spines['right'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.tick_params(left=False)
		ax1.patch.set_alpha(0)
		#
		x = np.linspace(ax1.set_ylim()[0],ax1.set_ylim()[1],1000)
		dens = scipy.stats.kde.gaussian_kde(np.log10(norm_var[norm_var>0]))(x)
		peak_val = x[np.argmax(dens)]
		rate_low_var = np.sum(norm_var[norm_var>0] < 0.90)/len(norm_var[norm_var>0])
		applicability = 'Unknown'
		backcolor = 'w'
		if (rate_low_var < 0.01) and (np.abs(peak_val)<0.1):
			applicability = 'Class A (strongly applicable)'
			backcolor = 'lightgreen'
		elif rate_low_var < 0.01:
			applicability = 'Class B (weakly applicable)'
			backcolor = 'yellow'
		else:
			applicability = 'Class C (inapplicabile)'
			backcolor = 'tomato'
		ax0.text(0.99, 0.982,applicability,va='top',ha='right', transform=ax0.transAxes,fontsize=14,backgroundcolor=backcolor)
		self.log_['Applicability'] = applicability
		self.log_['Rate of 0 < normalized variance < 0.9'] = "{:.0%}".format(rate_low_var)
		self.log_['Peak density of normalized variance'] = 10**peak_val
		if self.verbose:
			print('applicabity:',applicability)
		if save:
			plt.savefig('%s.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()
		
	
	def plot_procedures(
		self,
		titles = ('Original data','Normalized data','Projected data','Variance-modified data','Denoised data'),
		figsize=(7,5),
		save = False,
		save_filename = 'RECODE_procedures',
		save_filename_foots = ('1_Original','2_Normalized','3_Projected','4_Variance-modified','5_Denoised'),
		save_format = 'png',
		dpi=None,
		show=True
	):
		"""
		Plot procedures of RECODE. The vertical axes of feature are sorted by the mean. 
		
		Parameters
		----------
		titles : 5-tuple of str, default=('Original data','Normalized data','Projected data','Variance-modified data','Denoised data')
			Figure titles.
		
		figsize : 2-tuple of floats, default=(7,5)
			Figure dimension ``(width, height)`` in inches.
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default='RECODE_procedures',
			File name (path) of save figure (head). 
			
		save_filename_foots : 5-tuple of str, default=('1_Original','2_Normalized','3_Projected','4_Variance-modified','5_Denoised'),
			File name (path) of save figure (foot). 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		
		if self.seq_target=='ATAC':
			title = 'ATAC preprocessing'
			foot = '0_ATAC_preprocessing'
			self.plot_ATAC_preprocessing(title=title,figsize=figsize,
				save = save,save_filename = '%s_%s' % (save_filename,foot),
				save_format = save_format,dpi=dpi,show=show)

		self.plot_original_data(title=titles[0],figsize=figsize,
			save = save,save_filename = '%s_%s' % (save_filename,save_filename_foots[0]),
			save_format = save_format,dpi=dpi,show=show)
		 
		self.plot_normalized_data(title=titles[1],figsize=figsize,
			save = save,save_filename = '%s_%s' % (save_filename,save_filename_foots[1]),
			save_format = save_format,dpi=dpi,show=show)
		
		self.plot_projected_data(title=titles[2],figsize=figsize,
			save = save,save_filename = '%s_%s' % (save_filename,save_filename_foots[2]),
			save_format = save_format,dpi=dpi,show=show)
		
		self.plot_variance_modified_data(title=titles[3],figsize=figsize,
			save = save,save_filename = '%s_%s' % (save_filename,save_filename_foots[3]),
			save_format = save_format,dpi=dpi,show=show)
		
		self.plot_denoised_data(title=titles[4],figsize=figsize,
			save = save,save_filename = '%s_%s' % (save_filename,save_filename_foots[4]),
			save_format = save_format,dpi=dpi,show=show)
	
	def report(
		self,
		figsize=(8.27,11.69),
		save = False,
		save_filename = 'report',
		save_format = 'png',
		dpi = None,
		show = True
	):
		"""
		Check the applicability of RECODE. 
		Before using this function, you have to conduct ``fit(X)`` or ``fit_transform(X)`` for the target data matrix ``X``. 
		
		Parameters
		----------
		title : str, default=''
			Figure title.
		
		figsize : 2-tuple of floats, default=(8.27,11.69)
			Figure dimension ``(width, height)`` in inches.
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default= 'report',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		fs_label = 14
		X_scaled =(self.X_temp.T/np.sum(self.X_temp,axis=1)).T
		X_norm = self._noise_variance_stabilizing_normalization(self.X_temp)
		norm_var = np.var(X_norm,axis=0,ddof=1)
		size_factor = np.median(np.sum(self.X_trans,axis=1))
		X_ss_log = np.log2(size_factor*(self.X_trans[:,self.idx_nonsilent].T/np.sum(self.X_trans,axis=1)).T+1)
		X_RECODE_ss_log = np.log2(size_factor*(self.X_RECODE[:,self.idx_nonsilent].T/np.sum(self.X_RECODE,axis=1)).T+1)
		plot_EV = self.recode_.PCA_Ev[self.recode_.PCA_Ev>0]
		n_EV = len(plot_EV)
		plot_EV_mod = np.zeros(n_EV)
		plot_EV_mod[:self.recode_.ell] = self.recode_.PCA_Ev_NRM[:self.recode_.ell]
		#
		fig = plt.figure(figsize=(8.27,11.69))
		plt.rcParams["xtick.direction"] = "in"
		plt.rcParams["ytick.direction"] = "in"
		plt.subplots_adjust(left=0.05, right=0.97, bottom=0.01, top=0.98)
		gs_master = GridSpec(nrows=200, ncols=100,wspace=0,hspace=0)
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[8,0:50])
		ax = fig.add_subplot(gs[0,0])
		ax.text(0,0,'RECODE report',fontsize=25,fontweight='bold')
		ax.axis("off")
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[12:28,0:100])
		ax = fig.add_subplot(gs[0,0])
		ax.patch.set_facecolor('gainsboro')
		ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)         
		ax.tick_params(bottom=False,left=False,right=False,top=False)
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[8,50:100])
		ax = fig.add_subplot(gs[0,0])
		now = datetime.datetime.today()
		ax.text(1,0,'Date: %d-%02d-%02d %02d:%02d:%02d' % (now.year,now.month,now.day,now.hour,now.minute,now.second),fontsize=12,ha='right')
		ax.axis("off")
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[16:26,2:30])
		ax = fig.add_subplot(gs[0,0])
		ax.text(0,1,'Method: %s' % self.log_['seq_target'],fontsize=12)
		ax.text(0,0.5,'nCells: %s' % self.n_all,fontsize=12)
		ax.text(0,0.0,'n%ss: %s' % (self.Unit,self.d_all),fontsize=12)
		# ax.text(0,0.0,'Method: %s' % self.log_['seq_target'],fontsize=12)
		ax.axis("off")
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[16:26,25:64])
		ax = fig.add_subplot(gs[0,0])
		ax.text(0,1,'#significant %ss: %s' % (self.unit,self.log_['#significant %ss' % self.unit]),fontsize=12)
		ax.text(0,0.5,'#non-significant %ss: %s' % (self.unit,self.log_['#non-significant %ss' % self.unit]),fontsize=12)
		ax.text(0,0.0,'#silent %ss: %s' % (self.unit,self.log_['#silent %ss' % self.unit]),fontsize=12)
		ax.axis("off")
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[16:26,64:100])
		ax = fig.add_subplot(gs[0,0])
		ax.text(0,1.0,r'Essential dimension $\ell$: %s' % self.log_['ell'],fontsize=12)
		ax.text(0,0.5,r'Elapsed time: %s' % self.elapsed_time,fontsize=12)
		ax.axis("off")
		#
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[34,0])
		ax = fig.add_subplot(gs[0,0])
		ax.text(0,0.,'Applicability',fontsize=16,fontweight='bold')
		ax.axis("off")
		#
		ps = 2
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=2,width_ratios=[4, 1],subplot_spec=gs_master[37:85,8:100],wspace=0.)
		ax0 = fig.add_subplot(gs[0,0])
		x,y = np.mean(X_scaled,axis=0),norm_var
		idx_nonsig, idx_sig = y <= 1, y > 1
		ax0.scatter(x[idx_sig],y[idx_sig],color='b',s=ps,label='significant %ss' % self.unit,zorder=2)
		ax0.scatter(x[idx_nonsig],y[idx_nonsig],color='r',s=ps,label='non-significant %ss' % self.unit,zorder=3)
		ax0.axhline(1,color='gray',ls='--',lw=2,zorder=1)
		ax0.set_xscale('log')
		ax0.set_yscale('log')
		ax0.set_xlabel('Mean of scaled data',fontsize=14)
		ax0.set_ylabel('NVSN variance',fontsize=14)
		ax0.legend(loc='upper left',borderaxespad=0,fontsize=14,markerscale=5,handletextpad=0.).get_frame().set_alpha(0)
		ylim = ax0.set_ylim()
		ax1 = fig.add_subplot(gs[0,1])
		sns.kdeplot(y=np.log10(norm_var[norm_var>0]), color='k',shade=True,ax=ax1)
		ax1.axhline(0,c='gray',ls='--',lw=2,zorder=1)
		ax1.axvline(0,c='k',ls='-',lw=1,zorder=1)
		ax1.set_ylim(np.log10(ax0.set_ylim()))
		ax1.tick_params(labelbottom=True,labelleft=False,bottom=True)
		ax1.set_xlabel('Density',fontsize=14)
		ax1.spines['right'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.tick_params(left=False)
		ax1.patch.set_alpha(0)
		x = np.linspace(ax1.set_ylim()[0],ax1.set_ylim()[1],1000)
		dens = scipy.stats.kde.gaussian_kde(np.log10(norm_var[norm_var>0]))(x)
		peak_val = x[np.argmax(dens)]
		rate_low_var = np.sum(norm_var[norm_var>0] < 0.90)/len(norm_var[norm_var>0])
		applicability = 'Unknown'
		backcolor = 'w'
		if (rate_low_var < 0.01) and (np.abs(peak_val)<0.1):
			applicability = 'Class A (strongly applicable)'
			backcolor = 'lightgreen'
		elif rate_low_var < 0.01:
			applicability = 'Class A (weakly applicable)'
			backcolor = 'yellow'
		else:
			applicability = 'Class C (inapplicabile)'
			backcolor = 'tomato'
		ax0.text(0.99, 0.975,applicability,va='top',ha='right', transform=ax0.transAxes,fontsize=14,backgroundcolor=backcolor)
		#
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[100,0])
		ax = fig.add_subplot(gs[0,0])
		ax.text(0,0.,'PC variance modification/elimination',fontsize=16,fontweight='bold')
		ax.axis("off")
		#
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[105:145,8:80])
		ps = 10
		n_plot = min(n_EV,1000)
		ax = fig.add_subplot(gs[0,0])
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		ax.scatter(np.arange(n_plot)+1,plot_EV[:n_plot],color='lightblue',label='Original',marker='^',s=ps,zorder=1)
		ax.scatter(np.arange(self.recode_.ell)+1,plot_EV_mod[:self.recode_.ell],color='green',label='PC variance \nmodification',s=ps,zorder=2)
		ax.scatter(np.arange(self.recode_.ell,n_plot)+1,plot_EV_mod[self.recode_.ell:n_plot],color='orange',label='PC variance \nelimination',s=ps,zorder=2)
		ax.axhline(0,color='gray',ls='--',zorder=-10)
		ax.axvline(self.recode_.ell,color='gray',ls='--')
		ax.text(self.recode_.ell*1.1,0.3,'$\ell$=%d' % self.recode_.ell,color='k',fontsize=16,ha='left')
		ax.set_xlabel('PC',fontsize=fs_label)
		ax.set_ylabel('PC variance (eigenvalue)',fontsize=fs_label)
		ax.set_yscale('symlog')
		ax.set_xlim([-5,n_plot+5])
		ax.set_ylim([-0.5,max(plot_EV)*1.5])
		ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left',borderaxespad=0,fontsize=12,markerscale=2,handletextpad=0.).get_frame().set_alpha(0)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		#
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=1,subplot_spec=gs_master[155,0])
		ax = fig.add_subplot(gs[0,0])
		ax.text(0,0.,'Mean-variance plot (log-normalized data)',fontsize=16,fontweight='bold')
		ax.axis("off")
		#
		titles=('Original','RECODE')
		ps = 1
		fs_title = 14
		gs = GridSpecFromSubplotSpec(nrows=1,ncols=2,subplot_spec=gs_master[163:200,5:100])
		ax0 = fig.add_subplot(gs[0,0])
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		x,y = np.mean(X_ss_log,axis=0),np.var(X_ss_log,axis=0,ddof=1)
		ax0.scatter(x,y,color='b',s=ps,label='significant %ss' % self.unit,zorder=2)
		ax0.axhline(0,color='gray',ls='--',lw=2,zorder=1)
		ax0.set_xlabel('Mean',fontsize=fs_label)
		ax0.set_ylabel('Variance',fontsize=fs_label)
		ax0.set_title(titles[0],fontsize=fs_title)
		ylim = [-0.05*np.percentile(y,99.9),1.05*np.percentile(y,99.9)]
		ax0.set_ylim(ylim)	
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		ax1 = fig.add_subplot(gs[0,1])
		x,y = np.mean(X_RECODE_ss_log,axis=0),np.var(X_RECODE_ss_log,axis=0,ddof=1)
		ax1.scatter(x,y,color='b',s=ps,label='significant %ss' % self.unit,zorder=2)
		ax1.set_ylim(ylim)
		ax1.axhline(0,color='gray',ls='--',lw=2,zorder=1)
		ax1.set_xlabel('Mean',fontsize=fs_label)
		ax1.set_ylabel('Variance',fontsize=fs_label)
		ax1.set_title(titles[1],fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		if save:
			plt.savefig('%s.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()
	
	def plot_original_data(
			self,
		  title='',
		  figsize=(7,5),
		  save = False,
		  save_filename = 'original_data',
		  save_format = 'png',
		  dpi=None,
		  show = True
	):
		"""
		Plot noise variance for each features.
		
		Parameters
		----------
		title : str, default=''
			Figure title.
		
		figsize : 2-tuple of floats, default=(7,5)
			Figure dimension ``(width, height)`` in inches.
		
		ps : float, default=10,
			Point size. 
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default='original_data',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		ps = 2
		fs_title = 16
		fs_label = 14
		X_nUMI = np.sum(self.X_temp,axis=1)
		X_scaled = (self.X_temp.T/X_nUMI).T
		X_scaled_mean = np.mean(X_scaled,axis=0)
		noise_var = np.mean(self.X_temp.T/X_nUMI/X_nUMI,axis=1)
		noise_var[noise_var==0] = 1
		X_norm = (X_scaled-np.mean(X_scaled,axis=0))/np.sqrt(noise_var)
		fig,ax = plt.subplots(figsize=figsize)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		idx_sort = np.argsort(np.mean(X_scaled,axis=0))
		x = np.arange(X_scaled.shape[1])
		y1 = np.var(X_scaled,axis=0,ddof=1)[idx_sort]
		y2 = noise_var[idx_sort]
		plt1 = ax.scatter(x,y1,color='k',s=ps,label='Original',zorder=1)
		plt2 = ax.scatter(x,y2,color='r',s=ps,label='Noise',zorder=2,marker='x')
		ax.axhline(0,color='gray',ls='--',lw=2,zorder=1)
		ax.set_ylim([min(min(y1),min(y2))*0.5,max(max(y1),max(y2))])
		ax.set_xlabel(self.Unit,fontsize=fs_label)
		ax.set_ylabel('Variance',fontsize=fs_label)
		ax.set_yscale('log')
		ax.legend(loc='upper left',borderaxespad=0,fontsize=14,markerscale=5,handletextpad=0.).get_frame().set_alpha(0)
		plt1.set_alpha(0.1)
		plt2.set_alpha(0.1)
		ax.set_title(title,fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		if save:
			plt.savefig('%s.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()

	def plot_normalized_data(
			self,
		  title='Normalized data',
		  figsize=(7,5),
		  save = False,
		  save_filename = 'noise_variance',
		  save_format = 'png',
		  dpi=None,
		  show = True
	):
		"""
		Plot the transformed data by the noise variance-stabilizing normalization.
		
		Parameters
		----------
		title : str, default=''
			Figure title.
		
		figsize : 2-tuple of floats, default=(7,5)
			Figure dimension ``(width, height)`` in inches.
		
		ps : float, default=10,
			Point size. 
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default= 'noise_variance',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		ps = 1
		fs_title = 16
		fs_label = 14
		X_nUMI = np.sum(self.X_temp,axis=1)
		X_scaled = (self.X_temp.T/X_nUMI).T
		X_scaled_mean = np.mean(X_scaled,axis=0)
		noise_var = np.mean(self.X_temp.T/X_nUMI/X_nUMI,axis=1)
		noise_var[noise_var==0] = 1
		X_norm = (X_scaled-np.mean(X_scaled,axis=0))/np.sqrt(noise_var)
		fig,ax = plt.subplots(figsize=figsize)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		idx_sort = np.argsort(np.mean(X_scaled,axis=0))
		x = np.arange(X_scaled.shape[1])
		ax.scatter(x,np.var(X_norm,axis=0,ddof=1)[idx_sort],color='k',s=ps,zorder=2)
		ax.axhline(1,color='r',ls='--')
		ax.set_xlabel(self.Unit,fontsize=fs_label)
		ax.set_ylabel('NVSN variance',fontsize=fs_label)
		ax.set_yscale('log')
		ax.set_title(title,fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		if save:
			plt.savefig('%s.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()

	def plot_projected_data(
			self,
		  title='Projected data',
		  figsize=(7,5),
		  save = False,
		  save_filename = 'noise_variance',
		  save_format = 'png',
		  dpi=None,
		  show = True
	):
		"""
		Plot projected data.
		
		Parameters
		----------
		title : str, default=''
			Figure title.
		
		figsize : 2-tuple of floats, default=(7,5)
			Figure dimension ``(width, height)`` in inches.
		
		ps : float, default=10,
			Point size. 
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default= 'noise_variance',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		ps = 20
		fs_title = 16
		fs_label = 14
		
		plot_EV = self.recode_.PCA_Ev[self.recode_.PCA_Ev>0]
		n_EV = len(plot_EV)
		X_scaled = (self.X_temp.T/np.sum(self.X_temp,axis=1)).T
		idx_sort = np.argsort(np.mean(X_scaled,axis=0))

		fig,ax = plt.subplots(figsize=figsize)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		idx_sort = np.argsort(np.mean(X_scaled,axis=0))
		x = np.arange(X_scaled.shape[1])
		ax.scatter(np.arange(n_EV)+1,plot_EV,color='k',label='Original',s=ps,zorder=1)
		ax.set_xlabel('PC',fontsize=fs_label)
		ax.set_ylabel('PC variance (eigenvalue)',fontsize=fs_label)
		ax.set_yscale('symlog')
		ax.set_ylim([0,max(plot_EV)*1.5])
		ax.set_title(title,fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		if save:
			plt.savefig('%s.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()
		
	def plot_variance_modified_data(
			self,
		  title='Variance-modified data',
		  figsize=(7,5),
		  save = False,
		  save_filename = 'noise_variance',
		  save_format = 'png',
		  dpi=None,
		  show = True
	):
		"""
		Plot varainces (eigenvalues) of the variance-modified data.
		
		Parameters
		----------
		title : str, default=''
			Figure title.
		
		figsize : 2-tuple of floats, default=(7,5)
			Figure dimension ``(width, height)`` in inches.
		
		ps : float, default=10,
			Point size. 
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default= 'noise_variance',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		ps = 20
		fs_title = 16
		fs_label = 14
		
		plot_EV = self.recode_.PCA_Ev[self.recode_.PCA_Ev>0]
		n_EV = len(plot_EV)
		plot_EV_mod = np.zeros(n_EV)
		plot_EV_mod[:self.recode_.ell] = self.recode_.PCA_Ev_NRM[:self.recode_.ell]
		X_scaled = (self.X_temp.T/np.sum(self.X_temp,axis=1)).T
		idx_sort = np.argsort(np.mean(X_scaled,axis=0))
		
		fig,ax = plt.subplots(figsize=figsize)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		idx_sort = np.argsort(np.mean(X_scaled,axis=0))
		x = np.arange(X_scaled.shape[1])
		ax.scatter(np.arange(n_EV)+1,plot_EV,color='lightblue',label='Original',marker='^',s=ps,zorder=1)
		ax.scatter(np.arange(n_EV)+1,plot_EV_mod,color='k',label='Modified',s=ps,zorder=2)
		ax.axvline(self.recode_.ell,color='gray',ls='--')
		ax.text(self.recode_.ell*1.1,0.2,'$\ell$=%d' % self.recode_.ell,color='k',fontsize=16,ha='left')
		ax.set_xlabel('PC',fontsize=fs_label)
		ax.set_ylabel('PC variance (eigenvalue)',fontsize=fs_label)
		#ax.set_xscale('log')
		ax.set_yscale('symlog')
		ax.set_ylim([0,max(plot_EV)*1.5])
		ax.legend(loc='upper right',borderaxespad=0,fontsize=14,markerscale=2,handletextpad=0.).get_frame().set_alpha(0)
		ax.set_title(title,fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		if save:
			plt.savefig('%s.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()
	
	def plot_denoised_data(
			self,
		  title='',
		  figsize=(7,5),
		  save = False,
		  save_filename = 'noise_variance',
		  save_format = 'png',
		  dpi=None,
		  show = True
	):
		"""
		Plot variances of the denoised data.
		
		Parameters
		----------
		title : str, default=''
			Figure title.
		
		figsize : 2-tuple of floats, default=(7,5)
			Figure dimension ``(width, height)`` in inches.
		
		ps : float, default=10,
			Point size. 
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default= 'noise_variance',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		ps = 2
		fs_title = 16
		fs_label = 14
		X_scaled = (self.X_temp.T/np.sum(self.X_temp,axis=1)).T
		X_RECODE_scaled = (self.X_RECODE[:,self.idx_nonsilent].T/np.sum(self.X_RECODE,axis=1)).T
		idx_sort = np.argsort(np.mean(X_scaled,axis=0))
		fig,ax = plt.subplots(figsize=figsize)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		x = np.arange(X_scaled.shape[1])
		y1 = np.var(X_scaled,axis=0,ddof=1)[idx_sort]
		y2 = np.var(X_RECODE_scaled,axis=0,ddof=1)[idx_sort]
		plt1 = ax.scatter(x,y1,color='lightblue',s=ps,label='Original',zorder=1,marker='^')
		plt2 = ax.scatter(x,y2,color='k',s=ps,label='Denoised',zorder=2,marker='o')
		ax.set_ylim([min(min(y1),min(y2))*0.5,max(max(y1),max(y2))])
		ax.set_yscale('log')
		ax.set_xlabel(self.Unit,fontsize=fs_label)
		ax.set_ylabel('Variance',fontsize=fs_label)
		ax.legend(loc='upper left',borderaxespad=0,fontsize=14,markerscale=7,handletextpad=0.).get_frame().set_alpha(0)
		plt1.set_alpha(0.05)
		plt2.set_alpha(0.1)
		ax.set_title(title,fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		if save:
			plt.savefig('%s.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()
	
	def plot_mean_variance(
		self,
		titles=('Original','RECODE'),
		figsize=(7,5),
		ps = 2,
		size_factor = 'median',
		save = False,
		save_filename = 'plot_mean_variance',
		save_format = 'png',
		dpi=None,
	  show = True
	):
		"""
		Plot mean vs variance of features for log-normalized data
		
		Parameters
		----------
		titles : tuple, default=('Original','RECODE')
			Figure title.
		
		figsize : 2-tuple of floats, default=(7,5)
			Figure dimension ``(width, height)`` in inches.
		
		ps : float, default=10,
			Point size. 
		
		size_factor : float or {'median','mean'}, default='median',
			Size factor (total count constant of each cell before the log-normalization). 
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default= 'check_applicability',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		fs_label = 14
		fs_title = 14
		if size_factor=='median':
			size_factor = np.median(np.sum(self.X_trans,axis=1))
			size_factor_RECODE = np.median(np.sum(self.X_RECODE,axis=1))
		elif size_factor=='mean':
			size_factor = np.mean(np.sum(self.X_trans,axis=1))
			size_factor_RECODE = np.mean(np.sum(self.X_REECODE,axis=1))
		elif (type(size_factor) == int) | (type(size_factor) == float):
			size_factor_RECODE = size_factor
		else:
			size_factor = np.median(np.sum(self.X_trans,axis=1))
			size_factor_RECODE = np.median(np.sum(self.X_RECODE,axis=1))
		X_ss_log = np.log2(size_factor*(self.X_trans[:,self.idx_nonsilent].T/np.sum(self.X_trans,axis=1)).T+1)
		X_RECODE_ss_log = np.log2(size_factor_RECODE*(self.X_RECODE[:,self.idx_nonsilent].T/np.sum(self.X_RECODE,axis=1)).T+1)
		fig,ax0 = plt.subplots(figsize=figsize)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		x,y = np.mean(X_ss_log,axis=0),np.var(X_ss_log,axis=0,ddof=1)
		ax0.scatter(x,y,color='b',s=ps,label='significant %ss' % self.unit,zorder=2)
		ax0.axhline(0,color='gray',ls='--',lw=2,zorder=1)
		ax0.set_xlabel('Mean of log-scaled data',fontsize=fs_label)
		ax0.set_ylabel('Variance of log-scaled data',fontsize=fs_label)
		ax0.set_title(titles[0],fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		if save:
			plt.savefig('%s_Original.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		fig,ax1 = plt.subplots(figsize=figsize)
		x,y = np.mean(X_RECODE_ss_log,axis=0),np.var(X_RECODE_ss_log,axis=0,ddof=1)
		ax1.scatter(x,y,color='b',s=ps,label='significant %ss' % self.unit,zorder=2)
		#ax1.set_ylim(ax0.set_ylim())
		ax1.axhline(0,color='gray',ls='--',lw=2,zorder=1)
		ax1.set_xlabel('Mean of log-scaled data',fontsize=fs_label)
		ax1.set_ylabel('Variance of log-scaled data',fontsize=fs_label)
		ax1.set_title(titles[1],fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		if save:
			plt.savefig('%s_RECODE.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()

	def plot_mean_cv(
		self,
		titles=('Original','RECODE'),
		figsize=(7,5),
		ps = 2,
		save = False,
		save_filename = 'plot_mean_cv',
		save_format = 'png',
		dpi=None,
		show_features = False,
		n_show_features = 10,
		cut_detect_rate = 0.005,
		index = None,
	  show = True
	):
		"""
		Plot mean vs variance of features for log-normalized data
		
		Parameters
		----------
		title : str, default=''
			Figure title.
		
		figsize : 2-tuple of floats, default=(7,5)
			Figure dimension ``(width, height)`` in inches.
		
		ps : float, default=10,
			Point size. 
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default= 'check_applicability',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi : float or None, default=None
			Dots per inch.
		
		show_features : float or None, default=False,
			If True
		
		n_show_features : float, default=10,
		
		cut_detect_rate : float, default=0.005,
		
		index : array-like of shape (n_features,) or None, default=None,
			
		
		"""
		fs_label = 14
		fs_title = 14
		X_ss = (np.median(np.sum(self.X_trans[:,self.idx_nonsilent],axis=1))*self.X_trans[:,self.idx_nonsilent].T/np.sum(self.X_trans[:,self.idx_nonsilent],axis=1)).T
		fig,ax0 = plt.subplots(figsize=figsize)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		x = np.mean(X_ss,axis=0)
		cv = np.std(X_ss,axis=0)/np.mean(X_ss,axis=0)
		ax0.scatter(x,cv,color='b',s=ps,zorder=2)
		ax0.axhline(0,color='gray',ls='--',lw=2,zorder=1)
		ax0.set_xscale('log')
		ax0.set_xlabel('Mean',fontsize=fs_label)
		ax0.set_ylabel('Coefficient of variation',fontsize=fs_label)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		ax0.set_title(titles[0],fontsize=fs_title)
		if save:
			plt.savefig('%s_Original.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		
		X_RECODE_ss = (np.median(np.sum(self.X_RECODE[:,self.idx_nonsilent],axis=1))*self.X_RECODE[:,self.idx_nonsilent].T/np.sum(self.X_RECODE[:,self.idx_nonsilent],axis=1)).T
		fig,ax1 = plt.subplots(figsize=figsize)
		x = np.mean(X_RECODE_ss,axis=0)
		cv = np.std(X_RECODE_ss,axis=0)/np.mean(X_RECODE_ss,axis=0)
		#ax1.set_ylim(ax0.set_ylim())
		ax1.axhline(0,color='gray',ls='--',lw=2,zorder=1)
		ax1.set_xscale('log')
		ax1.set_xlabel('Mean',fontsize=fs_label)
		ax1.set_ylabel('Coefficient of variation',fontsize=fs_label)
		ax1.set_title(titles[1],fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		
		if show_features:
			if len(index) != self.X.shape[1]:
				warnings.warn("Warning: no index opotion or length of index did not fit X.shape[1]. Use feature numbers")
				index = np.arange(self.X.shape[1])+1
			detect_rate = np.sum(np.where(self.X>0,1,0),axis=0)[self.idx_nonsilent]/self.X.shape[0]
			idx_detect_rate_n = detect_rate <= cut_detect_rate
			idx_detect_rate_p = detect_rate >  cut_detect_rate
			ax1.scatter(x[idx_detect_rate_n],cv[idx_detect_rate_n],color='gray',s=ps,label='detection rate <= {:.2%}'.format(cut_detect_rate),alpha=0.5)
			ax1.scatter(x[idx_detect_rate_p],cv[idx_detect_rate_p],color='b',s=ps,label='detection rate > {:.2%}'.format(cut_detect_rate),alpha=0.5)
			ax1.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15),ncol=2,fontsize=12,markerscale=2)
			idx_rank_cv = np.argsort(cv[idx_detect_rate_p])[::-1]
			texts = [plt.text(x[idx_detect_rate_p][idx_rank_cv[i]],cv[idx_detect_rate_p][idx_rank_cv[i]],index[self.idx_nonsilent][idx_detect_rate_p][idx_rank_cv[i]],color='red') for i in range(n_show_features)]
			adjustText.adjust_text(texts,arrowprops=dict(arrowstyle='->', color='k'))
		else:
			ax1.scatter(x,cv,color='b',s=ps,zorder=2)
			
		if save:
			plt.savefig('%s_RECODE.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()
	
	def plot_ATAC_preprocessing(
		self,
		title='ATAC preprocessing',
		figsize=(7,5),
		ps = 10,
		save = False,
		save_filename = 'plot_ATAC_preprocessing',
		save_format = 'png',
		dpi=None,
	  show=True
	):
		"""
		Plot the number of values in scATAC-seq data matrix with and without preprocessing (odd-even stabilization).
		
		Parameters
		----------
		title : str, default='ATAC preprocessing'
			Figure title.
		
		figsize : 2-tuple of floats, default=(7,5)
			Figure dimension ``(width, height)`` in inches.
		
		ps : float, default=10,
			Point size. 
		
		save : bool, default=False
			If True, save the figure. 
		
		save_filename : str, default= 'plot_ATAC_preprocessing',
			File name (path) of save figure. 
		
		save_format : {'png', 'pdf', 'svg'}, default= 'png',
			File format of save figure. 
		
		dpi: float or None, default=None
			Dots per inch.
		"""
		if self.seq_target != 'ATAC':
			warnings.warn("Error: plot_ATAC_preprocessing is an option of scATAC-seq data")
			return
		ps = 1
		fs_title = 16
		fs_label = 14
		fs_legend = 14
		val,count = np.unique(self.X_trans,return_counts=True)
		idx_even = np.empty(len(val),dtype=bool)
		idx_odd = np.empty(len(val),dtype=bool)
		for i in range(len(val)):
			if i>0 and i%2==0:
				idx_even[i] = True
			else:
				idx_even[i] = False
			if i>0 and i%2==1:
				idx_odd[i] = True
			else:
				idx_odd[i] = False
		plt.figure(figsize=figsize)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		plt.plot(val[1:],count[1:],color='lightblue',zorder=1,marker='^',label='Original')
		val,count = np.unique(self.X_temp,return_counts=True)
		plt.plot(val[1:],count[1:],color='gray',marker='o',label='Preprpcessed',zorder=3)
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel('value',fontsize=fs_label)
		plt.ylabel('count',fontsize=fs_label)
		plt.title(title,fontsize=fs_title)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().spines['top'].set_visible(False)
		plt.legend(fontsize=fs_legend)
		if save:
			plt.savefig('%s.%s' % (save_filename,save_format),dpi=dpi, bbox_inches="tight")
		if show:
			plt.show()	

class RECODE_core():
	def __init__(
		self,
		solver = 'variance',
		variance_estimate = True,
		fast_algorithm = True,
		fast_algorithm_ell_ub = 1000,
		ell_manual = 10,
		ell_min = 3,
		version = 1
	):
		"""
		The core part of RECODE (for non-randam sampling data). 

		Parameters
		----------
		solver : {'variance','manual'}
			If 'variance', regular variance-based algorithm. 
			If 'manual', parameter ell, which identifies essential and noise parts in the PCA space, is manually set. The manual parameter is given by ``ell_manual``. 
		
		variance_estimate : boolean, default=True
			If True and ``solver='variance'``, the parameter estimation method will be conducted. 
		
		fast_algorithm : boolean, default=True
			If True, the fast algorithm is conducted. The upper bound of essential dimension :math:`\ell` is set in ``fast_algorithm_ell_ub``.
		
		fast_algorithm_ell_ub : int, default=1000
			Upper bound of parameter :math:`\ell` for the fast algorithm. Must be of range [1, infinity).
		
		ell_manual : int, default=10
			Manual essential dimension :math:`\ell` computed by ``solver='manual'``. Must be of range [1, infinity).
		
		ell_min : int, default=3
			Minimam value of essential dimension :math:`\ell`
		
		version : int default='1'
			Version of RECODE. 
		
		"""
		self.solver = solver
		self.variance_estimate = variance_estimate
		self.fast_algorithm = fast_algorithm
		self.fast_algorithm_ell_ub = fast_algorithm_ell_ub
		self.ell_manual = ell_manual
		self.ell_min = ell_min
		self.fit_idx = False
		self.version = version 
		self.RECODE_done = False
	
	def _noise_reductor(
		self,
		X,
		L,
		U,
		Xmean,
		ell,
		version=1,
		TO_CR = 1
	):
		if version==2 and self.RECODE_done == False:
			U_ell = U[:ell,:]
			# U_ell_mod = np.copy(U_ell)
			L_ell = L[:ell,:ell]
			for i in range(ell):
				# U[i,U_argsort[i][::-1][U_sort[::-1].cumsum() > TO_CR]] = 0
				# idx_order = np.argsort(np.abs(U[i]))[::-1]
				# idx_sparce = np.sort(np.abs(U[i]))[::-1].cumsum() > L_ell[i,i]
				idx_order = np.argsort(U[i]**2)[::-1]
				idx_sparce = np.sort(U[i]**2)[::-1].cumsum() > L_ell[i,i]
				U_ell[i,idx_order[idx_sparce]] = 0
				U_ell[i] = U_ell[i]/np.sqrt(np.sum(U_ell[i]**2))
			#U_ell[i,:] = U[i,:]/np.sqrt(np.sum(U[i]**2))
			return np.dot(np.dot(np.dot(X-Xmean,U_ell.T),L_ell),U_ell)+Xmean
		else:
			U_ell = U[:ell,:]
			L_ell = L[:ell,:ell]
			return np.dot(np.dot(np.dot(X-Xmean,U_ell.T),L_ell),U_ell)+Xmean

	def _noise_reduct_param(
		self,
		X,
		delta = 0.05
	):
		comp = max(np.sum(self.PCA_Ev_NRM>delta*self.PCA_Ev_NRM[0]),3)
		self.ell = min(self.ell_max,comp)
		self.X_RECODE =  self._noise_reductor(X,self.L,self.U,self.X_mean,self.ell,self.version,self.TO_CR)
		return self.X_RECODE
	
	def _noise_reduct_noise_var(
		self,
		X,
		noise_var = 1
	):
		X_RECODE = self._noise_reductor(X,self.L,self.U,self.X_mean,self.ell,self.version,self.TO_CR)

		
		return X_RECODE
	
	def _noise_var_est(
		self,
		X,
		cut_low_exp=1.0e-10
	):
		n,d = X.shape
		X_var = np.var(X,axis=0,ddof=1)
		idx_var_p = np.where(X_var>cut_low_exp)[0]
		X_var_sub = X_var[idx_var_p]
		X_var_min = np.min(X_var_sub)-1.0e-10
		X_var_max = np.max(X_var_sub)+1.0e-10
		X_var_range = X_var_max-X_var_min
		
		div_max = 1000
		num_div_max = int(min(0.1*d,div_max))
		error = np.empty(num_div_max)
		for i in range(num_div_max):
				num_div = i+1
				delta = X_var_range/num_div
				k = np.empty([num_div],dtype=int)
				for j in range(num_div):
					div_min = j*delta+X_var_min
					div_max = (j+1)*delta+X_var_min
					k[j] = len(np.where((X_var_sub<div_max) & (X_var_sub>div_min))[0])
				error[i] = (2*np.mean(k)-np.var(k))/delta/delta
		
		opt_div = int(np.argmin(error)+1)

		k = np.empty([opt_div],dtype=int)
		k_index = np.empty([opt_div],dtype=list)
		delta = X_var_range/opt_div
		for j in range(opt_div):
				div_min = j*delta+X_var_min
				div_max = (j+1)*delta+X_var_min
				k[j] = len(np.where((X_var_sub<=div_max) & (X_var_sub>div_min))[0])
		idx_k_max = np.argmax(k)
		div_min = idx_k_max*delta+X_var_min
		div_max = (idx_k_max+1)*delta+X_var_min
		idx_set_k_max = np.where((X_var_sub<div_max) & (X_var_sub>div_min))[0]
		var = np.mean(X_var_sub[idx_set_k_max])
		return var
	
	def fit(
			self,
			X
		):
		"""
		Create the transformation using X.

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features).
			Training data matrix, where ``n_samples`` is the number of samples
			and ``n_features`` is the number of features.
		Returns
		-------
		self : object
			Returns the instance itself.
		"""
		n,d = X.shape
		n_pca = min(n-1,d-1)
		if self.fast_algorithm:
			n_pca = min(n_pca,self.fast_algorithm_ell_ub)
		n_svd,d = X.shape
		X_mean = np.mean(X,axis=0)
		svd = sklearn.decomposition.TruncatedSVD(n_components=n_pca).fit(X-X_mean)
		SVD_Sv = svd.singular_values_
		PCA_Ev = (SVD_Sv**2)/(n_svd-1)
		PCA_Ev_sum_all = np.sum(np.var(X,axis=0,ddof=1))
		PCA_Ev_NRM = np.array(PCA_Ev,dtype=float)
		PCA_Ev_sum_diff = PCA_Ev_sum_all - np.sum(PCA_Ev)
		n_Ev_all = min(n,d)
		PCA_Ev_NRM = np.array([PCA_Ev[i]-(np.sum(PCA_Ev[i+1:])+PCA_Ev_sum_diff)/(n_Ev_all-i-1) for i in range(len(PCA_Ev_NRM)-1)])
		PCA_Ev_NRM = np.append(PCA_Ev_NRM,0)
		PCA_CCR = (PCA_Ev/PCA_Ev_sum_all).cumsum()
		PCA_CCR_NRM = (PCA_Ev_NRM/np.sum(PCA_Ev_NRM)).cumsum()
		PCA_Ev_sum_diff = PCA_Ev_sum_all - np.sum(PCA_Ev)
		PCA_Ev_sum = np.array([np.sum(PCA_Ev[i:]) for i in range(n_pca)])+PCA_Ev_sum_diff
		d_act = sum(np.var(X,axis=0,ddof=1)>0)
		X_var  = np.var(X,axis=0,ddof=1)
		dim = np.sum(X_var>0)
		noise_var = 1
		if self.variance_estimate:
			self.noise_var = self._noise_var_est(X)
		thrshold = (dim-np.arange(n_pca))*noise_var
		if np.sum(PCA_Ev_sum-thrshold<0) == 0:
			warnings.warn("Acceleration error: the optimal value of ell is larger than fast_algorithm_ell_ub. Set larger fast_algorithm_ell_ub than %d or 'fast_algorithm=False'" % self.fast_algorithm_ell_ub)
		comp = np.min(np.arange(n_pca)[PCA_Ev_sum-thrshold<0])
		self.ell_max = np.min([n,d,np.sum(PCA_Ev>1.0e-10)])
		self.ell = comp
		if self.ell > self.ell_max:
			self.ell = self.ell_max
		if self.ell < self.ell_min:
			self.ell = self.ell_max
		#self.ell = np.max(np.min(self.ell_max,comp),self.ell_min)
		self.TO_CR = PCA_CCR[self.ell]
		self.TO_CR_NRM = PCA_CCR_NRM[self.ell]
		self.PCA_Ev = PCA_Ev
		self.PCA_CCR = PCA_CCR
		self.n_pca = n_pca
		self.PCA_Ev_NRM = PCA_Ev_NRM
		self.U = svd.components_
		self.L = np.diag(np.sqrt(self.PCA_Ev_NRM[:self.ell_max]/self.PCA_Ev[:self.ell_max]))
		# self.X_fit = X
		self.X_mean = np.mean(X,axis=0)
		self.PCA_Ev_sum_all = PCA_Ev_sum_all
		self.noise_var = noise_var
		self.fit_idx = True

	def transform(self,X):
		"""
		Apply RECODE to X.

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features).
			Transsforming data matrix, where `n_samples` is the number of samples
			and `n_features` is the number of features.
		Returns
		-------
		X_new : ndarray of shape (n_samples, n_components)
			Denoised data matrix.
		"""
		if self.fit_idx == False:
			raise TypeError("Run fit before transform.")
		if self.solver=='variance':
			return self._noise_reduct_noise_var(X,self.noise_var)
		elif self.solver=='manual':
			self.ell = self.ell_manual
			return self._noise_reductor(X,self.L,self.U,self.X_mean,self.ell,self.version,self.TO_CR)
		
		self.RECODE_done = True

	def fit_transform(self,X):
		"""
		Fit and transform RECODE to X.

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features).
			Transsforming data matrix, where `n_samples` is the number of samples
			and `n_features` is the number of features.
		Returns
		-------
		X_new : ndarray of shape (n_samples, n_components)
			Denoised data matrix.
		"""
		self.fit(X)
		return self.transform(X)
