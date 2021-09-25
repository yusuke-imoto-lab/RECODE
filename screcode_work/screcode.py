import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition
import scipy.sparse
import seaborn as sns


class RECODE():
	def __init__(
		self,
		return_param = False,
		acceleration = True,
		acceleration_ell_max = 100,
		param_estimate = True,
		ell_manual = 10,
	):
		self.acceleration = acceleration
		self.acceleration_ell_max = acceleration_ell_max
		self.param_estimate=param_estimate
		self.ell_manual=ell_manual
		self.return_param=return_param
	
	def fit(self,data):
		n,d = data.shape
		if self.acceleration:
			self.n_pca = min(n-1,d-1)
		else:
			self.n_pca = min(n-1,d-1,self.acceleration_ell_max)
		data_svd = data
		n_svd,d = data_svd.shape
		data_mean = np.mean(data,axis=0)
		data_svd_mean = np.mean(data_svd,axis=0)
		svd = sklearn.decomposition.TruncatedSVD(n_components=self.n_pca).fit(data_svd-data_svd_mean)
		SVD_Sv = svd.singular_values_
		self.PCA_Ev = (SVD_Sv**2)/(n_svd-1)
		self.U = svd.components_
		PCA_Ev_sum_all = np.sum(np.var(data,axis=0))
		PCA_Ev_NRM = np.array(self.PCA_Ev,dtype=float)
		PCA_Ev_sum_diff = PCA_Ev_sum_all - np.sum(self.PCA_Ev)
		n_Ev_all = min(n,d)
		for i in range(len(PCA_Ev_NRM)-1):
			PCA_Ev_NRM[i] -= (np.sum(self.PCA_Ev[i+1:])+PCA_Ev_sum_diff)/(n_Ev_all-i-1)
		self.PCA_Ev_NRM = PCA_Ev_NRM
		self.ell_max = np.sum(self.PCA_Ev>1.0e-10)
		self.L = np.diag(np.sqrt(self.PCA_Ev_NRM[:self.ell_max]/self.PCA_Ev[:self.ell_max]))
		self.data = data
		self.data_mean = np.mean(data,axis=0)
		self.PCA_Ev_sum_all = PCA_Ev_sum_all
	
	def noise_reduct_param(
		self,
		delta = 0.05
	):
		comp = max(np.sum(self.PCA_Ev_NRM>delta*self.PCA_Ev_NRM[0]),3)
		self.ell = min(self.ell_max,comp)
		self.data_RECODE = noise_reductor(self.data,self.L,self.U,self.data_mean,self.ell)
		return self.data_RECODE
		
	def noise_reductor(self,X,L,U,Xmean,ell):
		U_ell = U[:ell,:]
		L_ell = L[:ell,:ell]
		return np.dot(np.dot(np.dot(X-Xmean,U_ell.T),L_ell),U_ell)+Xmean
	
	def noise_reduct_noise_var(
		self,
		noise_var = 1,
		ell_min = 3
	):
		PCA_Ev_sum_diff = self.PCA_Ev_sum_all - np.sum(self.PCA_Ev)
		PCA_Ev_sum = np.array([np.sum(self.PCA_Ev[i:]) for i in range(self.n_pca)])+PCA_Ev_sum_diff
		d_act = sum(np.var(self.data,axis=0)>0)
		data_var  = np.var(self.data,axis=0)
		dim = np.sum(data_var>0)
		thrshold = (dim-np.arange(self.n_pca))*noise_var
		comp = np.sum(PCA_Ev_sum-thrshold>0)
		self.ell = max(min(self.ell_max,comp),ell_min)
		data_RECODE = self.noise_reductor(self.data,self.L,self.U,self.data_mean,self.ell)
		return data_RECODE
	
	def noise_var_est(
		self,
		data,
		cut_low_exp=1.0e-10,
		out_file='variance'
	):
		n,d = data.shape
		data_var = np.var(data,axis=0)
		idx_var_p = np.where(data_var>cut_low_exp)[0]
		data_var_sub = data_var[idx_var_p]
		data_var_min = np.min(data_var_sub)-1.0e-10
		data_var_max = np.max(data_var_sub)+1.0e-10
		data_var_range = data_var_max-data_var_min
		
		div_max = 1000
		num_div_max = int(min(0.1*d,div_max))
		error = np.empty(num_div_max)
		for i in range(num_div_max):
				num_div = i+1
				delta = data_var_range/num_div
				k = np.empty([num_div],dtype=int)
				for j in range(num_div):
					div_min = j*delta+data_var_min
					div_max = (j+1)*delta+data_var_min
					k[j] = len(np.where((data_var_sub<div_max) & (data_var_sub>div_min))[0])
				error[i] = (2*np.mean(k)-np.var(k))/delta/delta
		
		opt_div = int(np.argmin(error)+1)

		k = np.empty([opt_div],dtype=int)
		k_index = np.empty([opt_div],dtype=list)
		delta = data_var_range/opt_div
		for j in range(opt_div):
				div_min = j*delta+data_var_min
				div_max = (j+1)*delta+data_var_min
				k[j] = len(np.where((data_var_sub<=div_max) & (data_var_sub>div_min))[0])
		idx_k_max = np.argmax(k)
		div_min = idx_k_max*delta+data_var_min
		div_max = (idx_k_max+1)*delta+data_var_min
		idx_set_k_max = np.where((data_var_sub<div_max) & (data_var_sub>div_min))[0]
		var = np.mean(data_var_sub[idx_set_k_max])
		return var
	
	def fit_transform(self,data):
		self.fit(data)
		if self.param_estimate:
			noise_var = recode_tools.noise_var_est(data)
		else:
			noise_var = 1
		return self.noise_reduct_noise_var(noise_var)


class scRECODE():
	def __init__(
		self,
		return_param = False,
		acceleration = True,
		acceleration_ell_max = 1000,
		):
		self.return_param = return_param
		self.acceleration = acceleration
		self.acceleration_ell_max = acceleration_ell_max

	def noise_variance_stabilizing_normalization(
		self,
		data,
		return_param=False
	):
		## scaled data
		data_nUMI = np.sum(data,axis=1)
		data_scaled = (data.T/data_nUMI).T
		data_scaled_mean = np.mean(data_scaled,axis=0)
		## normalization
		noise_var = np.mean(data.T/data_nUMI/data_nUMI,axis=1)
		noise_var[noise_var==0] = 1
		data_norm = (data_scaled-np.mean(data_scaled,axis=0))/np.sqrt(noise_var)
		self.data_nUMI = data_nUMI
		self.data_scaled = data_scaled
		self.data_scaled_mean = data_scaled_mean
		self.noise_var = noise_var
		if return_param == True:
			data_norm_var = np.var(data_norm,axis=0)
			n_sig = sum(data_norm_var>1)
			n_silent = sum(np.sum(data,axis=0)==0)
			n_nonsig = data.shape[1] - n_sig - n_silent
			param = {
				'#significant genes':n_sig,
				'#non-significant genes':n_nonsig,
				'#silent genes':n_silent
			}
			return data_norm, param
		else:
			return data_norm
	
	def inv_noise_variance_stabilizing_normalization(
		self,
		data
	):
		data_norm_inv_temp = data*np.sqrt(self.noise_var)+self.data_scaled_mean
		data_norm_inv = (data_norm_inv_temp.T*self.data_nUMI).T
		return data_norm_inv
	
	def fit(self,data):
		self.data = data
		self.idx_gene = np.sum(data,axis=0) > 0
		self.data_temp = data[:,self.idx_gene]

	def fit_transform(self,data):
		self.fit(data)
		param = {}
		data_norm,param_t = self.noise_variance_stabilizing_normalization(self.data_temp,return_param=True)
		param.update(param_t)
		recode = RECODE(return_param=True,param_estimate=False,acceleration=self.acceleration,acceleration_ell_max=self.acceleration_ell_max)
		data_norm_RECODE = recode.fit_transform(data_norm)
		data_scRECODE_scaled = self.inv_noise_variance_stabilizing_normalization(data_norm_RECODE)
		data_scRECODE = np.zeros(data.shape,dtype=float)
		data_scRECODE[:,self.idx_gene] = (data_scRECODE_scaled.T*np.sum(self.data_temp,axis=1)).T
		data_scRECODE = np.where(data_scRECODE>0,data_scRECODE,0)
		param['#silent genes'] = sum(np.sum(data,axis=0)==0)
		param['ell'] = recode.ell
		if self.return_param:
			return data_scRECODE, param
		else:
			return data_scRECODE
	
	def check_applicability(
			self,
		  title='',
		  figsize=(10,5),
		  ps = 10
	):
		data_scaled =(self.data_temp.T/np.sum(self.data_temp,axis=1)).T
		data_norm = self.noise_variance_stabilizing_normalization(self.data_temp)
		x,y = np.mean(data_scaled,axis=0),np.var(data_norm,axis=0)
		idx_nonsig, idx_sig = y <= 1, y > 1
		fig = plt.figure(figsize=figsize)
		spec = matplotlib.gridspec.GridSpec(ncols=2, nrows=1,width_ratios=[4, 1],wspace=0)
		ax0 = fig.add_subplot(spec[0])
		ax0.scatter(x[idx_sig],y[idx_sig],color='b',s=ps,label='significant genes',zorder=2)
		ax0.scatter(x[idx_nonsig],y[idx_nonsig],color='r',s=ps,label='non-significant genes',zorder=3)
		ax0.axhline(1,color='gray',ls='--',lw=2,zorder=1)
		ax0.set_xscale('log')
		ax0.set_yscale('log')
		ax0.set_title(title,fontsize=14)
		ax0.set_xlabel('Mean of scaled data',fontsize=14)
		ax0.set_ylabel('Variance of normalized data',fontsize=14)
		ax0.legend(loc='upper left',borderaxespad=0,fontsize=14,markerscale=2).get_frame().set_alpha(0)
		ylim = ax0.set_ylim()
		ax1 = fig.add_subplot(spec[1])
		sns.kdeplot(y=np.log10(y), color='k',shade=True,ax=ax1)
		ax1.axhline(0,c='gray',ls='--',lw=2,zorder=1)
		ax1.set_ylim(np.log10(ax0.set_ylim()))
		ax1.tick_params(labelbottom=True,labelleft=False,bottom=True)
		ax1.set_xlabel('Density',fontsize=14)
		ax1.spines['right'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.tick_params(left=False)
		plt.show()


def scRECODE_h5(h5_file, decimals=5):
    input_h5 = h5py.File(h5_file, 'r')
    data = scipy.sparse.csc_matrix((input_h5['matrix']['data'], input_h5['matrix']['indices'],
                                   input_h5['matrix']['indptr']), shape=input_h5['matrix']['shape']).toarray().T
    gene_list = [x.decode('ascii', 'ignore')
                 for x in input_h5['matrix']['features']['name']]
    cell_list = np.array([x.decode('ascii', 'ignore')
                         for x in input_h5['matrix']['barcodes']], dtype=object)
    data_xRECODE,param = scRECODE(return_param=True).fit_transform(data)
    data_xRECODE_csc = scipy.sparse.csc_matrix(
        np.round(data_xRECODE, decimals=decimals).T)
    outpur_h5 = '%s_xRECODE.h5' % (h5_file[:-3])
    with h5py.File(outpur_h5, 'w') as f:
        f.create_group('matrix')
        for key in input_h5['matrix'].keys():
            if key == 'data':
                f['matrix'].create_dataset(key, data=data_xRECODE_csc.data)
            elif key == 'indices':
                f['matrix'].create_dataset(key, data=data_xRECODE_csc.indices)
            elif key == 'indptr':
                f['matrix'].create_dataset(key, data=data_xRECODE_csc.indptr)
            elif key == 'shape':
                f['matrix'].create_dataset(key, data=data_xRECODE_csc.shape)
            elif type(input_h5['matrix'][key]) == h5py._hl.dataset.Dataset:
                f['matrix'].create_dataset(key, data=input_h5['matrix'][key])
            else:
                f['matrix'].create_group(key)
                for key_sub in input_h5['matrix'][key].keys():
                    f['matrix'][key].create_dataset(
                        key_sub, data=input_h5['matrix'][key][key_sub])
        f['matrix'].create_dataset('scRECODE parameters', data=param)
