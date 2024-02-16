#!/usr/bin/env python

# Folders
data_dir = '/home/surface3/afendric/RESULTS/CE_DYNAM'

# Loads packages
from osgeo import gdal
import math, os, glob, sys, numpy as np, scipy.sparse.linalg, scipy.optimize, datetime, pandas as pd, cPickle as pickle, subprocess, backports.lzma
# import cupy as cp
# from cupyx.scipy.sparse import csr_matrix as csr_gpu

import warnings
warnings.filterwarnings('ignore')
#gdal.PushErrorHandler('CPLQuietErrorHandler')

# Defines the soil pools
soil_pools = ['a', 's', 'p'] # Active, Slow and Passive pools
len_poo = len(soil_pools)

# Defines if we will run with or without erosion/deposition
set_eros = sys.argv[1] # Erosion ('y') or no erosion ('n')
set_depo = sys.argv[2] # Deposition ('y') or no deposition ('n')

years = range(1860, 2061)
if(len(sys.argv) == 4):
	years = range(int(sys.argv[3]), 2061)
months = range(12)
mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

len_soi = 3 # The number of soil layers
soil_layers = range(len_soi)

# Then, changes the output path according to the decision
out_dir = data_dir + '/output/Simulations/'
if(set_eros == 'y' and set_depo == 'y'):
	out_dir_app = 'Yeros-Ydepo_WithoutCCs_EF1/'
elif(set_eros == 'n' and set_depo == 'n'):
	out_dir_app = 'Neros-Ndepo_WithoutCCs_EF1/'
else:
	sys.exit()

# First, load the PFT dataset and extracts the number of rows, columns and layers (i.e., nr, nc and len_pft, respectively)
pft = gdal.Open(data_dir + '/input/landcover/landcover_1860.tif')
len_pft = pft.RasterCount # The number of PFTs used equals the number of layers
nc = pft.RasterXSize
nr = pft.RasterYSize

# Loads the study area mask, the hillslope fraction, and the slope
mask = gdal.Open(data_dir + '/input/others/Mask.tif')
slop = gdal.Open(data_dir + '/input/others/Slope.tif')

# START: Not working in OBELIX
## Function for GPU sparse matrix multiplication
#def spm(A_gpu, b, x, n):
#	b_gpu = cp.array(b)
#	x_gpu = cp.array(x)
#	for j in range(n):
#		x_gpu = x_gpu + b_gpu - A_gpu.dot(x_gpu)
#	return cp.asnumpy(x_gpu)
# END: Not working in OBELIX
def spm(A_gpu, b, x, n):
	b_gpu = b
	x_gpu = x
	for j in range(n):
		x_gpu = x_gpu + b_gpu - A_gpu.dot(x_gpu)
	return x_gpu

## Function to load files
def getff(mk, yk):
	name = out_dir + out_dir_app + 'FL-' + str(yk) +'-%02d' % (mk + 1) + '-'
	A_exp = lzma_open(name + 'A_cut.npz')

	A_carbon = A_exp[0].tocsr()
	A_lateral = A_exp[1].tocsr()
	A_vertical = A_exp[2].tocsr()

	A_cut = A_carbon + A_lateral + A_vertical
	A_cut = A_cut.astype('float32')
	A_cut.data = A_cut.data * 1./1e8

	B_cut = lzma_open(name + 'B_cut.npz')
	B_cut = B_cut[0] + B_cut[1]
	fi = lzma_open(name + 'fi.npz')

	return [A_cut, B_cut, fi]

def lzma_open(fname):
	out = pickle.load(backports.lzma.open(fname, 'rb'))
	return out

def ratio_calc(x, yr_old, yr):
	# There is a small error on the emulator. It does not scale stocks on 01/Jan as ORCHIDEE does.
	# Since this is the only error, we can solve it by simply deriving such ratio.

	xout = 1. * x

	## For the years > 2018, we recycle the data of 2010-18
	df = yr_old - yr
	if yr > 2018:
		yr = 2010 + ((yr - 2019) % 9)
		yr_old = yr + df

	lc = gdal.Open(data_dir + '/input/landcover/landcover_' + str(yr) + '.tif')
	lc_old = gdal.Open(data_dir + '/input/landcover/landcover_' + str(yr_old) + '.tif')
	k = np.arange(nc)

	for i in range(1, nr + 1):
		dlc = lc.ReadAsArray(0, i-1, nc, 1).astype('float64')
		dlc[dlc == lc.GetRasterBand(1).GetNoDataValue()] = 0.
		dlc_old = lc_old.ReadAsArray(0, i-1, nc, 1).astype('float64')
		dlc_old[dlc_old == lc_old.GetRasterBand(1).GetNoDataValue()] = 0.

		dlc = dlc/100.
		dlc_old = dlc_old/100.

		delta_veg = dlc - dlc_old
		delta_veg_sum = np.sum(delta_veg * (delta_veg < 0), axis = 0)
		ratio = (delta_veg < 0) * delta_veg/delta_veg_sum
		ratio[np.isnan(ratio)] = 0.

		delta_veg = delta_veg * (delta_veg > 0)
		
		ncell = (i-1)*nc + k
		for z in range(len_soi):
			x_carb = np.zeros((len_pft, len_poo, nc), dtype = 'float32')
			for j in range(len_pft):
				idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + j*len_poo
				x_carb[j, 0, :] = xout[idx + 0]
				x_carb[j, 1, :] = xout[idx + 1]
				x_carb[j, 2, :] = xout[idx + 2]
			dilu_soil_carbon = np.sum(x_carb * ratio, axis = 0)
			dilu_soil_carbon[np.isnan(dilu_soil_carbon)] = 0.

			for j in range(len_pft):
				idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + j*len_poo
				mds = delta_veg[j, 0, :] > 0

				xout[(idx + 0)[mds]] = ((x_carb[j, 0, :] * dlc_old[j] + dilu_soil_carbon[0] * delta_veg[j])/dlc[j])[0, mds]
				xout[(idx + 1)[mds]] = ((x_carb[j, 1, :] * dlc_old[j] + dilu_soil_carbon[1] * delta_veg[j])/dlc[j])[0, mds]
				xout[(idx + 2)[mds]] = ((x_carb[j, 2, :] * dlc_old[j] + dilu_soil_carbon[2] * delta_veg[j])/dlc[j])[0, mds]
	xout[np.isnan(xout) | np.isinf(xout) | np.isneginf(xout)] = 0.
	return xout

def gen_mats(v, yr, mo):
	print 'Year: ' + str(yr) + ', Month: ' + str(mo) + ', Time: ' + str(datetime.datetime.now())
	sys.stdout.flush()

	if(yr == 1860 and mo == 0):
		x = np.zeros((nc * nr * len_pft * len_soi * len_poo), dtype = 'float32')

		print 'Starting the iterative procedure: - Time: ' + str(datetime.datetime.now())
		Ayk = []
		Byk = []
		fik = []
#		for yk in range(1860, 1870):
		for yk in range(1860, 1866):
			ai = []
			for mm in months: ai.append(yk)
			at = po.map(getff, months, ai)
			for mm in at:
				Ayk.append(mm[0])
				Byk.append(mm[1])
				fik.append(mm[2])

		# Here we try to approximate the final solution before iterating
		# There is not enough memory to do this for FL (and it is also not necessary according to the results only using iteration)
		p = 1e-7 * scipy.sparse.eye(Ayk[0].shape[0])
		p_cut = Ayk[0] + p.tocsr()
		x[fik[0]] = scipy.sparse.linalg.spsolve(p_cut, Byk[0])

#		for l in range(len(Ayk)): Ayk[l] = csr_gpu(Ayk[l])

		it = 1
#		while it < 40: # Iteration criteria
		while it < 100: # Iteration criteria
			it += 1
#			for yk in range(1860, 1870):
			for yk in range(1860, 1866):
				x_old = 1. * x
				for mk in months:
					A_cut = Ayk[(yk-1860)*len(months) + mk]
					B_cut = Byk[(yk-1860)*len(months) + mk]
					fi = fik[(yk-1860)*len(months) + mk]

					x_cut = 1. * x[fi]
#					x_cut = spm(A_gpu, B_cut, x_cut, mdays[mk])
					x_cut = spm(A_cut, B_cut, x_cut, mdays[mk])
					x[fi] = 1. * x_cut

					if mk == 11:
						yr_old = yk

						if(yk < 1869): yr_new = yk + 1
						else: yr_new = 1860

						x = ratio_calc(x, yr_old, yr_new)

				p = np.nansum(abs(x - x_old)/abs(x_old) < 1e-2)/float(np.sum(x > 0)) # Convergence criteria
				# print 'max(x) = ' + str(np.nanmax(x))
				# print 'which.max(x) = ' + str(np.where(x == np.nanmax(x))[0])
				# print 'problem point = ' + str(x[12626177])
		print 'p = ' + str(round(p, 4))
		sys.stdout.flush()
		x[np.isnan(x)] = 0.
		print 'Ending the iterative procedure: - Time: ' + str(datetime.datetime.now())
###
		# Then, we load for mo = 0 only and return
		name = out_dir + out_dir_app + 'FL-' + str(yr) + '-%02d' % (mo + 1) + '-'
		A_exp = lzma_open(name + 'A_cut.npz')

		A_carbon = A_exp[0].tocsr()
		A_lateral = A_exp[1].tocsr()
		A_vertical = A_exp[2].tocsr()

		A_cut = A_carbon + A_lateral + A_vertical
		A_cut = A_cut.astype('float32')
		A_cut.data = A_cut.data * 1./1e8
		B_cut = lzma_open(name + 'B_cut.npz')
		B_cut = B_cut[0] + B_cut[1]
		fi = lzma_open(name + 'fi.npz')

		return fi, A_cut, B_cut, x

	name = out_dir + out_dir_app + 'FL-' + str(yr) + '-%02d' % (mo + 1) + '-'
	A_exp = lzma_open(name + 'A_cut.npz')

	A_carbon = A_exp[0].tocsr()
	A_lateral = A_exp[1].tocsr()
	A_vertical = A_exp[2].tocsr()

	A_cut = A_carbon + A_lateral + A_vertical
	A_cut = A_cut.astype('float32')
	A_cut.data = A_cut.data * 1./1e8
	B_cut = lzma_open(name + 'B_cut.npz')
	B_cut = B_cut[0] + B_cut[1]
	fi = lzma_open(name + 'fi.npz')

	return fi, A_cut, B_cut


## Now, a function to evaluate the model for the whole period
def sq_fun(v):
	if(years[0] > 1860):
		x = np.zeros((nc * nr * len_pft * len_soi * len_poo), dtype = 'float32')

		name = out_dir + out_dir_app + 'FL-' + str(years[0] - 1) + '-12-'
		va = gdal.Open(name + 'a.tif')
		vs = gdal.Open(name + 's.tif')
		vp = gdal.Open(name + 'p.tif')

		k = np.arange(nc)
		for z in range(len_soi):
			for i in range(1, nr + 1):
				ncell = (i-1)*nc + k

				dva = va.ReadAsArray(0, i-1, nc, 1).astype('float64')
				dvs = vs.ReadAsArray(0, i-1, nc, 1).astype('float64')
				dvp = vp.ReadAsArray(0, i-1, nc, 1).astype('float64')

				for j in range(len_pft):
					idx_lyr = z * len_pft + j

					idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + j*len_poo

					x[idx + 0] = dva[idx_lyr, 0, :]
					x[idx + 1] = dvs[idx_lyr, 0, :]
					x[idx + 2] = dvp[idx_lyr, 0, :]
		x[np.isnan(x)] = 0.

		fi, A_cut, B_cut = gen_mats(v, years[0] - 1, 11)
#		A_gpu = csr_gpu(A_cut)
		x_cut = 1. * x[fi]
#		x_cut = spm(A_gpu, B_cut, x_cut, mdays[11])
		x_cut = spm(A_cut, B_cut, x_cut, mdays[11])
		x[fi] = 1. * x_cut
		x = ratio_calc(x, years[0] - 1, years[0])

	# Now, updates the model for the transient period and calculates the cost function
	for yr in years:
		for mo in months:
			if(yr == 1860 and mo == 0):
				fi, A_cut, B_cut, x = gen_mats(v, yr, mo)
			else:
				fi, A_cut, B_cut = gen_mats(v, yr, mo)
#			A_gpu = csr_gpu(A_cut)

			x_av = 1. * x

#			pi = 142
#			pk = 271
#			pcell = (pi - 1) * nc + pk
#			ppp = pcell*len_soi*len_pft*len_poo + 0*len_pft*len_poo + 5*len_poo
#			print x[ppp + 1]

			# For each month, we export the calculations to rasters
			output = {}
			name = out_dir + out_dir_app + 'FL-' + str(yr) + '-%02d' % (mo + 1) + '-' # I add +1 to mo just to avoid confusions with the file name
			for p in soil_pools:
				output['soil' + p + '_FL_out'] = gdal.GetDriverByName('VRT').CreateCopy('tmp5' + set_eros + set_depo + '.vrt', slop)
				ntot = len_pft * len_soi - slop.RasterCount
				for k in range(ntot): output['soil' + p + '_FL_out'].AddBand(gdal.GDT_Float32)
				output['soil' + p + '_FL_out'] = gdal.Translate(name + p + '-unc.tif', output['soil' + p + '_FL_out'], format = 'GTiff')
				os.remove('tmp5' + set_eros + set_depo + '.vrt')

			## Layer ordering in the output file will follow:
			## 1 = PFT 1 & Soil Layer 1; 2 = PFT 2 & Soil Layer 1; ... 260 = PFT 13 & Soil Layer 20
			t_j0 = np.arange(len_pft)
			k = np.arange(nc)
			for z in range(len_soi):
				for j in range(len_pft):
					idx_lyr = z * len_pft + j

					for i in range(1, nr + 1):
						m = mask.ReadAsArray(0, i-1, nc, 1).astype('float32')
						m[m != 1.] = np.nan

						export_acti = 0. * m
						export_slow = 0. * m
						export_pass = 0. * m
						ncell = (i-1)*nc + k
						idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + j*len_poo
							
						export_acti[0, :] = x_av[idx + 0]
						export_slow[0, :] = x_av[idx + 1]
						export_pass[0, :] = x_av[idx + 2]

						output['soila_FL_out'].GetRasterBand(idx_lyr + 1).WriteArray(export_acti * m, 0, i-1)
						output['soils_FL_out'].GetRasterBand(idx_lyr + 1).WriteArray(export_slow * m, 0, i-1)
						output['soilp_FL_out'].GetRasterBand(idx_lyr + 1).WriteArray(export_pass * m, 0, i-1)
			output = None

			## And we compress the output to avoid using excessive storage
			for p in soil_pools:
				os.system('gdal_translate -co "COMPRESS=LZW" ' + name + p + '-unc.tif ' + name + p + '.tif > /dev/null 2>&1')
				os.remove(name + p + '-unc.tif')

			# Then, we go to the next month
			x_cut = 1. * x[fi]
#			x_cut = spm(A_gpu, B_cut, x_cut, mdays[mo])
			x_cut = spm(A_cut, B_cut, x_cut, mdays[mo])
			x[fi] = 1. * x_cut

			if(mo == 11): x = ratio_calc(x, yr, yr + 1)

from pathos.multiprocessing import ProcessingPool
po = ProcessingPool(nodes = 5)

# Finally, we evaluate the function
v = 0
sq_fun(v)

