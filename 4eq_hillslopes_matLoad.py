#!/usr/bin/env python

# Folders
data_dir = '/home/surface3/afendric/RESULTS/CE_DYNAM'

# Loads packages
from osgeo import gdal
import math, os, glob, sys, numpy as np, scipy.sparse.linalg, scipy.optimize, datetime, pandas as pd, patsy, cPickle as pickle, subprocess

import warnings
warnings.filterwarnings('ignore')
#gdal.PushErrorHandler('CPLQuietErrorHandler')

# Defines the soil pools
soil_pools = ['a', 's', 'p'] # Active, Slow and Passive pools
len_poo = len(soil_pools)

# Defines if we will run with or without erosion/deposition
set_eros = sys.argv[1] # Erosion ('y') or no erosion ('n')
set_depo = sys.argv[2] # Deposition ('y') or no deposition ('n')

years = range(1860, 2019)
if(len(sys.argv) == 4):
	years = range(int(sys.argv[3]), 2019)
months = range(12)
mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

len_soi = 3 # The number of soil layers
soil_layers = range(len_soi)

# Then, changes the output path according to the decision
out_dir = data_dir + '/output/Simulations/'
if(set_eros == 'y' and set_depo == 'y'):
	out_dir_app = 'Yeros-Ydepo/'
elif(set_eros == 'n' and set_depo == 'n'):
	out_dir_app = 'Neros-Ndepo/'
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
hsfr = gdal.Open(data_dir + '/input/others/Hillslope_fraction.tif')
grid = gdal.Open(data_dir + '/input/others/Grid_area.tif')
flac = gdal.Open(data_dir + '/input/others/Flow_accumulation.tif')
bulk = gdal.Open(data_dir + '/input/others/Bulk_density.tif')
soc = gdal.Open(data_dir + '/input/others/Carbon_content.tif')
dpth = gdal.Open(data_dir + '/input/others/Soil_depth.tif')

def gzip_open(fname):
	# This is just a faster version of:
	# import pickle, gzip; a = pickle.load(gzip.open(fname, 'rb'))

	lnk = subprocess.Popen(["zcat", fname], stdout = subprocess.PIPE)
	stdo = lnk.communicate()
	out = pickle.loads(stdo[0])
	return out

def ratio_calc(yr_old, yr_new):
	# There is a small error on the emulator. It does not scale stocks on 01/Jan as ORCHIDEE does.
	# Since this is the only error, we can solve it by simply deriving such ratio from the carbon stocks, as below.
	# We also load the land cover because this calculation is only triggered when land cover changes.
	ratio = 1. + np.zeros((nc * nr * len_pft * len_soi * len_poo), dtype = 'float32')

	mo_new = 0
	va_new = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_active_' + str(yr_new) + '.tif')
	vs_new = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_slow_' + str(yr_new) + '.tif')
	vp_new = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_passive_' + str(yr_new) + '.tif')
	lc_new = gdal.Open(data_dir + '/input/landcover/landcover_' + str(yr_new) + '.tif')

	mo_old = 11
	va_old = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_active_' + str(yr_old) + '.tif')
	vs_old = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_slow_' + str(yr_old) + '.tif')
	vp_old = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_passive_' + str(yr_old) + '.tif')
	lc_old = gdal.Open(data_dir + '/input/landcover/landcover_' + str(yr_old) + '.tif')

	k = np.arange(nc)
	for j in range(len_pft):
		idx_lyr_new = mo_new * len_pft + j
		idx_lyr_old = mo_old * len_pft + j

		for i in range(1, nr + 1):
			ncell = (i-1)*nc + k

			dlc_new = lc_new.ReadAsArray(0, i-1, nc, 1).astype('float64')
			dlc_old = lc_old.ReadAsArray(0, i-1, nc, 1).astype('float64')
			delta_veg = dlc_new[j] - dlc_old[j]
			mds = np.where(delta_veg > 0)[1] # The change is triggered only when the share increases

			dlc_new[dlc_new == lc_new.GetRasterBand(1).GetNoDataValue()] = np.nan
			dva_new = va_new.ReadAsArray(0, i-1, nc, 1).astype('float64')
			dva_new[dva_new == va_new.GetRasterBand(1).GetNoDataValue()] = np.nan
			dvs_new = vs_new.ReadAsArray(0, i-1, nc, 1).astype('float64')
			dvs_new[dvs_new == vs_new.GetRasterBand(1).GetNoDataValue()] = np.nan
			dvp_new = vp_new.ReadAsArray(0, i-1, nc, 1).astype('float64')
			dvp_new[dvp_new == vp_new.GetRasterBand(1).GetNoDataValue()] = np.nan
			dva_new = dva_new[idx_lyr_new, 0, :]
			dvs_new = dvs_new[idx_lyr_new, 0, :]
			dvp_new = dvp_new[idx_lyr_new, 0, :]

			dlc_old[dlc_old == lc_old.GetRasterBand(1).GetNoDataValue()] = np.nan
			dva_old = va_old.ReadAsArray(0, i-1, nc, 1).astype('float64')
			dva_old[dva_old == va_old.GetRasterBand(1).GetNoDataValue()] = np.nan
			dvs_old = vs_old.ReadAsArray(0, i-1, nc, 1).astype('float64')
			dvs_old[dvs_old == vs_old.GetRasterBand(1).GetNoDataValue()] = np.nan
			dvp_old = vp_old.ReadAsArray(0, i-1, nc, 1).astype('float64')
			dvp_old[dvp_old == vp_old.GetRasterBand(1).GetNoDataValue()] = np.nan

			dva_old = dva_old[idx_lyr_old, 0, :]
			dvs_old = dvs_old[idx_lyr_old, 0, :]
			dvp_old = dvp_old[idx_lyr_old, 0, :]

			# Trick: I had to do this to avoid division by very small numbers
			dva_new[dva_new < 1e-3] = 0.
			dvs_new[dvs_new < 1e-3] = 0.
			dvp_new[dvp_new < 1e-3] = 0.
			dva_old[dva_old < 1e-3] = 0.
			dvs_old[dvs_old < 1e-3] = 0.
			dvp_old[dvp_old < 1e-3] = 0.

			t_hs = hsfr.ReadAsArray(0, i-1, nc, 1).astype('float64')
			t_hs[t_hs == hsfr.GetRasterBand(1).GetNoDataValue()] = np.nan

			for z in range(len_soi):
				idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + j*len_poo

				v = dva_new[mds]/dva_old[mds]
				v[np.isinf(v) | np.isneginf(v) | np.isnan(v)] = 1.
				ratio[idx[mds] + 0] = v

				v = dvs_new[mds]/dvs_old[mds]
				v[np.isinf(v) | np.isneginf(v) | np.isnan(v)] = 1.
				ratio[idx[mds] + 1] = v

				v = dvp_new[mds]/dvp_old[mds]
				v[np.isinf(v) | np.isneginf(v) | np.isnan(v)] = 1.
				ratio[idx[mds] + 2] = v
	ratio[np.isnan(ratio)] = 1.
	return ratio

def gen_mats(v, yr, mo):
	print 'Year: ' + str(yr) + ', Month: ' + str(mo) + ', Time: ' + str(datetime.datetime.now())
	sys.stdout.flush()

	if(yr == 1860 and mo == 0):
#		# We have to load the data for the whole year

###
		x = np.zeros((nc * nr * len_pft * len_soi * len_poo), dtype = 'float32')
		va = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_active_1860.tif')
		vs = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_slow_1860.tif')
		vp = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_passive_1860.tif')

		for i in range(1, nr + 1):
			# First, calculates the vertical discretization of SOC
			t_dp = dpth.ReadAsArray(0, i-1, nc, 1).astype('float64') # Reads the soil depth in cm
			t_dp[t_dp == dpth.GetRasterBand(1).GetNoDataValue()] = np.nan
			t_dp = t_dp * 1./100 # Converts from cm to m
			dz_all = np.zeros((len_soi, nc))
			zmax = np.zeros((len_soi, nc))
			zmin = np.zeros((len_soi, nc))
			p_a = np.exp(v[20])
			p_b = np.real(- np.exp(p_a) - 1. * scipy.special.lambertw(- np.exp(p_a - np.exp(p_a)))) # We calculate for d = 1, only to get the proportions in %
			for k in range(nc):
				if np.isnan(t_dp[0, k]): continue
				for z in soil_layers:
					dz_all[z, k] = t_dp[0, k] * 1/p_b * (np.exp(p_a + p_b * (z+1)/len_soi) - np.exp(p_a + p_b * z/len_soi))
				dz_all[:, k] = np.flip(dz_all[:, k]) # We flip because we calculate an exponential increase above, but we want an exponential decrease
				zmax[:, k] = dz_all[:, k].cumsum()
				zmin[1:, k] = zmax[0:(len_soi-1), k]

			dep_max = np.array([0.05, 0.15, 0.30, 0.60, 1., 2.]) # in meters, taken from SoilGrids (the source of bulk density)
			dep_min = np.r_[0, dep_max[0:-1]]
			t_b = bulk.ReadAsArray(0, i-1, nc, 1).astype('float64') * 0.01 # Multiplies by 0.01 because the data is in 100*g/cm3 originally
			t_b[t_b == bulk.GetRasterBand(1).GetNoDataValue()] = np.nan
			t_b[t_b < 0.] = 0.
			t_c = soc.ReadAsArray(0, i-1, nc, 1).astype('float64') # Units are dg/kg, so we should multiply it by (t_b * grid_area * height).
			# ... However, since grid_area is constant and we only need percentages, I multiply by t_b here and height after. The unit of the final quantity is a ratio MASS/AREA.
			t_c[t_c == soc.GetRasterBand(1).GetNoDataValue()] = np.nan
			t_c[t_b < 0.] = 0.
			t_c = t_c * t_b
			b_al = np.zeros((len_soi, t_b.shape[1], t_b.shape[2]))
			c_al = np.zeros((len_soi, t_b.shape[1], t_b.shape[2]))
			for k in range(nc):
				for z in soil_layers:
					lmax = np.min(np.where(dep_max >= round(zmax[z, k], 2))) # Finds the corresponding layers, the rounding is just to solve numerical issues
					lmin = np.max(np.where(dep_min <= round(zmin[z, k], 2)))

					# Creates a vector with min and max depths to interpolate the dataset
					d_min = dep_min[range(lmin, lmax + 1)]
					d_min[0] = zmin[z, k]
					d_max = dep_max[range(lmin, lmax + 1)]
					d_max[-1] = zmax[z, k]
					for j in range(lmax - lmin + 1):
						b_al[z, 0, k] = b_al[z, 0, k] + (d_max[j] - d_min[j]) * t_b[lmin + j, 0, k]
						c_al[z, 0, k] = c_al[z, 0, k] + (d_max[j] - d_min[j]) * t_c[lmin + j, 0, k]
					b_al[z, 0, k] = b_al[z, 0, k]/(zmax[z, k] - zmin[z, k]) # For b_al we need the absolute values...
				c_al[:, 0, k] = c_al[:, 0, k]/np.sum(c_al[:, 0, k]) # ... but for c_al we need the percentages
			t_b = None
			t_c = None

			# Then, values at equilibrium
			for j in range(len_pft):
				idx_lyr = mo * len_pft + j

				k = np.arange(nc)
				ncell = (i-1)*nc + k
				dva = va.ReadAsArray(0, i-1, nc, 1)
				dva[dva == va.GetRasterBand(1).GetNoDataValue()] = np.nan
				dvs = vs.ReadAsArray(0, i-1, nc, 1)
				dvs[dvs == vs.GetRasterBand(1).GetNoDataValue()] = np.nan
				dvp = vp.ReadAsArray(0, i-1, nc, 1)
				dvp[dvp == vp.GetRasterBand(1).GetNoDataValue()] = np.nan
				t_hs = hsfr.ReadAsArray(0, i-1, nc, 1).astype('float64')
				t_hs[t_hs == hsfr.GetRasterBand(1).GetNoDataValue()] = np.nan

				dva = dva[idx_lyr, 0, :]
				dvs = dvs[idx_lyr, 0, :]
				dvp = dvp[idx_lyr, 0, :]

				for z in range(len_soi):
					idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + j*len_poo

					x[idx + 0] = (dva * t_hs) * c_al[z, 0, :]
					x[idx + 1] = (dvs * t_hs) * c_al[z, 0, :]
					x[idx + 2] = (dvp * t_hs) * c_al[z, 0, :]
		x[np.isnan(x)] = 0.
###
		if set_eros == 'y' and set_depo == 'y':
			print 'Starting the iterative procedure: - Time: ' + str(datetime.datetime.now())
			it = 1
			while it <= 6: # Iteration criteria
				it += 1
				for yk in range(1860, 1870):
					x_old = 1. * x
					for mk in months:
						name = out_dir + out_dir_app + 'HS-' + str(yk) +'-%02d' % (mk + 1) + '-'
						A_exp = gzip_open(name + 'A_cut.npz')
						A_cut = A_exp[0].tocsr() + A_exp[1].tocsr() + A_exp[2].tocsr()

						B_cut = gzip_open(name + 'B_cut.npz')
						fi = gzip_open(name + 'fi.npz')

						x_cut = 1. * x[fi]
						dif = mdays[mk]
						x_cut = x_cut + dif * (B_cut - A_cut.dot(x_cut))
						x_cut[x_cut < 0] = 0.
						x[fi] = 1. * x_cut

						if mk == 11:
							yr_old = yk

							if(yk < 1869): yr_new = yk + 1
							else: yr_new = 1860

							ratio = ratio_calc(yr_old, yr_new)
							x = x * ratio

					p = np.nansum(abs(x - x_old)/abs(x_old) < 1e-2)/float(np.sum(x > 0)) # Convergence criteria
					print 'p = ' + str(round(p, 4))
					print 'max(x) = ' + str(np.nanmax(x))
					# print 'which.max(x) = ' + str(np.where(x == np.nanmax(x))[0])
					print 'problem point = ' + str(x[12626177])
					sys.stdout.flush()
			x[np.isnan(x)] = 0.
			print 'Ending the iterative procedure: - Time: ' + str(datetime.datetime.now())
###
		# Then, we load for mo = 0 only and return
		name = out_dir + out_dir_app + 'HS-' + str(yr) + '-%02d' % (mo + 1) + '-'
		A_exp = gzip_open(name + 'A_cut.npz')
		A_cut = A_exp[0].tocsr() + A_exp[1].tocsr() + A_exp[2].tocsr()

		B_cut = gzip_open(name + 'B_cut.npz')
		fi = gzip_open(name + 'fi.npz')

		return fi, A_cut, B_cut, x

	name = out_dir + out_dir_app + 'HS-' + str(yr) + '-%02d' % (mo + 1) + '-'
	A_exp = gzip_open(name + 'A_cut.npz')
	A_cut = A_exp[0].tocsr() + A_exp[1].tocsr() + A_exp[2].tocsr()
	B_cut = gzip_open(name + 'B_cut.npz')
	fi = gzip_open(name + 'fi.npz')
	if(mo < 11):
		return fi, A_cut, B_cut
	elif(mo == 11):
		ratio = ratio_calc(yr, yr+1)
		return fi, A_cut, B_cut, ratio


## Now, a function to evaluate the model for the whole period
def sq_fun(v):
	if(years[0] > 1860):
		x = np.zeros((nc * nr * len_pft * len_soi * len_poo), dtype = 'float32')

		name = out_dir + out_dir_app + 'HS-' + str(years[0] - 1) + '-12-'
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

		fi, A_cut, B_cut, ratio = gen_mats(v, years[0] - 1, 11)
		x_cut = 1. * x[fi]
		dif = mdays[11]
		x_cut = x_cut + dif * (B_cut - A_cut.dot(x_cut))
		x_cut[x_cut < 0] = 0.
		x[fi] = 1. * x_cut
		x = x * ratio

	# Now, updates the model for the transient period and calculates the cost function
	for yr in years:
		for mo in months:
			if (yr == 2018 and mo == 11): continue

			if(yr == 1860 and mo == 0):
				fi, A_cut, B_cut, x = gen_mats(v, yr, mo)
			else:
				if(mo < 11): fi, A_cut, B_cut = gen_mats(v, yr, mo)
				elif(mo == 11): fi, A_cut, B_cut, ratio = gen_mats(v, yr, mo)

			x_av = 1. * x

#			pi = 142
#			pk = 271
#			pcell = (pi - 1) * nc + pk
#			ppp = pcell*len_soi*len_pft*len_poo + 0*len_pft*len_poo + 5*len_poo
#			print x[ppp + 1]

			# For each month, we export the calculations to rasters
			output = {}
			name = out_dir + out_dir_app + 'HS-' + str(yr) + '-%02d' % (mo + 1) + '-' # I add +1 to mo just to avoid confusions with the file name
			for p in soil_pools:
				output['soil' + p + '_HS_out'] = gdal.GetDriverByName('VRT').CreateCopy('tmp4' + set_eros + set_depo + '.vrt', slop)
				ntot = len_pft * len_soi - slop.RasterCount
				for k in range(ntot): output['soil' + p + '_HS_out'].AddBand(gdal.GDT_Float32)
				output['soil' + p + '_HS_out'] = gdal.Translate(name + p + '-unc.tif', output['soil' + p + '_HS_out'], format = 'GTiff')
				os.remove('tmp4' + set_eros + set_depo + '.vrt')

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

						output['soila_HS_out'].GetRasterBand(idx_lyr + 1).WriteArray(export_acti * m, 0, i-1)
						output['soils_HS_out'].GetRasterBand(idx_lyr + 1).WriteArray(export_slow * m, 0, i-1)
						output['soilp_HS_out'].GetRasterBand(idx_lyr + 1).WriteArray(export_pass * m, 0, i-1)
			output = None

			## And we compress the output to avoid using excessive storage
			for p in soil_pools:
				os.system('gdal_translate -co "COMPRESS=LZW" ' + name + p + '-unc.tif ' + name + p + '.tif > /dev/null 2>&1')
				os.remove(name + p + '-unc.tif')

			# Then, we go to the next month
			x_cut = 1. * x[fi]
			dif = mdays[mo]
			x_cut = x_cut + dif * (B_cut - A_cut.dot(x_cut))
			x_cut[x_cut < 0] = 0.
			x[fi] = 1. * x_cut

			if(mo == 11): x = x * ratio

		## IMPORTANT: commented in case floodplain simulation is not supposed to be automatic
		os.system('cd /home/users/afendric/CE_DYNAM/matSave/; ./submit-auto_' + set_eros + set_depo + '.sh ' + str(yr) + '; cd -')

# Finally, we evaluate the function
v = [2.97491262, 2.55649123, 2.52046258, 2.86138299, 1.86101897, 3.22409303, 4.26845489, 1.25774512, 7.36785917, 4.43445958, 4.3076853, 6.94343175, 1.3667537, 1.99010179, 2.22882103, 7.653916, 5.12329096, 4.16539457, 8.70465835, 9.09948618, -2.52965945] # Taken from output_2_1st_order
sq_fun(v)

