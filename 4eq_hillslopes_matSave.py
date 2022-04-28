#!/usr/bin/env python

# Folders
data_dir = '/home/surface3/afendric/RESULTS/CE_DYNAM' 

# Loads packages
from osgeo import gdal
import math, os, glob, sys, numpy as np, scipy.sparse.linalg, scipy.optimize, datetime, pandas as pd, patsy, cPickle as pickle, gzip

import warnings
warnings.filterwarnings('ignore')
#gdal.PushErrorHandler('CPLQuietErrorHandler')

# Defines the soil pools
soil_pools = ['a', 's', 'p'] # Active, Slow and Passive pools
len_poo = len(soil_pools)

# Defines if we will run with or without erosion/deposition
set_eros = sys.argv[1] # Erosion ('y') or no erosion ('n')
set_depo = sys.argv[2] # Deposition ('y') or no deposition ('n')

year_min = int(sys.argv[3])
year_max = year_min + 1

years = range(year_min, year_max)
months = range(12)
mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

len_soi = 3 # The number of soil layersz
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
#
# bass = gdal.Open('/home/users/afendric/CE_DYNAM/3sediment_calibration/input/basins-binary.tif')
#

# We have to generate the basis matrix for slope and flow accumulation
vals = slop.ReadAsArray().astype('float64')
vals[vals == slop.GetRasterBand(1).GetNoDataValue()] = np.nan
vals = np.log(np.concatenate(vals) + 0.1) # This log-scale is to deal with outliers
vals = vals[~np.isnan(vals)]
dsm_sl = patsy.dmatrix("bs(x, df=5, degree=3, include_intercept=True) - 1", {"x": vals})

vals = flac.ReadAsArray().astype('float64')
vals[vals == flac.GetRasterBand(1).GetNoDataValue()] = np.nan
vals[(vals < 0.)] = np.nan
vals = np.log(np.concatenate(vals)) # This log-scale is to deal with outliers
vals = vals[~np.isnan(vals)]
dsm_fl = patsy.dmatrix("bs(x, df=5, degree=3, include_intercept=True) - 1", {"x": vals})

vals = None

## Function to iterate over the area for equilibrium and transient calculation
def gen_AB(i, v, yr, mo):
	# Define the output lists that will contain all the information
	A_carbon_source = list()
	A_carbon_target = list()
	A_carbon_values = list()
	A_etdIn_source = list()
	A_etdIn_target = list()
	A_etdIn_values = list()
	A_etdOut_source = list()
	A_etdOut_target = list()
	A_etdOut_values = list()
	B_target = list()
	B_values = list()

	## In the emulator, we always calculate rate = flux[m + 1]/stock[m]
	yr_stock = yr
	mo_stock = mo

	if(mo == 11):
		yr_flux = yr + 1
		mo_flux = 0		
	else:
		yr_flux = yr
		mo_flux = mo + 1


	# First, load some datasets
	pft = gdal.Open(data_dir + '/input/landcover/landcover_' + str(yr_flux) + '.tif')
	eros = gdal.Open(data_dir + '/output/Erosion/erosion_' + str(yr_flux) + '.tif')
	bulk = gdal.Open(data_dir + '/input/others/Bulk_density.tif')
	clay = gdal.Open(data_dir + '/input/ORCHIDEE/Clay_fraction.tif')
	grid = gdal.Open(data_dir + '/input/others/Grid_area.tif')

	## Reads the share of hillslopes and the grid area
	t_hs = hsfr.ReadAsArray(0, i-1, nc, 1).astype('float64')
	t_hs[t_hs == hsfr.GetRasterBand(1).GetNoDataValue()] = np.nan
	t_g = grid.ReadAsArray(0, i-1, nc, 1).astype('float64')
	t_g[t_g == grid.GetRasterBand(1).GetNoDataValue()] = np.nan

	## Calculates the vertical discretization parameters
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

	## Calculates the respiration rate in each soil pool
	kresp = {}
	for p in soil_pools:
		if p == 'a':
			var = 'active'
		elif p == 's':
			var = 'slow'
		elif p == 'p':
			var = 'passive'

		kresp[var] = np.zeros((len_pft, 1, nc))

		f_resp0 = gdal.Open(data_dir + '/input/ORCHIDEE/respiration/resp_hetero_soil_' + p + '_' + str(yr_flux) + '.tif') # data is in gC/m^2/day, according to documentation
		f_resp0_na = f_resp0.GetRasterBand(1).GetNoDataValue()
		f_resp0 = f_resp0.ReadAsArray(0, i-1, nc, 1).astype('float64')

		stock0 = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_' + var + '_' + str(yr_stock) + '.tif') # data is in gC/m^2/pft, according to documentation
		stock0_na = stock0.GetRasterBand(1).GetNoDataValue()
		stock0 = stock0.ReadAsArray(0, i-1, nc, 1).astype('float64')

		for j in range(len_pft):
			if j == 0: continue

			idx_flux = mo_flux * len_pft + j
			idx_stock = mo_stock * len_pft + j

			t = f_resp0[idx_flux, 0, :]
			t[t == f_resp0_na] = np.nan
			
			s = stock0[idx_stock, 0, :]
			s[s == stock0_na] = np.nan
			
			t = t/s # This is actually t = (t * t_g * t_hs * (vertical_discretization) * 1e-6)/(s * t_g * t_hs * (vertical_discretization) * 1e-6)
			t[np.isinf(t) | np.isneginf(t) | np.isnan(t)] = 0.

			kresp[var][j, 0, :] = t
		kresp[var][np.isnan(kresp[var]) | np.isinf(kresp[var]) | np.isneginf(kresp[var])] = 0.

	# If there is erosion, then we must calculate the turnover rates from the equilibrium erosion dataset
	kEE = np.zeros((len_pft, 1, nc)) # The daily gross erosion from hillslopes
	kE = np.zeros((len_pft, 1, nc)) # The daily net erosion from hillslopes
	t_kD = np.zeros((len_soi, len_pft, 1, nc)) ## The daily turnover rate

	# Calculates the vertical discretization of bulk density and SOC, which will be useful later
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

	if(set_eros == 'y' and set_depo == 'y'):
		## Calculates the gross and net erosion from hillslopes
		t_e = eros.ReadAsArray(0, i-1, nc, 1).astype('float64') # Reads the erosion rates in t/(ha.month)
		t_e[t_e < 0.] = 0.

		E_tot = np.zeros((len_pft, 1, t_g.shape[1]))
		for l in range(1, len_pft):
			if(l == 0): continue
			idxe = mo * len_pft + l
			E_tot[l] = t_e[idxe] * t_g * 1e-4 # Conversion from m2 to ha, E_tot is in t/year

		t_s = slop.ReadAsArray(0, i-1, nc, 1).astype('float64')
		t_pf = pft.ReadAsArray(0, i-1, nc, 1).astype('float64')
		t_pf[t_pf == pft.GetRasterBand(1).GetNoDataValue()] = np.nan
		t_pf = t_pf/100.

		t_s[t_s < 0] = np.nan
		t_s = np.log(0.1 + t_s) # This is just to improve scale
		na = np.where(~np.isnan(t_s[0]))[0]
		nd = {"x": t_s[~np.isnan(t_s)]}
		f = np.zeros((len_pft, 1, nc))
		f = f * t_s * 0.
		if(len(nd['x']) > 0):
			new_dsm_sl = patsy.build_design_matrices([dsm_sl.design_info], nd)[0]

			# We assume four different curves for f: crop, grass, bare = 0, and forest
			for l in range(1, len_pft):
				if l == 0: continue # Bare
				elif l in [1, 2, 3, 4, 5, 6, 7, 8]: # Forest
					b = np.array([v[0], v[1], v[2], v[3], v[4]])
				elif l in [9, 10, 13, 14]: # Grass
					b = np.array([v[5], v[6], v[7], v[8], v[9]])
				elif l in [11, 12]: # Croplands
					b = np.array([v[10], v[11], v[12], v[13], v[14]])
				f[l, 0, na] = np.dot(new_dsm_sl, b) * t_pf[l, 0, na]
		f = np.sum(f, axis = 0)
		f = 1./(1. + np.exp(f))

		for l in range(1, len_pft):
			if(l == 0): continue

			if(yr == 1860 and mo == 0):
				kEE[l, 0, :] = E_tot[l]/np.sum(mdays)
			else:
				kEE[l, 0, :] = E_tot[l]/mdays[mo]
			kE[l, 0, :] = f * kEE[l, 0, :]

		for z in soil_layers:
			for l in range(len_pft):
				for k in range(nc):
					t_kD[z, l, 0, k] = kE[l, 0, k]/(b_al[z, 0, k] * (zmax[z, k] - zmin[z, k]) * t_hs[0, k] * t_pf[l, 0, k] * t_g[0, k])
		t_kD[(np.isnan(t_kD)) | (np.isinf(t_kD)) | (np.isneginf(t_kD))] = 0.

	# Now, calculates the inputs from litter (reminder: unlike the others, it is an absolute value [t] instead of a rate [1/day])
	km1 = gdal.Open(data_dir + '/input/ORCHIDEE/soilcarbon/soilcarb_input_ma_a_' + str(yr_flux) + '.tif') # data is in gC/m^2/day, according to documentation
	km1_na = km1.GetRasterBand(1).GetNoDataValue()
	km1 = km1.ReadAsArray(0, i-1, nc, 1).astype('float64')

	km2 = gdal.Open(data_dir + '/input/ORCHIDEE/soilcarbon/soilcarb_input_sa_a_' + str(yr_flux) + '.tif')
	km2_na = km2.GetRasterBand(1).GetNoDataValue()
	km2 = km2.ReadAsArray(0, i-1, nc, 1).astype('float64')

	km3 = gdal.Open(data_dir + '/input/ORCHIDEE/soilcarbon/soilcarb_input_mb_a_' + str(yr_flux) + '.tif')
	km3_na = km3.GetRasterBand(1).GetNoDataValue()
	km3 = km3.ReadAsArray(0, i-1, nc, 1).astype('float64')

	km4 = gdal.Open(data_dir + '/input/ORCHIDEE/soilcarbon/soilcarb_input_sb_a_' + str(yr_flux) + '.tif')
	km4_na = km4.GetRasterBand(1).GetNoDataValue()
	km4 = km4.ReadAsArray(0, i-1, nc, 1).astype('float64')

	km5 = gdal.Open(data_dir + '/input/ORCHIDEE/soilcarbon/soilcarb_input_sa_s_' + str(yr_flux) + '.tif')
	km5_na = km5.GetRasterBand(1).GetNoDataValue()
	km5 = km5.ReadAsArray(0, i-1, nc, 1).astype('float64')

	km6 = gdal.Open(data_dir + '/input/ORCHIDEE/soilcarbon/soilcarb_input_sb_s_' + str(yr_flux) + '.tif')
	km6_na = km6.GetRasterBand(1).GetNoDataValue()
	km6 = km6.ReadAsArray(0, i-1, nc, 1).astype('float64')

	k_lit_soi = {} # This will store the fluxes between litter and soil pools
	for k in ['littera', 'litters']:
		k_lit_soi[k] = np.zeros((len_pft, 1, nc))
		k_lit_soi[k + '_vd'] = np.zeros((len_soi, len_pft, 1, nc))
	for p in range(len_pft):
		if p == 0: continue

		idx_flux = mo_flux * len_pft + p
		idx_stock = mo_stock * len_pft + p

		klitma_soila = km1[idx_flux, 0, :]
		klitma_soila[klitma_soila == km1_na] = np.nan

		klitsa_soila = km2[idx_flux, 0, :]
		klitsa_soila[klitsa_soila == km2_na] = np.nan

		klitmb_soila = km3[idx_flux, 0, :]
		klitmb_soila[klitmb_soila == km3_na] = np.nan

		klitsb_soila = km4[idx_flux, 0, :]
		klitsb_soila[klitsb_soila == km4_na] = np.nan

		klitsa_soils = km5[idx_flux, 0, :]
		klitsa_soils[klitsa_soils == km5_na] = np.nan

		klitsb_soils = km6[idx_flux, 0, :]
		klitsb_soils[klitsb_soils == km6_na] = np.nan

		k_lit_soi['littera'][p] = (klitma_soila + klitsa_soila + klitmb_soila + klitsb_soila) * t_hs # in g/(m2.day)
		k_lit_soi['litters'][p] = (klitsa_soils + klitsb_soils) * t_hs # in g/(m2.day)

#		### ... and vertically discretizes them proportionally to the observed SOC
		for k in ['littera', 'litters']:
			for z in soil_layers:
				k_lit_soi[k + '_vd'][z, p, 0, :] = k_lit_soi[k][p] * c_al[z, 0, :]
	k_lit_soi[k + '_vd'][np.isnan(k_lit_soi[k + '_vd']) | np.isinf(k_lit_soi[k + '_vd']) | np.isneginf(k_lit_soi[k + '_vd'])] = 0.

	# Then, calculates the fluxes between pools (a -> s, s -> p etc.). These are assumed identical for all soil layers
	## Begins with the coefficients defined on ORCHIDEE
	METABOLIC_REF_FRAC = 0.85
	FRAC_CARB_AP = 0.004
	FRAC_CARB_SA = 0.420
	FRAC_CARB_SP = 0.030
	FRAC_CARB_PA = 0.450
	## FRAC_CARB_PS = 0.000
	ACTIVE_TO_PASS_CLAY_FRAC = 0.680

	cl = clay.GetRasterBand(1).ReadAsArray(0, i-1, nc, 1).astype('float64')
	cl[cl == clay.GetRasterBand(1).GetNoDataValue()] = np.nan
	FRAC_CARB_AS = 1. - METABOLIC_REF_FRAC + ACTIVE_TO_PASS_CLAY_FRAC * cl - FRAC_CARB_AP

	## Defines the outputs
	ksoila_s_h = np.zeros((len_pft, 1, nc))
	ksoila_p_h = np.zeros((len_pft, 1, nc))
	ksoils_a_h = np.zeros((len_pft, 1, nc))
	ksoils_p_h = np.zeros((len_pft, 1, nc))
	ksoilp_a_h = np.zeros((len_pft, 1, nc))

	## Manually iterates over the soil pools for the fluxes
	stock0_a = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_active_' + str(yr_stock) + '.tif') # data is in gC/m^2/pft, according to documentation
	stock0_a_na = stock0_a.GetRasterBand(1).GetNoDataValue()
	stock0_a = stock0_a.ReadAsArray(0, i-1, nc, 1).astype('float64')

	stock0_s = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_slow_' + str(yr_stock) + '.tif')
	stock0_s_na = stock0_s.GetRasterBand(1).GetNoDataValue()
	stock0_s = stock0_s.ReadAsArray(0, i-1, nc, 1).astype('float64')

	stock0_p = gdal.Open(data_dir + '/input/ORCHIDEE/carbon/carbon_passive_' + str(yr_stock) + '.tif')
	stock0_p_na = stock0_p.GetRasterBand(1).GetNoDataValue()
	stock0_p = stock0_p.ReadAsArray(0, i-1, nc, 1).astype('float64')

	fl_a = gdal.Open(data_dir + '/input/ORCHIDEE/fluxtot/fluxtot_a_' + str(yr_flux) + '.tif') # data is in gC/m^2/day, according to documentation
	fl_a_na = fl_a.GetRasterBand(1).GetNoDataValue()
	fl_a = fl_a.ReadAsArray(0, i-1, nc, 1).astype('float64')

	fl_s = gdal.Open(data_dir + '/input/ORCHIDEE/fluxtot/fluxtot_s_' + str(yr_flux) + '.tif') # data is in gC/m^2/day, according to documentation
	fl_s_na = fl_s.GetRasterBand(1).GetNoDataValue()
	fl_s = fl_s.ReadAsArray(0, i-1, nc, 1).astype('float64')

	fl_p = gdal.Open(data_dir + '/input/ORCHIDEE/fluxtot/fluxtot_p_' + str(yr_flux) + '.tif') # data is in gC/m^2/day, according to documentation
	fl_p_na = fl_p.GetRasterBand(1).GetNoDataValue()
	fl_p = fl_p.ReadAsArray(0, i-1, nc, 1).astype('float64')

	for p in range(len_pft):
		if p == 0: continue

		idx_flux = mo_flux * len_pft + p
		idx_stock = mo_stock * len_pft + p
 
		s_a = stock0_a[idx_stock, 0, :]
		s_a[s_a == stock0_a_na] = np.nan

		s_s = stock0_s[idx_stock, 0, :]
		s_s[s_s == stock0_s_na] = np.nan

		s_p = stock0_p[idx_stock, 0, :]
		s_p[s_p == stock0_p_na] = np.nan

		flux_a = fl_a[idx_flux, 0, :]
		flux_a[flux_a == fl_a_na] = np.nan

		flux_s = fl_s[idx_flux, 0, :]
		flux_s[flux_s == fl_s_na] = np.nan

		flux_p = fl_p[idx_flux, 0, :]
		flux_p[flux_p == fl_p_na] = np.nan

		ksoila_s_h[p] = FRAC_CARB_AS * flux_a/s_a ## This is actually ksoila_s_h = (FRAC_CARB_AS * flux * t_g * t_hs * (vertical_discretization) * 1e-6)/(s * t_g * t_hs * (vertical_discretization) * 1e-6)
		ksoila_s_h[p][np.isnan(ksoila_s_h[p]) | np.isinf(ksoila_s_h[p]) | np.isneginf(ksoila_s_h[p])] = 0.
		ksoila_s_h[p][ksoila_s_h[p] > 1.] = 1.

		ksoila_p_h[p] = FRAC_CARB_AP * flux_a/s_a ## ... analogous comment
		ksoila_p_h[p][np.isnan(ksoila_p_h[p]) | np.isinf(ksoila_p_h[p]) | np.isneginf(ksoila_p_h[p])] = 0.
		ksoila_p_h[p][ksoila_p_h[p] > 1.] = 1.

		ksoilp_a_h[p] = FRAC_CARB_PA * flux_p/s_p # ... analogous comment
		ksoilp_a_h[p][np.isnan(ksoilp_a_h[p]) | np.isinf(ksoilp_a_h[p]) | np.isneginf(ksoilp_a_h[p])] = 0.
		ksoilp_a_h[p][ksoilp_a_h[p] > 1.] = 1.

		ksoils_a_h[p] = FRAC_CARB_SA * flux_s/s_s # ... analogous comment
		ksoils_a_h[p][np.isnan(ksoils_a_h[p]) | np.isinf(ksoils_a_h[p]) | np.isneginf(ksoils_a_h[p])] = 0.
		ksoils_a_h[p][ksoils_a_h[p] > 1.] = 1.

		ksoils_p_h[p] = FRAC_CARB_SP * flux_s/s_s # ... analogous comment
		ksoils_p_h[p][np.isnan(ksoils_p_h[p]) | np.isinf(ksoils_p_h[p]) | np.isneginf(ksoils_p_h[p])] = 0.
		ksoils_p_h[p][ksoils_p_h[p] > 1.] = 1.

	# Finally, builds the matrices of the linear system
	m = mask.ReadAsArray(0, i-1, nc, 1).astype('float32')
	m[m != 1.] = np.nan
#
#	mb = bass.ReadAsArray(0, i-1, nc, 1).astype('float16')
#	mb[mb != 1.] = np.nan
#

	t_j1 = np.arange(1, len_pft) ## The bare soil PFT is not included

	for z in soil_layers:
		for k in range(nc):
			if np.isnan(m[0, k]): continue
#			if np.isnan(mb[0, k]): continue

			## Calculates all necessary cell indexes
			ncell = (i-1)*nc + k
			idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + t_j1*len_poo
			idx_zp1 = ncell*len_soi*len_pft*len_poo + (z+1)*len_pft*len_poo + t_j1*len_poo
			
			## Active pool (+0)
			A_carbon_target.append(idx + 0)
			A_carbon_source.append(idx + 0)
			A_carbon_values.append(kresp['active'][t_j1, 0, k] + ksoila_s_h[t_j1, 0, k] + ksoila_p_h[t_j1, 0, k])

			A_etdOut_target.append(idx + 0)
			A_etdOut_source.append(idx + 0)
			A_etdOut_values.append(t_kD[z, t_j1, 0, k])

			A_carbon_target.append(idx + 0)
			A_carbon_source.append(idx + 1)
			A_carbon_values.append(-ksoils_a_h[t_j1, 0, k])

			A_carbon_target.append(idx + 0)
			A_carbon_source.append(idx + 2)
			A_carbon_values.append(-ksoilp_a_h[t_j1, 0, k])

			if z < (len_soi - 1): ## For all layers except the last, we have vertical flow from [z+1] to [z]
				A_etdIn_target.append(idx + 0)
				A_etdIn_source.append(idx_zp1 + 0)
				A_etdIn_values.append(-t_kD[z+1, t_j1, 0, k])

			## Slow pool (+1)
			A_carbon_target.append(idx + 1)
			A_carbon_source.append(idx + 0)
			A_carbon_values.append(-ksoila_s_h[t_j1, 0, k])

			A_carbon_target.append(idx + 1)
			A_carbon_source.append(idx + 1)
			A_carbon_values.append(kresp['slow'][t_j1, 0, k] + ksoils_p_h[t_j1, 0, k] + ksoils_a_h[t_j1, 0, k])

			A_etdOut_target.append(idx + 1)
			A_etdOut_source.append(idx + 1)
			A_etdOut_values.append(t_kD[z, t_j1, 0, k])

			if z < (len_soi - 1):
				A_etdIn_target.append(idx + 1)
				A_etdIn_source.append(idx_zp1 + 1)
				A_etdIn_values.append(- t_kD[z+1, t_j1, 0, k])

			## Passive pool (+2)
			A_carbon_target.append(idx + 2)
			A_carbon_source.append(idx + 0)
			A_carbon_values.append(-ksoila_p_h[t_j1, 0, k])

			A_carbon_target.append(idx + 2)
			A_carbon_source.append(idx + 1)
			A_carbon_values.append(-ksoils_p_h[t_j1, 0, k])

			A_carbon_target.append(idx + 2)
			A_carbon_source.append(idx + 2)
			A_carbon_values.append(kresp['passive'][t_j1, 0, k] + ksoilp_a_h[t_j1, 0, k])

			A_etdOut_target.append(idx + 2)
			A_etdOut_source.append(idx + 2)
			A_etdOut_values.append(t_kD[z, t_j1, 0, k])

			if z < (len_soi - 1):
				A_etdIn_target.append(idx + 2)
				A_etdIn_source.append(idx_zp1 + 2)
				A_etdIn_values.append(- t_kD[z+1, t_j1, 0, k])

			## For the B matrix
			B_target.append(idx + 0)
			B_values.append(k_lit_soi['littera_vd'][z, t_j1, 0, k])

			B_target.append(idx + 1)
			B_values.append(k_lit_soi['litters_vd'][z, t_j1, 0, k])

	if len(A_carbon_target) > 0:
		A_carbon_out = pd.DataFrame({'target': np.concatenate(A_carbon_target), 'source': np.concatenate(A_carbon_source), 'values': np.concatenate(A_carbon_values)})
		A_carbon_out = A_carbon_out.groupby(['target', 'source']).agg({'values': 'sum'})
		A_carbon_out = A_carbon_out.reset_index()
	else:
		A_carbon_out = pd.DataFrame({'target': 1, 'source': 1, 'values': 0}, index = [0])

	if len(A_etdOut_target) > 0:
		A_etdOut_out = pd.DataFrame({'target': np.concatenate(A_etdOut_target), 'source': np.concatenate(A_etdOut_source), 'values': np.concatenate(A_etdOut_values)})
		A_etdOut_out = A_etdOut_out.groupby(['target', 'source']).agg({'values': 'sum'})
		A_etdOut_out = A_etdOut_out.reset_index()
	else:
		A_etdOut_out = pd.DataFrame({'target': 1, 'source': 1, 'values': 0}, index = [0])

	if len(A_etdIn_target) > 0:
		A_etdIn_out = pd.DataFrame({'target': np.concatenate(A_etdIn_target), 'source': np.concatenate(A_etdIn_source), 'values': np.concatenate(A_etdIn_values)})
		A_etdIn_out = A_etdIn_out.groupby(['target', 'source']).agg({'values': 'sum'})
		A_etdIn_out = A_etdIn_out.reset_index()
	else:
		A_etdIn_out = pd.DataFrame({'target': 1, 'source': 1, 'values': 0}, index = [0])

	if len(B_target) > 0:
		B_out = pd.DataFrame({'target': np.concatenate(B_target), 'values': np.concatenate(B_values)})
		B_out = B_out.groupby(['target']).agg({'values': 'sum'})
		B_out = B_out.reset_index()
	else:
		B_out = pd.DataFrame({'target': 1, 'values': 0}, index = [0])


	return i, A_carbon_out, A_etdOut_out, A_etdIn_out, B_out

def gen_mats(v, yr, mo):
	print 'Year: ' + str(yr) + ', Month: ' + str(mo) + ', Time: ' + str(datetime.datetime.now())
	sys.stdout.flush()

	# Builds the A and B matrices
	A_carbon = scipy.sparse.dok_matrix((nc * nr * len_pft * len_soi * len_poo, nc * nr * len_pft * len_soi * len_poo))
	A_etdOut = scipy.sparse.dok_matrix((nc * nr * len_pft * len_soi * len_poo, nc * nr * len_pft * len_soi * len_poo))
	A_etdIn = scipy.sparse.dok_matrix((nc * nr * len_pft * len_soi * len_poo, nc * nr * len_pft * len_soi * len_poo))
	B = np.zeros((nc * nr * len_pft * len_soi * len_poo), dtype = 'float32')

	for i in range(1, nr+1):
		i, A_carbon_out, A_etdOut_out, A_etdIn_out, B_out = gen_AB(i, v, yr, mo)

		A_carbon[A_carbon_out['target'], A_carbon_out['source']] += np.array(A_carbon_out['values'], dtype = np.float32)
		A_etdOut[A_etdOut_out['target'], A_etdOut_out['source']] += np.array(A_etdOut_out['values'], dtype = np.float32)
		A_etdIn[A_etdIn_out['target'], A_etdIn_out['source']] += np.array(A_etdIn_out['values'], dtype = np.float32)
		B[B_out['target']] += B_out['values']
	A_carbon = A_carbon.tocsr()
	A_etdOut = A_etdOut.tocsr()
	A_etdIn = A_etdIn.tocsr()
	A = A_carbon + A_etdOut + A_etdIn

	# We first find the filled rows, then continue
	fi = np.unique(A.nonzero()[0])
	A_carbon = A_carbon[fi,:][:,fi]
	A_etdOut = A_etdOut[fi,:][:,fi]
	A_etdIn = A_etdIn[fi,:][:,fi]
	B_cut = B[fi]

	name = out_dir + out_dir_app + 'HS-' + str(yr) + '-%02d' % (mo + 1) + '-' # I add +1 to mo just to avoid confusions with the file name
	A_exp = list()
	A_exp.append(A_carbon)
	A_exp.append(A_etdOut)
	A_exp.append(A_etdIn)
	pickle.dump(A_exp, gzip.open(name + 'A_cut.npz', 'wb', compresslevel=6))
	pickle.dump(B_cut, gzip.open(name + 'B_cut.npz', 'wb', compresslevel=6))
	pickle.dump(fi, gzip.open(name + 'fi.npz', 'wb', compresslevel=6))

	return 1

## Now, a function to evaluate the model for the whole period
def sq_fun(v):
	for yr in years:
		for mo in months:
			if (yr != 2018 or mo != 11): ok =  gen_mats(v, yr, mo)

# Finally, we evaluate the function
v = [2.97491262, 2.55649123, 2.52046258, 2.86138299, 1.86101897, 3.22409303, 4.26845489, 1.25774512, 7.36785917, 4.43445958, 4.3076853, 6.94343175, 1.3667537, 1.99010179, 2.22882103, 7.653916, 5.12329096, 4.16539457, 8.70465835, 9.09948618, -2.52965945] # Taken from output_2_1st_order
sq_fun(v)

