#!/usr/bin/env python

# Folders
data_dir = '/home/surface3/afendric/RESULTS/CE_DYNAM' 

# Loads packages
from osgeo import gdal
import math, os, glob, sys, numpy as np, scipy.sparse.linalg, scipy.optimize, datetime, pandas as pd, cPickle as pickle, backports.lzma

import warnings
warnings.filterwarnings('ignore')
#gdal.PushErrorHandler('CPLQuietErrorHandler')

# Defines the soil pools
soil_pools = ['a', 's', 'p'] # Active, Slow and Passive pools
len_poo = len(soil_pools)

# Defines the effect of CC to reduce soil erosion (CCf) and enrichment factor (EFf)
EFf = 1.00

# Defines if we will run with or without erosion/deposition
set_eros = sys.argv[1] # Erosion ('y') or no erosion ('n')
set_depo = sys.argv[2] # Deposition ('y') or no deposition ('n')

year_min = int(sys.argv[3])
year_max = year_min + 1
years = range(year_min, year_max)

mo = int(sys.argv[4])
months = range(mo, mo + 1)
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
grid = gdal.Open(data_dir + '/input/others/Grid_area.tif')
flac = gdal.Open(data_dir + '/input/others/Flow_accumulation.tif')
soc = gdal.Open(data_dir + '/input/others/Carbon_content.tif')
bulk = gdal.Open(data_dir + '/input/others/Bulk_density.tif')
dpth = gdal.Open(data_dir + '/input/others/Soil_depth.tif')
sgra = gdal.Open(data_dir + '/input/others/ORC_SG-ratio.tif')
#
# bass = gdal.Open('/home/users/afendric/CE_DYNAM/3sediment_calibration/input/basins-binary.tif')
#

def lzma_open(fname):
	out = pickle.load(backports.lzma.open(fname, 'rb'))
	return out

## Function to iterate over the area for equilibrium and transient calculation
def gen_AB(i, v, yr, mo):
	# Define the output lists that will contain all the information
	A_carbon_source = list()
	A_carbon_target = list()
	A_carbon_values = list()
	A_lateralIn_source = list()
	A_lateralIn_target = list()
	A_lateralIn_values = list()
	A_lateralOut_source = list()
	A_lateralOut_target = list()
	A_lateralOut_values = list()
	A_verticalIn_source = list()
	A_verticalIn_target = list()
	A_verticalIn_values = list()
	A_verticalOut_source = list()
	A_verticalOut_target = list()
	A_verticalOut_values = list()
	B_litter_target = list()
	B_litter_values = list()
	B_hs_target = list()
	B_hs_values = list()

	## In the emulator, we always calculate rate = flux[m + 1]/stock[m]
	yr_stock = yr
	mo_stock = mo

	if(mo == 11):
		yr_flux = yr + 1
		mo_flux = 0		
	else:
		yr_flux = yr
		mo_flux = mo + 1

	## For the years > 2018, we recycle the data of 2010-18
	df = yr_stock - yr_flux
	if yr_flux > 2018:
		yr_flux = 2010 + ((yr_flux - 2019) % 9)
		yr_stock = yr_flux + df

	# First, loads the equilibrium in hillslopes
	name = out_dir + out_dir_app + 'HS-' + str(yr) + '-%02d' % (mo + 1) + '-'
	eq_hs_a = gdal.Open(name + 'a.tif')
	eq_hs_s = gdal.Open(name + 's.tif')
	eq_hs_p = gdal.Open(name + 'p.tif')

	# Load some datasets
	pft = gdal.Open(data_dir + '/input/landcover/landcover_' + str(yr_flux) + '.tif')
	eros = gdal.Open(data_dir + '/output/Erosion/erosion_' + str(yr_flux) + '.tif')
	bulk = gdal.Open(data_dir + '/input/others/Bulk_density.tif')
	clay = gdal.Open(data_dir + '/input/ORCHIDEE/Clay_fraction.tif')
	grid = gdal.Open(data_dir + '/input/others/Grid_area.tif')
	flac = gdal.Open(data_dir + '/input/others/Flow_accumulation.tif')

	m = mask.ReadAsArray(0, i-1, nc, 1).astype('float32')
	m[m != 1.] = np.nan
#
#	mb = bass.ReadAsArray(0, i-1, nc, 1).astype('float16')
#	mb[mb != 1.] = np.nan
#

	## Reads the share of hillslopes, the grid area, and the PFT map
	t_g = grid.ReadAsArray(0, i-1, nc, 1).astype('float64')
	t_g[t_g == grid.GetRasterBand(1).GetNoDataValue()] = np.nan

	## Calculates the vertical discretization parameters
	t_dp = dpth.ReadAsArray(0, i-1, nc, 1).astype('float64') # Reads the soil depth in cm
	t_dp[t_dp == dpth.GetRasterBand(1).GetNoDataValue()] = np.nan
	t_dp = t_dp * 1./100 # Converts from cm to m

	dz_all = np.zeros((len_soi, nc))
	zmax = np.zeros((len_soi, nc))
	zmin = np.zeros((len_soi, nc))
	for k in range(nc):
		if np.isnan(t_dp[0, k]): continue
		dz_all[0, k] = 0.30 # First layer is assumed to have 30cm, so we have the correct erosion rate affecting it
		dz_all[1:, k] = np.repeat((t_dp[0, k] - 0.3) * 1./(len_soi - 1), (len_soi - 1))
		zmax[:, k] = dz_all[:, k].cumsum()
		zmin[1:, k] = zmax[0:(len_soi-1), k]

	## Calculates the respiration rate in each soil pool
	fx = sgra.ReadAsArray(0, i-1, nc, 1).astype('float64')
	fx[fx == sgra.GetRasterBand(1).GetNoDataValue()] = np.nan

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
			s[s < 1e-2] = 0.
			s[s == stock0_na] = np.nan
			
			t = t/s # This is actually t = (t * t_g * (vertical_discretization) * 1e-6)/(s * t_g * (vertical_discretization) * 1e-6)
			t = t * fx # We multiply by fx to get SoilGrid stocks
			t[np.isinf(t) | np.isneginf(t) | np.isnan(t)] = 0.

			kresp[var][j, 0, :] = t
		kresp[var][np.isnan(kresp[var]) | np.isinf(kresp[var]) | np.isneginf(kresp[var])] = 0.

	# If there is erosion, then we must calculate the turnover rates from the equilibrium erosion dataset
	kEE = np.zeros((len_pft, 1, nc)) ## The daily gross erosion from hillslopes
	kE = np.zeros((len_pft, 1, nc)) ## kEE times the transport factor
	t_kD = np.zeros((len_soi, len_pft, 1, nc)) ## The daily turnover rate
	t_kD_HS = np.zeros((len_soi, len_pft, 1, nc)) ## The analogous, for hillslopes

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

	t_lc = pft.ReadAsArray(0, i-1, nc, 1).astype('float64')
	t_lc[t_lc == pft.GetRasterBand(1).GetNoDataValue()] = np.nan
	t_lc = t_lc/100.
	if(set_eros == 'y' and set_depo == 'y'):
		## Calculates the gross and net erosion from hillslopes
		t_e = eros.ReadAsArray(0, i-1, nc, 1).astype('float64') # Reads the erosion rates in t/(ha.month)
		t_e[t_e < 0.] = 0.

		E_tot = np.zeros((len_pft, 1, t_g.shape[1]))
		for l in range(1, len_pft):
			if(l == 0): continue
			idxe = mo * len_pft + l
			E_tot[l] = 1. * t_e[idxe]
		E_sum = np.nansum(E_tot, axis = 0) # The total erosion, t/(ha.month)

		for l in range(1, len_pft):
			if(l == 0): continue
			kEE[l, 0, :] = EFf * E_tot[l] * (t_g * 1e-4)/mdays[mo] # Conversion to t/(day)
			kE[l, 0, :] = 1.0 * kEE[l, 0, :]
		kE[(np.isnan(kE)) | (np.isinf(kE)) | (np.isneginf(kE))] = 0.

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

		k_lit_soi['littera'][p] = (klitma_soila + klitsa_soila + klitmb_soila + klitsb_soila) * 0. # in g/(m2.day)
		k_lit_soi['litters'][p] = (klitsa_soils + klitsb_soils) * 0. # in g/(m2.day)

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
		s_a[s_a < 1e-2] = 0.
		s_a[s_a == stock0_a_na] = np.nan

		s_s = stock0_s[idx_stock, 0, :]
		s_s[s_s < 1e-2] = 0.
		s_s[s_s == stock0_s_na] = np.nan

		s_p = stock0_p[idx_stock, 0, :]
		s_p[s_p < 1e-2] = 0.
		s_p[s_p == stock0_p_na] = np.nan

		flux_a = fl_a[idx_flux, 0, :]
		flux_a[flux_a == fl_a_na] = np.nan

		flux_s = fl_s[idx_flux, 0, :]
		flux_s[flux_s == fl_s_na] = np.nan

		flux_p = fl_p[idx_flux, 0, :]
		flux_p[flux_p == fl_p_na] = np.nan

		ksoila_s_h[p] = FRAC_CARB_AS * flux_a/s_a ## This is actually ksoila_s_h = (FRAC_CARB_AS * flux * t_g * (vertical_discretization) * 1e-6)/(s * t_g * (vertical_discretization) * 1e-6)
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

	#### Now, the routing scheme. Here, we already load the up and down neighbors
	if i == 1:
		t_pf = pft.ReadAsArray(0, i-1, nc, 2)
		t_pf[t_pf == pft.GetRasterBand(1).GetNoDataValue()] = np.nan
		t_gr = grid.ReadAsArray(0, i-1, nc, 2)
		t_gr[t_gr == grid.GetRasterBand(1).GetNoDataValue()] = np.nan

		t_pf = np.concatenate((np.zeros((len_pft, 1, nc))*np.nan, t_pf), axis = 1)
		t_gr = np.r_[np.zeros((1, nc))*np.nan, t_gr]
	elif i == nr:
		t_pf = pft.ReadAsArray(0, i-2, nc, 2)
		t_pf[t_pf == pft.GetRasterBand(1).GetNoDataValue()] = np.nan
		t_gr = grid.ReadAsArray(0, i-2, nc, 2)
		t_gr[t_gr == grid.GetRasterBand(1).GetNoDataValue()] = np.nan

		t_pf = np.concatenate((t_pf, np.zeros((len_pft, 1, nc))*np.nan), axis = 1)
		t_gr = np.r_[t_gr, np.zeros((1, nc))*np.nan]
	else: 
		t_pf = pft.ReadAsArray(0, i-2, nc, 3)
		t_pf[t_pf == pft.GetRasterBand(1).GetNoDataValue()] = np.nan
		t_gr = grid.ReadAsArray(0, i-2, nc, 3)
		t_gr[t_gr == grid.GetRasterBand(1).GetNoDataValue()] = np.nan
	t_pf = t_pf/100
	t_pf[t_pf < 0.] = 0.
	t_gr[t_gr < 0] = 0.

	if i == 1:
		m_fl = mask.ReadAsArray(0, i-1, nc, 2).astype('float32')
		m_fl = np.r_[np.zeros((1, nc))*np.nan, m_fl]

		t_fl = flac.ReadAsArray(0, i-1, nc, 2).astype('float64')
		t_fl[t_fl == flac.GetRasterBand(1).GetNoDataValue()] = np.nan
		t_fl = np.r_[np.zeros((1, nc))*np.nan, t_fl]
	elif i == nr:
		m_fl = mask.ReadAsArray(0, i-2, nc, 2).astype('float32')
		m_fl = np.r_[m_fl, np.zeros((1, nc))*np.nan]

		t_fl = flac.ReadAsArray(0, i-2, nc, 2).astype('float64')
		t_fl[t_fl == flac.GetRasterBand(1).GetNoDataValue()] = np.nan
		t_fl = np.r_[t_fl, np.zeros((1, nc))*np.nan]
	else: 
		m_fl = mask.ReadAsArray(0, i-2, nc, 3).astype('float64')

		t_fl = flac.ReadAsArray(0, i-2, nc, 3).astype('float64')
		t_fl[t_fl == flac.GetRasterBand(1).GetNoDataValue()] = np.nan
	t_fl[(t_fl < 0.)] = np.nan
	t_fl = np.log(1. + t_fl)
	m_fl[m_fl != 1.] = np.nan
	DEM_in = 1./t_fl * m_fl

	# The sediment residence time, a function of the upstream area
	vmin = 1./(1. + np.exp(v[0]))
	vmax = 1./(1. + np.exp(v[1]))
	kOut = vmin + (vmax - vmin)/(1. + np.exp(v[2] + t_fl[1] * v[3]))**np.exp(v[4])
	kOut[(np.isnan(kOut)) | (np.isinf(kOut)) | (np.isneginf(kOut))] = 0.

	for z in soil_layers:
		for l in range(len_pft):
			for k in range(nc):
				t_kD[z, l, 0, k] = (1. - kOut[k]) # (1 - kOut[k]) is the share of sediments that do not go to the routing scheme
				t_kD_HS[z, l, 0, k] = kE[l, 0, k]/(b_al[z, 0, k] * (zmax[z, k] - zmin[z, k]) * t_g[0, k])
	t_kD[(np.isnan(t_kD)) | (np.isinf(t_kD)) | (np.isneginf(t_kD))] = 0.
	t_kD[t_kD > 1.] = 1.
	t_kD_HS[(np.isnan(t_kD_HS)) | (np.isinf(t_kD_HS)) | (np.isneginf(t_kD_HS))] = 0.
	t_kD_HS[t_kD_HS > 1.] = 1.

	## With the DEM, we calculate the denominator (W_den)
	W_den = np.zeros((1, nc))
	for k in range(nc):
		##### We have to go through each of the neighbors. Here we adopt the following convention:
		# 0 1 2
		# 7 C 3
		# 6 5 4
		# With C representing the central cell

		### We don't have to look all of them if we are on the borders
		if k == 0:
			index = np.array([1, 2, 3, 4, 5])
		elif k == nc-1:
			index = np.array([0, 1, 5, 6, 7])
		else:
			index = range(8)

		### Arrays index_x and index_y only tell the relative location of each neighbor
		### Array weight is the weight of Quinn (1991)
		index_x = np.array([-1, 0, 1, 1, 1, 0, -1, -1]) + k
		index_y = np.array([0, 0, 0, 1, 2, 2, 2, 1])
		weights = np.array([np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1])

		for nn in index:
			if DEM_in[index_y[nn], index_x[nn]] < DEM_in[1, k]:
				theta = (DEM_in[1, k] - DEM_in[index_y[nn], index_x[nn]])/weights[nn]
				W_den[0, k] = W_den[0, k] + theta

	## Now, we start to fill the A and B matrices
	t_j0 = np.arange(len_pft)
	t_j1 = np.arange(1, len_pft)
	kOut_adj = np.zeros((nc, len(t_j0)))

	val_eq_hs_a = np.zeros((len_soi, len_pft, 1, nc))
	val_eq_hs_s = np.zeros((len_soi, len_pft, 1, nc))
	val_eq_hs_p = np.zeros((len_soi, len_pft, 1, nc))
	k = np.arange(nc)

	va = eq_hs_a.ReadAsArray(0, i-1, nc, 1).astype('float64')
	va[np.isnan(va) | np.isinf(va) | np.isneginf(va)] = 0.
	vs = eq_hs_s.ReadAsArray(0, i-1, nc, 1).astype('float64')
	vs[np.isnan(vs) | np.isinf(vs) | np.isneginf(vs)] = 0.
	vp = eq_hs_p.ReadAsArray(0, i-1, nc, 1).astype('float64')
	vp[np.isnan(vp) | np.isinf(vp) | np.isneginf(vp)] = 0.
	for z in soil_layers:
		for j in range(len_pft):
			if j == 0: continue

			ncell = (i-1)*nc + k

			idx_lyr = z * len_pft + j
			val_eq_hs_a[z, j, 0, :] = va[idx_lyr, 0, :]
			val_eq_hs_s[z, j, 0, :] = vs[idx_lyr, 0, :]
			val_eq_hs_p[z, j, 0, :] = vp[idx_lyr, 0, :]

	for z in soil_layers:
		for k in range(nc):
			if np.isnan(m[0, k]): continue
#			if np.isnan(mb[0, k]): continue

			# We have to update index_x at every cell
			# And again, we don't have to look all neighbors if we are on the borders
			index_x = np.array([-1, 0, 1, 1, 1, 0, -1, -1]) + k
			index_y = np.array([0, 0, 0, 1, 2, 2, 2, 1])
			if k == 0:
				index = np.array([1, 2, 3, 4, 5])
			elif k == nc-1:
				index = np.array([0, 1, 5, 6, 7])
			else:
				index = range(8)

			# Calculates all necessary cell indexes
			ncell = (i-1)*nc + k
			idx_zm1 = ncell*len_soi*len_pft*len_poo + (z-1)*len_pft*len_poo + t_j0*len_poo #zm1 = [z-1]
			idx_zp1 = ncell*len_soi*len_pft*len_poo + (z+1)*len_pft*len_poo + t_j0*len_poo #zp1 = [z+1]

			# By definition, the matrix row corresponds to the target, and the column, to the source.
			# For example: A[idx + 0, idx + 1] corresponds to the flux to the active (+ 0) from the slow (+ 1) pool
			idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + t_j0*len_poo
			if(z == 0):
				idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + t_j1*len_poo
				if set_eros == 'y' and set_depo == 'y':
					for nn in index:
						if DEM_in[index_y[nn], index_x[nn]] < DEM_in[1, k]:
							theta = (DEM_in[1, k] - DEM_in[index_y[nn], index_x[nn]])/weights[nn]
							kIout = kOut[k] * theta/(W_den[0, k])
							if np.isnan(kIout) or np.isinf(kIout) or np.isneginf(kIout): kIout = 0.

							# Important reminder: the routing scheme sends the carbon from all existing PFTs except the first in one cell to all existing PFTs except the first of another
							## This is done by first weighting carbon by PFTs in source, and then redistributing also proportionally to the PFTs in the target

							# First, computes the loss from the current (source) cell - kOut_adj
							## Reminder: in the original code, this was a "+ kout" to A[idx+0, idx+0], A[idx+1, idx+1] etc.
							## ... but this would not conserve all the mass, since we re-weight kout here and exclude PFT 0.
							kIout_w = kIout * t_pf[t_j1, index_y[nn], index_x[nn]]/np.sum(t_pf[t_j1, index_y[nn], index_x[nn]])
							kIout_w[(np.isnan(kIout_w)) | (np.isinf(kIout_w)) | (np.isneginf(kIout_w))] = 0.
							n_nei = np.sum(t_pf[t_j1, index_y[nn], index_x[nn]] > 0)

							if n_nei == 0:
								kIout_w = 0. * kIout_w
							else:
								kIout_w = kIout_w * 1./n_nei
								# We divide by n_nei because each existing PFT in the source cell distributes ...
								# ... to all existing PFTs except bare soil of the target cell
							kOut_adj[k, 1:] += np.sum(kIout_w) * (t_pf[t_j1, 1, k] > 0) # This is the value that will be lost

							# Then, adds as a flow to the PFTs of the neighbor (target) cell
							ncell_n = (i - 2 + index_y[nn])*nc + index_x[nn]
							idx_p = np.repeat(idx, len(t_j1))
							t_n = np.tile(t_j1, len(t_j1))
							idx_n = ncell_n*len_soi*len_pft*len_poo + z*len_pft*len_poo + t_n*len_poo
							kIout_w_all = np.tile(kIout_w, len(t_j1)) * np.repeat((t_pf[t_j1, 1, k] > 0), len(t_j1))

							# Finally, we have to add the following ratio
							# That happens because calculations are done in a gC/(m2.PFT) scale
							# Therefore, if we don't rescale like this, then the sum of stock*pft for the losses and gains won't match
							ratio = np.repeat(t_pf[t_j1, 1, k], len(t_j1))/np.tile(t_pf[t_j1, index_y[nn], index_x[nn]], len(t_j1))
							ratio[(np.isnan(ratio)) | (np.isinf(ratio)) | (np.isneginf(ratio))] = 0.
							kIout_w_all = kIout_w_all * ratio

							## Active
#							A_lateralIn_target.append(idx_n + 0)
#							A_lateralIn_source.append(idx_p + 0)
#							A_lateralIn_values.append(-kIout_w_all)
							## Slow
#							A_lateralIn_target.append(idx_n + 1)
#							A_lateralIn_source.append(idx_p + 1)
#							A_lateralIn_values.append(-kIout_w_all)
							## Passive
#							A_lateralIn_target.append(idx_n + 2)
#							A_lateralIn_source.append(idx_p + 2)
#							A_lateralIn_values.append(-kIout_w_all)

				# Additional fluxes
				idx = ncell*len_soi*len_pft*len_poo + z*len_pft*len_poo + t_j0*len_poo

				## Active
				### Lateral movement to the routing scheme
#				A_lateralOut_target.append(idx + 0)
#				A_lateralOut_source.append(idx + 0)
#				A_lateralOut_values.append(kOut_adj[k])

				### Carbon dynamics
#				A_carbon_target.append(idx + 0)
#				A_carbon_source.append(idx + 0)
#				A_carbon_values.append(kresp['active'][t_j0, 0, k] + ksoila_s_h[t_j0, 0, k] + ksoila_p_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 0)
#				A_carbon_source.append(idx + 1)
#				A_carbon_values.append(-ksoils_a_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 0)
#				A_carbon_source.append(idx + 2)
#				A_carbon_values.append(-ksoilp_a_h[t_j0, 0, k])

				### The deposition movement from [z-1] to [z]
#				A_verticalOut_target.append(idx + 0)
#				A_verticalOut_source.append(idx + 0)
#				A_verticalOut_values.append(t_kD[z, t_j0, 0, k])

				### The movement from [z+1] to [z]
#				A_verticalIn_target.append(idx + 0)
#				A_verticalIn_source.append(idx_zp1 + 0)
#				A_verticalIn_values.append(-kOut_adj[k])

				## Slow
				### Lateral movement to the routing scheme
#				A_lateralOut_target.append(idx + 1)
#				A_lateralOut_source.append(idx + 1)
#				A_lateralOut_values.append(kOut_adj[k])

				### Carbon dynamics
#				A_carbon_target.append(idx + 1)
#				A_carbon_source.append(idx + 0)
#				A_carbon_values.append(-ksoila_s_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 1)
#				A_carbon_source.append(idx + 1)
#				A_carbon_values.append(kresp['slow'][t_j0, 0, k] + ksoils_p_h[t_j0, 0, k] + ksoils_a_h[t_j0, 0, k])

				### The deposition movement from [z-1] to [z]
#				A_verticalOut_target.append(idx + 1)
#				A_verticalOut_source.append(idx + 1)
#				A_verticalOut_values.append(t_kD[z, t_j0, 0, k])

				### The movement from [z+1] to [z]
#				A_verticalIn_target.append(idx + 1)
#				A_verticalIn_source.append(idx_zp1 + 1)
#				A_verticalIn_values.append(-kOut_adj[k])


				## Passive
				### Lateral movement to the routing scheme
#				A_lateralOut_target.append(idx + 2)
#				A_lateralOut_source.append(idx + 2)
#				A_lateralOut_values.append(kOut_adj[k])

				### Carbon dynamics
#				A_carbon_target.append(idx + 2)
#				A_carbon_source.append(idx + 0)
#				A_carbon_values.append(-ksoila_p_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 2)
#				A_carbon_source.append(idx + 1)
#				A_carbon_values.append(-ksoils_p_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 2)
#				A_carbon_source.append(idx + 2)
#				A_carbon_values.append(kresp['passive'][t_j0, 0, k] + ksoilp_a_h[t_j0, 0, k])

				### The deposition movement from [z-1] to [z]
#				A_verticalOut_target.append(idx + 2)
#				A_verticalOut_source.append(idx + 2)
#				A_verticalOut_values.append(t_kD[z, t_j0, 0, k])

				### The movement from [z+1] to [z]
#				A_verticalIn_target.append(idx + 2)
#				A_verticalIn_source.append(idx_zp1 + 2)
#				A_verticalIn_values.append(-kOut_adj[k])

				# The inputs
				## Active
				B_litter_target.append(idx + 0)
				B_litter_values.append(k_lit_soi['littera_vd'][z, t_j0, 0, k])
				B_hs_target.append(idx + 0)
				B_hs_values.append(t_kD_HS[z, t_j0, 0, k] * val_eq_hs_a[z, t_j0, 0, k])

				## Slow
				B_litter_target.append(idx + 1)
				B_litter_values.append(k_lit_soi['litters_vd'][z, t_j0, 0, k])
				B_hs_target.append(idx + 1)
				B_hs_values.append(t_kD_HS[z, t_j0, 0, k] * val_eq_hs_s[z, t_j0, 0, k])

				## Passive
				B_hs_target.append(idx + 2)
				B_hs_values.append(t_kD_HS[z, t_j0, 0, k] * val_eq_hs_p[z, t_j0, 0, k])

			elif(z == (len_soi - 1)):
				# The fluxes
				## Active
				#### The movement from [z+1] to [z]
#				A_verticalOut_target.append(idx + 0)
#				A_verticalOut_source.append(idx + 0)
#				A_verticalOut_values.append(kOut_adj[k])

				### The deposition movement from [z-1] to [z]
#				A_verticalIn_target.append(idx + 0)
#				A_verticalIn_source.append(idx_zm1 + 0)
#				A_verticalIn_values.append(-t_kD[z-1, t_j0, 0, k])

				### Carbon
#				A_carbon_target.append(idx + 0)
#				A_carbon_source.append(idx + 0)
#				A_carbon_values.append(kresp['active'][t_j0, 0, k] + ksoila_s_h[t_j0, 0, k] + ksoila_p_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 0)
#				A_carbon_source.append(idx + 1)
#				A_carbon_values.append(-ksoils_a_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 0)
#				A_carbon_source.append(idx + 2)
#				A_carbon_values.append(-ksoilp_a_h[t_j0, 0, k])


				## Slow
				#### The movement from [z+1] to [z]
#				A_verticalOut_target.append(idx + 1)
#				A_verticalOut_source.append(idx + 1)
#				A_verticalOut_values.append(kOut_adj[k])

				### The deposition movement from [z-1] to [z]
#				A_verticalIn_target.append(idx + 1)
#				A_verticalIn_source.append(idx_zm1 + 1)
#				A_verticalIn_values.append(-t_kD[z-1, t_j0, 0, k])

				### Carbon
#				A_carbon_target.append(idx + 1)
#				A_carbon_source.append(idx + 0)
#				A_carbon_values.append(-ksoila_s_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 1)
#				A_carbon_source.append(idx + 1)
#				A_carbon_values.append(kresp['slow'][t_j0, 0, k] + ksoils_p_h[t_j0, 0, k] + ksoils_a_h[t_j0, 0, k])


				## Passive
				#### The movement from [z+1] to [z]
#				A_verticalOut_target.append(idx + 2)
#				A_verticalOut_source.append(idx + 2)
#				A_verticalOut_values.append(kOut_adj[k])

				### The deposition movement from [z-1] to [z]
#				A_verticalIn_target.append(idx + 2)
#				A_verticalIn_source.append(idx_zm1 + 2)
#				A_verticalIn_values.append(-t_kD[z-1, t_j0, 0, k])

				### Carbon
#				A_carbon_target.append(idx + 2)
#				A_carbon_source.append(idx + 0)
#				A_carbon_values.append(-ksoila_p_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 2)
#				A_carbon_source.append(idx + 1)
#				A_carbon_values.append(-ksoils_p_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 2)
#				A_carbon_source.append(idx + 2)
#				A_carbon_values.append(kresp['passive'][t_j0, 0, k] + ksoilp_a_h[t_j0, 0, k])

				# The inputs
				## Active
				B_litter_target.append(idx + 0)
				B_litter_values.append(k_lit_soi['littera_vd'][z, t_j0, 0, k])

				## Slow
				B_litter_target.append(idx + 1)
				B_litter_values.append(k_lit_soi['litters_vd'][z, t_j0, 0, k])
			else:
				## Active
				### The vertical movements
#				A_verticalOut_target.append(idx + 0)
#				A_verticalOut_source.append(idx + 0)
#				A_verticalOut_values.append(kOut_adj[k] + t_kD[z, t_j0, 0, k])

				### The deposition movement from [z-1] to [z]
#				A_verticalIn_target.append(idx + 0)
#				A_verticalIn_source.append(idx_zm1 + 0)
#				A_verticalIn_values.append(-t_kD[z-1, t_j0, 0, k])

				### The movement from [z+1] to [z]
#				A_verticalIn_target.append(idx + 0)
#				A_verticalIn_source.append(idx_zp1 + 0)
#				A_verticalIn_values.append(-kOut_adj[k])

				### Carbon
#				A_carbon_target.append(idx + 0)
#				A_carbon_source.append(idx + 0)
#				A_carbon_values.append(kresp['active'][t_j0, 0, k] + ksoila_s_h[t_j0, 0, k] + ksoila_p_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 0)
#				A_carbon_source.append(idx + 1)
#				A_carbon_values.append(-ksoils_a_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 0)
#				A_carbon_source.append(idx + 2)
#				A_carbon_values.append(-ksoilp_a_h[t_j0, 0, k])

				## Slow
				### The vertical movements
#				A_verticalOut_target.append(idx + 1)
#				A_verticalOut_source.append(idx + 1)
#				A_verticalOut_values.append(kOut_adj[k] + t_kD[z, t_j0, 0, k])

				### The deposition movement from [z-1] to [z]
#				A_verticalIn_target.append(idx + 1)
#				A_verticalIn_source.append(idx_zm1 + 1)
#				A_verticalIn_values.append(-t_kD[z-1, t_j0, 0, k])

				### The movement from [z+1] to [z]
#				A_verticalIn_target.append(idx + 1)
#				A_verticalIn_source.append(idx_zp1 + 1)
#				A_verticalIn_values.append(-kOut_adj[k])

				### Carbon
#				A_carbon_target.append(idx + 1)
#				A_carbon_source.append(idx + 0)
#				A_carbon_values.append(-ksoila_s_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 1)
#				A_carbon_source.append(idx + 1)
#				A_carbon_values.append(kresp['slow'][t_j0, 0, k] + ksoils_p_h[t_j0, 0, k] + ksoils_a_h[t_j0, 0, k])

				## Passive
				#### The vertical movements
#				A_verticalOut_target.append(idx + 2)
#				A_verticalOut_source.append(idx + 2)
#				A_verticalOut_values.append(kOut_adj[k] + t_kD[z, t_j0, 0, k])

				### The deposition movement from [z-1] to [z]
#				A_verticalIn_target.append(idx + 2)
#				A_verticalIn_source.append(idx_zm1 + 2)
#				A_verticalIn_values.append(-t_kD[z-1, t_j0, 0, k])

				### The movement from [z+1] to [z]
#				A_verticalIn_target.append(idx + 2)
#				A_verticalIn_source.append(idx_zp1 + 2)
#				A_verticalIn_values.append(-kOut_adj[k])

				### Carbon
#				A_carbon_target.append(idx + 2)
#				A_carbon_source.append(idx + 0)
#				A_carbon_values.append(-ksoila_p_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 2)
#				A_carbon_source.append(idx + 1)
#				A_carbon_values.append(-ksoils_p_h[t_j0, 0, k])

#				A_carbon_target.append(idx + 2)
#				A_carbon_source.append(idx + 2)
#				A_carbon_values.append(kresp['passive'][t_j0, 0, k] + ksoilp_a_h[t_j0, 0, k])


				# The inputs
				## Active
				B_litter_target.append(idx + 0)
				B_litter_values.append(k_lit_soi['littera_vd'][z, t_j0, 0, k])

				## Slow
				B_litter_target.append(idx + 1)
				B_litter_values.append(k_lit_soi['litters_vd'][z, t_j0, 0, k])

	if len(A_carbon_target) > 0:
		A_carbon_out = pd.DataFrame({'target': np.concatenate(A_carbon_target), 'source': np.concatenate(A_carbon_source), 'values': np.concatenate(A_carbon_values)})
		A_carbon_out = A_carbon_out.groupby(['target', 'source']).agg({'values': 'sum'})
		A_carbon_out = A_carbon_out.reset_index()
	else:
		A_carbon_out = pd.DataFrame({'target': 1, 'source': 1, 'values': 0}, index = [0])

	if len(A_lateralIn_target) > 0:
		A_lateralIn_out = pd.DataFrame({'target': np.concatenate(A_lateralIn_target), 'source': np.concatenate(A_lateralIn_source), 'values': np.concatenate(A_lateralIn_values)})
		A_lateralIn_out = A_lateralIn_out.groupby(['target', 'source']).agg({'values': 'sum'})
		A_lateralIn_out = A_lateralIn_out.reset_index()
	else:
		A_lateralIn_out = pd.DataFrame({'target': 1, 'source': 1, 'values': 0}, index = [0])

	if len(A_lateralOut_target) > 0:
		A_lateralOut_out = pd.DataFrame({'target': np.concatenate(A_lateralOut_target), 'source': np.concatenate(A_lateralOut_source), 'values': np.concatenate(A_lateralOut_values)})
		A_lateralOut_out = A_lateralOut_out.groupby(['target', 'source']).agg({'values': 'sum'})
		A_lateralOut_out = A_lateralOut_out.reset_index()
	else:
		A_lateralOut_out = pd.DataFrame({'target': 1, 'source': 1, 'values': 0}, index = [0])

	if len(A_verticalIn_target) > 0:
		A_verticalIn_out = pd.DataFrame({'target': np.concatenate(A_verticalIn_target), 'source': np.concatenate(A_verticalIn_source), 'values': np.concatenate(A_verticalIn_values)})
		A_verticalIn_out = A_verticalIn_out.groupby(['target', 'source']).agg({'values': 'sum'})
		A_verticalIn_out = A_verticalIn_out.reset_index()
	else:
		A_verticalIn_out = pd.DataFrame({'target': 1, 'source': 1, 'values': 0}, index = [0])

	if len(A_verticalOut_target) > 0:
		A_verticalOut_out = pd.DataFrame({'target': np.concatenate(A_verticalOut_target), 'source': np.concatenate(A_verticalOut_source), 'values': np.concatenate(A_verticalOut_values)})
		A_verticalOut_out = A_verticalOut_out.groupby(['target', 'source']).agg({'values': 'sum'})
		A_verticalOut_out = A_verticalOut_out.reset_index()
	else:
		A_verticalOut_out = pd.DataFrame({'target': 1, 'source': 1, 'values': 0}, index = [0])

	if len(B_litter_target) > 0:
		B_litter_out = pd.DataFrame({'target': np.concatenate(B_litter_target), 'values': np.concatenate(B_litter_values)})
		B_litter_out = B_litter_out.groupby(['target']).agg({'values': 'sum'})
		B_litter_out = B_litter_out.reset_index()
	else:
		B_litter_out = pd.DataFrame({'target': 1, 'values': 0}, index = [0])

	if len(B_hs_target) > 0:
		B_hs_out = pd.DataFrame({'target': np.concatenate(B_hs_target), 'values': np.concatenate(B_hs_values)})
		B_hs_out = B_hs_out.groupby(['target']).agg({'values': 'sum'})
		B_hs_out = B_hs_out.reset_index()
	else:
		B_hs_out = pd.DataFrame({'target': 1, 'values': 0}, index = [0])

	return i, A_carbon_out, A_lateralIn_out, A_lateralOut_out, A_verticalIn_out, A_verticalOut_out, B_litter_out, B_hs_out


def gen_mats(v, yr, mo):
	print 'Year: ' + str(yr) + ', Month: ' + str(mo) + ', Time: ' + str(datetime.datetime.now())
	sys.stdout.flush()

	# Builds the A and B matrices
	A_carbon = scipy.sparse.dok_matrix((nc * nr * len_pft * len_soi * len_poo, nc * nr * len_pft * len_soi * len_poo))
	A_lateralIn = scipy.sparse.dok_matrix((nc * nr * len_pft * len_soi * len_poo, nc * nr * len_pft * len_soi * len_poo))
	A_lateralOut = scipy.sparse.dok_matrix((nc * nr * len_pft * len_soi * len_poo, nc * nr * len_pft * len_soi * len_poo))
	A_verticalIn = scipy.sparse.dok_matrix((nc * nr * len_pft * len_soi * len_poo, nc * nr * len_pft * len_soi * len_poo))
	A_verticalOut = scipy.sparse.dok_matrix((nc * nr * len_pft * len_soi * len_poo, nc * nr * len_pft * len_soi * len_poo))
	B_litter = np.zeros((nc * nr * len_pft * len_soi * len_poo), dtype = 'float32')
	B_hs = np.zeros((nc * nr * len_pft * len_soi * len_poo), dtype = 'float32')

	for i in range(1, nr+1):
		i, A_carbon_out, A_lateralIn_out, A_lateralOut_out, A_verticalIn_out, A_verticalOut_out, B_litter_out, B_hs_out = gen_AB(i, v, yr, mo)

		A_carbon[A_carbon_out['target'], A_carbon_out['source']] += np.array(A_carbon_out['values'], dtype = np.float32)
		A_lateralIn[A_lateralIn_out['target'], A_lateralIn_out['source']] += np.array(A_lateralIn_out['values'], dtype = np.float32)
		A_lateralOut[A_lateralOut_out['target'], A_lateralOut_out['source']] += np.array(A_lateralOut_out['values'], dtype = np.float32)
		A_verticalIn[A_verticalIn_out['target'], A_verticalIn_out['source']] += np.array(A_verticalIn_out['values'], dtype = np.float32)
		A_verticalOut[A_verticalOut_out['target'], A_verticalOut_out['source']] += np.array(A_verticalOut_out['values'], dtype = np.float32)
		B_litter[B_litter_out['target']] += B_litter_out['values']
		B_hs[B_hs_out['target']] += B_hs_out['values']

	A_carbon = A_carbon.tocsr()
	A_lateralIn = A_lateralIn.tocsr()
	A_lateralOut = A_lateralOut.tocsr()
	p = scipy.sparse.eye(A_lateralIn.shape[0])
	A_lateral = p - A_lateralIn - A_lateralOut
	A_lateral = p - A_lateral**v[5]

	A_verticalIn = A_verticalIn.tocsr()
	A_verticalOut = A_verticalOut.tocsr()
	A_vertical = p - A_verticalIn - A_verticalOut
	A_vertical = p - A_vertical**v[6]

	A_verticalIn = A_verticalIn.tocsr()
	A_verticalOut = A_verticalOut.tocsr()
	A = A_carbon + A_lateral + A_vertical
	B = B_litter + B_hs

	# We first find the filled rows, then continue
	# fi = np.unique(A.nonzero()[0])
	name = out_dir + out_dir_app + 'FL-' + str(yr) + '-%02d' % (mo + 1) + '-' # I add +1 to mo just to avoid confusions with the file name
	fi = lzma_open(name + 'fi.npz')

	A_carbon = A_carbon[fi,:][:,fi]
	A_lateral = A_lateral[fi,:][:,fi]
	A_vertical = A_vertical[fi,:][:,fi]
	A_cut = A[fi,:][:,fi]
	B_litter = B_litter[fi]
	B_hs = B_hs[fi]
	B_cut = B[fi]

	name = out_dir + out_dir_app + 'FL-' + str(yr) + '-%02d' % (mo + 1) + '-' # I add +1 to mo just to avoid confusions with the file name

#	A_exp = list()
#	A_exp.append(A_carbon)
#	A_exp.append(A_lateral)
#	A_exp.append(A_vertical)
#	pickle.dump(A_exp, backports.lzma.open(name + 'A_cut.npz', 'wb'))
#	pickle.dump(fi, backports.lzma.open(name + 'fi.npz', 'wb'))
	B_exp = list()
	B_exp.append(B_litter)
	B_exp.append(B_hs)
	pickle.dump(B_exp, backports.lzma.open(name + 'B_cut.npz', 'wb'))

	return 1

## Now, a function to evaluate the model for the whole period
def sq_fun(v):
	for yr in years:
		for mo in months:
			ok =  gen_mats(v, yr, mo)

# Finally, we evaluate the function
v = [4.557645357127455, -2.512184070618636, -7.099665937772622, 5.005065316309821, -0.45921697318257326, 3, 2]
sq_fun(v)

