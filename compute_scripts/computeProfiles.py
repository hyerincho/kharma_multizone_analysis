import pyharm
import pickle
import numpy as np
import os
import glob

def shellAverage(dump, quantity, imin=0, mass_weight=True):
	"""
	Starting with a pyharm dump object, average some quantity with respect to phi and theta.
	Includes some smart parsing of quantities that are not keys in the dump object.
	"""

	#Many quantities we want are already keys, but some are not.  For those, we must come up with our own formula.
	if quantity == 'Mdot':
		#This is the weirdest one.  We won't average ourselves, and we'll use a built-in pyharm function.
		return pyharm.shell_sum(dump, 'FM')
	if quantity == 'T':
		to_average = dump['u'] / dump['rho'] * (dump['gam']-1)
	else:
		to_average = dump[quantity]

	#Weighting for the average.
	volumetric_weight = dump['gdet']
	if mass_weight:
		density = dump['rho']

	if dump['n3'] > 1: #3d
		return np.sum(to_average[imin:,:,:] * volumetric_weight * density, axis=(1,2)) / np.sum(volumetric_weight * density, axis=(1,2))
	else:
		return np.sum(to_average[imin:,:] * volumetric_weight * density, axis=1) / np.sum(volumetric_weight * density, axis=1)

def computeProfileSet(dump, quantities=['Mdot', 'rho', 'u', 'T', 'u^r', 'u^phi'], imin=0, mass_weight=True):
	"""
	Open one dump, then compute various profiles from it.  Return a list of profiles.
	"""

	output = []
	for quantity in quantities:
		print(f"   {quantity}")
		output.append(shellAverage(dump, quantity, imin=imin, mass_weight=mass_weight))

	return output

def computeAllProfiles(runName, outPickleName, quantities=['Mdot', 'rho', 'u', 'T', 'u^r', 'u^phi'], mass_weight=True):
	"""
	Loop through every file of a given run.  Compute profiles, then save a dictionary to a pickle.
	"""

	subFolders = np.array(os.listdir(runName))
	runIndices = np.array([int(name.split('_')[-1]) for name in subFolders])
	order = np.argsort(runIndices)
	subFolders = subFolders[order]
	runIndices = runIndices[order]

	#Collect all profiles.  This is a good place to parallelize if desired later.
	profiles = []
	radii = []
	for runIndex in runIndices:
		finalFile = glob.glob(os.path.join(runName, subFolders[runIndex], '*.final.phdf'))[0]
		print(f"Loading {finalFile}...")
		dump = pyharm.load_dump(os.path.join(finalFile))
		radii.append(dump['r1d'])
		profiles.append(computeProfileSet(dump, quantities=quantities, mass_weight=mass_weight))
		
	#Create a dictionary to save.
	D = {}
	D['input'] = runName
	D['runIndices'] = runIndices
	D['folders'] = subFolders
	D['radii'] = radii
	D['profiles'] = profiles
	D['quantities'] = quantities

	with open(outPickleName, 'wb') as openFile:
		pickle.dump(D, openFile, protocol=2)
	print(f"Output saved to {outPickleName}.")

if __name__ == '__main__':
	#Input and output locations.
	grmhdLocation = '/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/data'
	dataLocation = '../data_products'

	# For example...
	# python computeProfiles.py bondi_multizone_121322_gizmo_3d_ext_g
	import sys
	run = sys.argv[1]

	inName = os.path.join(grmhdLocation, run)
	outName = os.path.join(dataLocation, run + '_profiles.pkl')
	computeAllProfiles(inName, outName)
