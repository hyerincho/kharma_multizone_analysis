import pyharm
import pickle
import numpy as np
import os
import glob
import pdb

def shellAverage(dump, quantity, imin=0, mass_weight=True):
  """
  Starting with a pyharm dump object, average some quantity with respect to phi and theta.
  Includes some smart parsing of quantities that are not keys in the dump object.
  """

  #Many quantities we want are already keys, but some are not.  For those, we must come up with our own formula.
  if quantity == 'Mdot':
    #This is the weirdest one.  We won't average ourselves, and we'll use a built-in pyharm function.
    return -pyharm.shell_sum(dump, 'FM')
  if quantity == 'T':
    to_average = dump['u'] / dump['rho'] * (dump['gam']-1)
  else:
    to_average = dump[quantity]

  #Weighting for the average.
  volumetric_weight = dump['gdet']
  if mass_weight:
    density = dump['rho']
  else:
    #Obviously it isn't, but basically we just want to pretend this doesn't exist.
    density = 1.0

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

def computeAllProfiles(runName, outPickleName, quantities=['Mdot', 'rho', 'u', 'T', 'u^r', 'u^phi'], mass_weight=True, final_only=True, onezone_override=False):
  """
  Loop through every file of a given run.  Compute profiles, then save a dictionary to a pickle.
  """

  #subFolders = np.array([name for name in glob.glob(runName+"/*/") if name.split('/')[-2].split('_')[0] == 'bondi'])  #POTENTIALLY CHANGE THIS CONDITION WITH NEW NAMING SCHEME
  #runIndices = np.array([int(name.split('_')[-1][:-1]) for name in subFolders])
  subFolders = np.array([name for name in glob.glob(runName+"/*[0-9][0-9][0-9][0-9][0-9]/")])
  runIndices = np.array([int(name[-6:-1]) for name in subFolders])
  order = np.argsort(runIndices)
  subFolders = subFolders[order]
  runIndices = runIndices[order]
  # the arrays that will be used for calculation
  subFolders_calc = np.copy(subFolders)
  runIndices_calc = np.copy(runIndices)

  #Collect all profiles.  This is a good place to parallelize if desired later.
  if onezone_override:
    nzone = 1
  else:
    nzone = None
  r_sonic = None
  base = None
  listOfListOfProfiles = []
  listOfListOfTimes= []
  radii = []
  zones = []

  # if file exists, don't do the whole calculation
  if os.path.isfile(outPickleName):
    with open(outPickleName, 'rb') as openFile:
      D_read = pickle.load(openFile)
    existing_idx = D_read["runIndices"]
    to_del =[idx for (idx, value) in enumerate(runIndices) if value in existing_idx] # indices to delete
    if len(set(existing_idx)-set(runIndices_calc))>0:
      print("WARNING: There are some runs in the pre-existing file that are not present now. This might screw up the calculation.")
      pdb.set_trace()
    runIndices_calc = np.delete(runIndices_calc,to_del)
    subFolders_calc = np.delete(subFolders_calc,to_del)
    print("file exists for runs {}-{}. starting from {}".format(existing_idx[0], existing_idx[-1], runIndices[0]))

    # override with previously calculated data
    listOfListOfProfiles = D_read["profiles"]
    listOfListOfTimes = D_read["times"]
    radii = D_read["radii"]
    zones = D_read["zones"]

  for i,runIndex in enumerate(runIndices_calc):
    allFiles = sorted(glob.glob(os.path.join(subFolders_calc[i], '*.phdf')))
    if final_only:
      allFiles = [allFiles[-1]]
    listOfProfiles = []
    listOfTimes = []
    radiiForThisAnnulus = None
    for file in allFiles:
      print(f"Loading {file}...")
      dump = pyharm.load_dump(os.path.join(file))

      #Just need these values once for the entire simulation.
      if nzone is None:
        nzone = dump['nzone']
      if r_sonic is None:
        r_sonic = dump['rs']
      if base is None:
        base = int(dump['base'])
      
      listOfProfiles.append(computeProfileSet(dump, quantities=quantities, mass_weight=mass_weight))
      listOfTimes.append(pyharm.io.get_dump_time(file))

    #Just need this value once for a given annulus.
    radii.append(dump['r1d'])
    zone = (nzone-1) - int(np.log(dump['r_in'])/np.log(base))
    zones.append(zone)
    listOfListOfProfiles.append(listOfProfiles)
    listOfListOfTimes.append(listOfTimes)

  #Sometimes one-zone models need to restart and it looks like there were multiple zones by the directory structure.
  #In this case, merge all zones.
  if onezone_override: # HYERIN: I don't know how my new changes have affected this
    listOfListOfProfiles = [[item for sublist in listOfListOfProfiles for item in sublist]]
    listOfListOfTimes = [[item for sublist in listOfListOfTimes for item in sublist]]
    radii = [radii[0]] 

  #Create a dictionary to save.
  D = {}
  D['input'] = runName
  D['runIndices'] = runIndices
  D['folders'] = subFolders
  D['radii'] = radii
  D['quantities'] = quantities
  D['runName'] = runName
  D['nzone'] = nzone
  D['profiles'] = listOfListOfProfiles
  D['times'] = listOfListOfTimes
  D['r_sonic'] = r_sonic
  D['zones'] = zones
  D['base'] = base

  with open(outPickleName, 'wb') as openFile:
    pickle.dump(D, openFile, protocol=2)
  print(f"Output saved to {outPickleName}.")

if __name__ == '__main__':
  #Input and output locations.
  grmhdLocation = '../data'
  dataLocation = '../data_products'

  # For example...
  # python computeProfiles.py bondi_multizone_121322_gizmo_3d_ext_g
  import sys
  run = sys.argv[1]

  inName = os.path.join(grmhdLocation, run)
  outName = os.path.join(dataLocation, run + '_profiles_all.pkl')
  computeAllProfiles(inName, outName, quantities=['Mdot', 'rho', 'u', 'T', 'abs_u^r', 'u^phi', 'u^th', 'u^r','abs_u^th', 'abs_u^phi', 'u^t', 'b', 'inv_beta', 'beta'], final_only=False, onezone_override=('onezone' in inName))
