"""
Converts variable names into LaTeX.
"""

ylabel_dictionary = {}
ylabel_dictionary['u^r'] = '$u^r$'
ylabel_dictionary['u^phi'] = '$u^\phi$'
ylabel_dictionary['u^th'] = r'$u^\theta$'
ylabel_dictionary['abs_u^r'] = '$|u^r|$'
ylabel_dictionary['abs_u^phi'] = '$|u^\phi|$'
ylabel_dictionary['abs_u^th'] = r'$|u^\theta|$'
ylabel_dictionary['rho'] = r'$\rho$ [arbitrary units]'
ylabel_dictionary['Mdot'] = r'$\dot{M}$ [arbitrary units]'
ylabel_dictionary['eta'] = r'$\eta = 1-\dot{E}/\dot{M}$'
ylabel_dictionary['beta'] = r'$\beta$'
ylabel_dictionary['sigma'] = r'$\sigma$'
ylabel_dictionary['Omega'] = '$\Omega$'

def variableToLabel(variable):
  try:
    return ylabel_dictionary[variable]
  except KeyError:
    return variable
