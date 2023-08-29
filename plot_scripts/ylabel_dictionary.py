"""
Converts variable names into LaTeX.
"""

ylabel_dictionary = {}
ylabel_dictionary['u^r'] = r'$\langle u^r \rangle$'
ylabel_dictionary['u^phi'] = '$u^\phi$'
ylabel_dictionary['u^th'] = r'$u^\theta$'
ylabel_dictionary['abs_u^r'] = '$|u^r|$'
ylabel_dictionary['abs_u^phi'] = '$|u^\phi|$'
ylabel_dictionary['abs_u^th'] = r'$|u^\theta|$'
ylabel_dictionary['rho'] = r'$\overline{\langle \rho \rangle_1}$ [arb. units]'
ylabel_dictionary['Mdot'] = r'$\overline{\dot{M}}$ [arb. units]'
#ylabel_dictionary['eta'] = r'$\eta = 1-\dot{E}/\dot{M}$'
ylabel_dictionary['eta'] = r'$\eta = \overline{\dot{M}-\dot{E}}/\overline{\dot{M}}_{10}$'
ylabel_dictionary['etaMdot'] = r'$\dot{M}-\dot{E}$'
ylabel_dictionary['beta'] = r'$\langle\beta^{-1}\rangle ^{-1}$'
ylabel_dictionary['sigma'] = r'$\sigma$'
ylabel_dictionary['abs_Omega'] = r'$\langle|\Omega|\rangle /\Omega_K$'
ylabel_dictionary['Pg'] = r'$p_{\rm g}$'
ylabel_dictionary['T'] = r'$\langle\Theta = k\,T/(\mu\, c^2) \rangle$'

def variableToLabel(variable):
  try:
    return ylabel_dictionary[variable]
  except KeyError:
    return variable
