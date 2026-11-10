"""Baseline planners.

Each baseline depends on an optional third-party planner, so all of them are
imported defensively: a partial install still exposes the baselines you can
actually run. A baseline whose dependencies are missing is still bound to a name,
and raises ImportError only when called. Use ``AVAILABLE`` to see what was
importable.
"""


def _unavailable(name, requires):
	"""Return a stand-in for a baseline whose optional dependencies are missing."""

	def _raise(*args, **kwargs):
		raise ImportError(
			f"{name} is unavailable in this interpreter: it requires {requires}. "
			f"See 'Optional back-ends' in the README. "
			f"Available here: {AVAILABLE}.")

	_raise.__name__ = name
	return _raise


__all__ = []

# Requires: drake (pydrake), MOSEK
try:
	from baseline.gcs.gcs_soln import planWithGCS
	__all__.append('planWithGCS')
except ImportError:
	planWithGCS = _unavailable('planWithGCS', 'drake (pydrake) and a MOSEK licence')

# Requires: python-fcl
try:
	from baseline.custom.prm import planWithCustomPRM
	__all__.append('planWithCustomPRM')
except ImportError:
	planWithCustomPRM = _unavailable('planWithCustomPRM', 'python-fcl')

# Requires: klampt, PythonCDT
try:
	from baseline.deprecated.klampt_soln import planWithKlampt
	__all__.append('planWithKlampt')
except ImportError:
	planWithKlampt = _unavailable('planWithKlampt', 'klampt and PythonCDT')

# Requires: ompl (with Python bindings)
try:
	from baseline.ompl.ompl_soln import planWithOMPL
	__all__.append('planWithOMPL')
except ImportError:
	planWithOMPL = _unavailable('planWithOMPL', 'ompl with its Python bindings')

AVAILABLE = tuple(__all__)
