"""Environment back-ends.

``CleanEnv`` is the only back-end required by the method in the paper; it depends
on nothing beyond the core scientific stack. Every other back-end pulls in an
optional third-party planner (python-fcl, Klampt + PythonCDT, Drake, OMPL) and is
therefore imported defensively, so that a partial install still lets you run the
MILP planner and whichever baselines you did install.

A back-end whose dependencies are missing is still bound to a name here, so that
``from environment import DrakeEnvironment`` keeps working; the ImportError is
deferred until something actually tries to construct it. Use ``AVAILABLE`` to see
what was importable in the current interpreter.
"""


def _unavailable(name, requires):
	"""Return a stand-in for a back-end whose optional dependencies are missing."""

	def _raise(*args, **kwargs):
		raise ImportError(
			f"{name} is unavailable in this interpreter: it requires {requires}. "
			f"See 'Optional back-ends' in the README. "
			f"Available here: {AVAILABLE}.")

	_raise.__name__ = name
	return _raise


from .clean_env import CleanEnv

__all__ = ['CleanEnv']

# Requires: python-fcl
try:
	from .collision_checking_env import CollisionCheckingEnvironment
	__all__.append('CollisionCheckingEnvironment')
except ImportError:
	CollisionCheckingEnvironment = _unavailable('CollisionCheckingEnvironment', 'python-fcl')

# Requires: klampt, PythonCDT
try:
	from .klampt_env import KlamptEnvironment
	__all__.append('KlamptEnvironment')
except ImportError:
	KlamptEnvironment = _unavailable('KlamptEnvironment', 'klampt and PythonCDT')

# Requires: drake (pydrake)
try:
	from .drake_env import DrakeEnvironment
	__all__.append('DrakeEnvironment')
except ImportError:
	DrakeEnvironment = _unavailable('DrakeEnvironment', 'drake (pydrake)')

# Requires: ompl (with Python bindings), python-fcl
try:
	from .ompl_env import OMPLEnvironment
	__all__.append('OMPLEnvironment')
except ImportError:
	OMPLEnvironment = _unavailable('OMPLEnvironment', 'ompl with its Python bindings, and python-fcl')

AVAILABLE = tuple(__all__)
