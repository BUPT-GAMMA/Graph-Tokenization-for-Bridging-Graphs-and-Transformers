# Compatibility shims for libraries expecting legacy symbols in `collections`
# This file is auto-imported by Python if present.
try:
	import collections as _collections
	import collections.abc as _abc
	for _name in ("Mapping", "MutableMapping", "Sequence", "Iterable"):
		try:
			getattr(_collections, _name)
		except AttributeError:
			setattr(_collections, _name, getattr(_abc, _name))
except Exception:
	# Fail silently; tests may still proceed if not needed
	pass
