"""verl integration layer for MosaicGym.

Thin adapters that bridge our toolbox / scoring to verl's RL training, kept in
our own package (not in the vendored verl tree) so verl stays pristine. Loaded
by verl via dotted path / file path — see rl/env and the training configs.
"""
