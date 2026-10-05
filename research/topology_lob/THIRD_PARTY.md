# Third-party provenance

The supplied project configuration identifies:
- https://github.com/giotto-ai/giotto-tda
- https://github.com/artemmavrin/focal-loss
- https://github.com/scikit-learn-contrib/imbalanced-learn

The root repository retains the prefixed/vendored source trees and their license files. The research application uses released packages and documented APIs instead of importing the mechanically altered module names.

New code under research/topology_lob is project-specific orchestration, L2 validation, feature engineering, topology construction, fractional differentiation, model objective, evaluation, tests, and benchmarks.