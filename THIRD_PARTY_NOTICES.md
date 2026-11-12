# Third-party code

- **OpenPI**: the source under `openpi/` is a subset of
  [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi),
  commit `215abfb217dbac7d5f1273282331b9b1866c0479`. FR3 transforms and training
  configurations were added for this project. The Apache 2.0 license is preserved
  in `openpi/LICENSE`; the supplied Gemma terms are in `openpi/LICENSE_GEMMA.txt`.
- **DataHive**: `config/datahive_tasks.yaml` comes from
  [HiveBoard DataHive](https://github.com/hiveboard-bench/DataHive), commit
  `1ccea29e5b8de510bfe311ca9100b39d762f909d`. Its MIT license is preserved in
  `config/DATAHIVE_LICENSE`. Integration tests use DataHive's public recording
  and annotation APIs from the checkout selected by `DATAHIVE_ROOT`.

Other dependencies are recorded in `openpi/uv.lock` and retain their own licenses.
