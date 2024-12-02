Generating combinations is failing

The only requirements are:
- The use_ocr ProcessingStep must stay on (but its parameter can be toggled)
- When we write best_parameters, we should only write the ProcessingStep and Parameter values that were "on" when run.
- We should strive to cache and reuse parameter information as much as possible using functools and such

```
I'm in the middle of implementing a new `ProcessingState` class in `extractor.py` to handle immutable dictionaries of `ProcessingStep` and `Parameter` combinations for caching. Can you review the script for `extractor` and make sure I haven't left anything redundant in with the advent of this new class? Do not focus on `optimizer`/`ParameterOptimizer` right now.

Assume that if a dictionary is going to be passed in and cached, it's going to look like the nested structure seen for `ProcessingState` and `params_override`. The idea is for it to look as similar to `params.yml` as possible for the end user. We also want to sort and preserve space by only caching or processing or running `ProcessingStep` and corresponding `Parameters` that are `enabled`.

Furthermore, we should stick to "Enabled" and "Disabled" language in the viewer and the docs and the script whenever possible (not "On", "Off")

Finally, the "use_" prefix logic is now outdates. We should assume that `params_override` and the initial parameter structure in general should follow the same nested pattern seen in `params.yml`, which we will instantiate as a dictionary.
```