Generating combinations is failing

The only requirements are:
- The use_ocr ProcessingStep must stay on (but its parameter can be toggled)
- When we write best_parameters, we should only write the ProcessingStep and Parameter values that were "on" when run.
- We should strive to cache and reuse parameter information as much as possible using functools and such