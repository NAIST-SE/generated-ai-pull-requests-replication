# Preprocessing source of truth

The authoritative sampling workflow for the manuscript is
`sampling_and_distribution.ipynb`. It produces the 308-PR sample with fixed random seed
23 after applying the documented filters and upper-tail IQR exclusions. Manual review
confirmed that the retained descriptions contain no near-empty cases.

`filter_and_sample_prs.py` is an earlier fixed-tier prototype (50 PRs per tier, at most
150 total). It was not used for the manuscript dataset and is retained only as
historical code. Do not cite or run it as the reproduction path.

`merge_additional_original_prs.py` merges source exports consumed before the
authoritative notebook. It does not define the sampling procedure.
