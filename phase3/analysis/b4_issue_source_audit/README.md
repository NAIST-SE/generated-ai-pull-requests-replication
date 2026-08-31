# B4 issue-source audit

This server-only audit repairs the source-provenance layer before any corrected P5/P6
generation. It does not regenerate descriptions or alter frozen inputs.

The audit:

1. verifies the frozen source hashes;
2. losslessly decomposes the aggregated issue fields into one row per PR-body reference;
3. verifies that each number occurs in the corresponding PR body;
4. classifies the stored GitHub artifact as an issue or pull request from its canonical URL;
5. retains only genuine issues with non-empty stored titles and bodies in a deterministic
   regeneration manifest, while recording empty fields on excluded pull-request artifacts; and
6. optionally checks current GitHub identity/type and content-hash drift with authenticated REST
   requests. Current API text never replaces the stored snapshot.

For PR-body evidence, the audit follows the authoritative sampling notebook by accepting both
the canonical `JabRef/jabref` URL and the historical `koppor/jabref` URL. Artifact type is still
determined only from the canonical URL returned in the stored fetch.

Run synthetic tests anywhere:

```bash
python -m unittest -v test_audit_utils.py
```

Run the real audit only on `nokogiri`:

```bash
/home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/analysis/b4_issue_source_audit/audit_issue_sources.py \
  --with-api --tokens-file /server/path/to/tokens.txt
```

The script refuses to overwrite its versioned output directory and writes atomically only after
all selected gates pass. Token values are never written to output; only the supplied token-file
path is recorded.
