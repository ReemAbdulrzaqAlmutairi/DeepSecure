# Dataset Access Instructions

## VUDENC Datasets (SQL, XSS, Command Injection, RCE)

Download from the original VUDENC repository:

- **Zenodo:** https://zenodo.org/record/5903630
- **Files needed:** `plain_sql.json`, `plain_xss.json`, `plain_command_injection.json`, `plain_remote_code_execution.json`
- Place them in this `data/` directory.

**Citation:**
> Wartschinski, L., et al. (2022). VUDENC: Vulnerability Detection with Deep Learning on a Natural Codebase. *Information and Software Technology*, 144, 106809.

---

## Novel Datasets (Broken Authentication & Hard-coded Credentials)

Collected by the authors of DeepSecure using the same commit-based methodology as VUDENC, with new keyword sets targeting two previously unbenchmarked vulnerability types.

- **Zenodo:** https://doi.org/10.5281/zenodo.19152451 (DOI: 10.5281/zenodo.19152451)
- **Files:** `plain_broken_authentication.json`, `plain_use_of_hardcoded_credentials.json`
- Place them in this `data/` directory.

### Collection methodology:
1. `data_collection/broken_authentication/scrapingGithub.py` — scrape commits
2. `data_collection/broken_authentication/filterShowcases.py` — filter CTF/showcase repos
3. `data_collection/broken_authentication/getDiffs.py` — download diffs
4. `data_collection/broken_authentication/getData.py` — extract source code + labels

Same steps apply for `hardcoded_credentials/`.

---

## Data Format

Each `plain_*.json` file has the structure:
```json
{
  "https://github.com/repo/name": {
    "commit_sha": {
      "keyword": "fix sql injection",
      "diff": "...",
      "files": {
        "/path/to/file.py": {
          "source": "...",
          "changes": [{"badparts": [...], "goodparts": [...]}]
        }
      }
    }
  }
}
```

## License

Novel datasets are released under **CC BY 4.0**.  
Please cite both DeepSecure and VUDENC when using these datasets.

