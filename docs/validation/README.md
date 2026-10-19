# Manual validation

Three variables in this study come from heuristics, so they were checked by hand on random
samples. The samples were drawn by script with a fixed seed, so anyone can draw the same issues
again. They are blind: the sheets show the issue but not the study's own classification, which
is stored separately in `results/tables/validation_*_key.csv`.

| Sample | Issues | Drawn by | What it checks |
|---|---|---|---|
| `V1_attribution` | 99, nine per tested agent | `src/10_validation.py` | Whether the agent label (`agent_type`) is right, and the agent's role in the issue; this label underlies all three research questions |
| `V2_topic_labels` | 25, five per fine topic | `src/07_rq3_topics.py` | Whether each RQ3 topic label describes the issues of that topic |
| `V3_language` | 40, half flagged non-English, half English | `src/10_validation.py` | The language flag that decides which issues enter RQ3 |

The author's answers are in `VALIDATION_FILLED_BY_AUTHOR.xlsx`, one tab per sample. The blank
samples are also available as `V1_attribution_sheet.csv`, `V2_topic_labels_sheet.csv` and
`V3_language_sheet.csv`. `src/10_validation.py` scores the answers against the keys and writes
`results/tables/validation_summary.csv`, and `checks/check_08_validation.py` recomputes those
scores independently. The results are reported in the Threats to Validity section of the report.

To re-rate the samples, fill in a copy of the workbook and run
`python checks/check_08_validation.py <your copy>`; it prints the scores of the copy so they can
be compared with the author's. Rating takes about a minute per issue, with the GitHub page open
whenever the title and excerpt are not enough.

## Coding scheme

### V1: attribution

`agent_involved` asks whether the issue is really linked to the agent named in the column
`agent_in_dataset`.

| Code | Meaning |
|---|---|
| Y | Yes, in one of the roles below |
| N | No visible link to that agent; the attribution is wrong |
| U | Cannot tell, because the issue was deleted or is private, or the link is ambiguous |

`role` records how the agent is linked, only when the answer is Y. When several apply, the
strongest is chosen, in the order W, T, M.

| Code | Meaning | Typical evidence |
|---|---|---|
| W, wrote | The agent created the issue | Opened by the agent's own account, such as `coderabbitai[bot]`, or text saying it was generated with the agent ("Generated with Claude Code") |
| T, triggered | A person asked or assigned the agent to act | A `sweep` label, a "Sweep:" title prefix, an `@agent` command |
| M, mentioned | The agent is only discussed | A bug report about the agent product itself, or a digest that lists it |
| O, other | Linked in some other way | Explained in the notes |

### V2: topic labels

`fits_label` records whether the proposed label describes what the issue is about, judged on the
issue as a whole rather than on single words (Y or N). Topic 2 has no label, so its issues are
described in the notes instead.

### V3: language

`language` records whether the title and body are written in English: EN for English, NOT_EN
for mainly another language, and MIXED when large parts are in both, with the dominant language
noted.
