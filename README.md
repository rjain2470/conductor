# Conductor 🔋

Conductor is a natural language interface to the LMFDB (L-functions and Modular Forms Database). It translates mathematical questions into SQL, executes them against the LMFDB PostgreSQL database, and returns structured data with optional exploratory analysis and plots.

For instance, rather than navigating the LMFDB or constructing SQL queries by hand, a mathematician can ask Conductor:

"Can you plot the regulator against the conductor for the rank-1 elliptic curves over Q with conductor under 10,000 on a log-log scale?"

"Plot the real period vs the analytic order of Sha for elliptic curves of rank 2 with conductor under 5,000."

"Which semistable elliptic curves have prime conductor under 500 and non-trivial torsion? Show me the distribution of torsion subgroup structures."

"I'm interested in the relationship between regulator and discriminant for totally real cubic fields of class number 1 — can you pull those and plot them on a log-log scale?"

"Give me a table of the weight-2 newforms with CM at squarefree levels under 500."


# Database coverage 📊
The LMFDB contains the following 86 tables across 16 mathematical domains:
DomainTablesClassical modular formsmf_newforms, mf_newspaces, mf_hecke_, mf_twists_, mf_gamma1, mf_starkMaass formsmaass_newforms, maass_rigor (and coefficient tables)Hilbert / Bianchi / Siegel modular formshmf_forms, bmf_forms, smf_samples (and auxiliary tables)Other modular formshalfmf_forms, modlmf_forms, modlgal_repsL-functionslfunc_lfunctions, lfunc_search, lfunc_instancesElliptic curves over Qec_curvedata, ec_mwbsd, ec_localdata, ec_classdata, ec_galrep, ec_padic, ec_iwasawa, ec_torsion_growthElliptic curves over number fieldsec_nfcurvesGenus-2 curvesg2c_curves, g2c_endomorphisms, g2c_galrep, g2c_ratpts, g2c_tamagawaAbelian varieties over finite fieldsav_fq_isog, av_fq_endalg_data, av_fq_endalg_factorsNumber fieldsnf_fields, nf_fields_extra, nf_fields_reflexLocal fields and finite fieldslf_fields, lf_families, fq_fieldsArtin representationsartin_reps, artin_field_dataDirichlet characterschar_dirichletHypergeometric motiveshgm_families, hgm_motives, hgm_monodromy, hgm_euler_surveyModular curvesmodcurve_models, modcurve_points, modcurve_modelmapsGroupsgps_groups, gps_transitive, gps_st, gps_char, gps_small (and more)Lattices and otherlat_lattices, cluster_pictures, hgcwa_passports, belyi_galmaps, etc.

# Setup 

## Prerequisites 🟡
Python 3.12+
Access to the LMFDB PostgreSQL mirror (default: devmirror.lmfdb.xyz)

## Installation 🔌
TBD.

# Architecture 🏛️
Conductor is a five-stage FastAPI pipeline with graceful error handling. As of June 2026, we use claude-sonnet-4-6 for every step apart from the second one, which instead uses the cheaper claude-haiku-4.5 to optimize for latency and token spend.

1. An LLM-as-judge assesses query precision before any database interaction. If the query is ambiguous in a way that would materially change what is queried, it asks one focused question. If clear, it returns a refined restatement passed to all subsequent stages.
2. An LLM maps the query to a list of relevant LMFDB table names using a two-layer hierarchical schema index.
3. Our LLM produces a validated SQL query using only the schema slice for the tables identified in Stage 2, keeping prompt size proportional to query complexity. Correctness is enforced by using our preloaded schema as a ground truth.
4. We run the SQL over a read-only SQLAlchemy connection with a 15-second timeout, returning a pandas DataFrame.
5. (optional) We translate a follow-up natural language instruction into Python, which is executed in a restricted exec() namespace. Plots are captured in-memory and returned as base64-encoded PNGs.

## Project structure 🏗️
conductor/
├── main.py                  # FastAPI app, endpoints, auth, rate limiting.
├── pipeline/
│   ├── router.py            # Stage 2: NL query → table names 
│   ├── sql_gen.py           # Stage 3: query + tables → validated SQL
│   ├── executor.py          # Stage 4: SQL → DataFrame
│   ├── analysis.py          # Stage 5: instruction + DataFrame → plot
│   └── chat.py              # Orchestrator: session state, error handling
├── schema/
│   ├── lmfdb_schema.json    # Full schema: 86 tables, 2,006 columns
│   └── routing_index.json   # Two-layer routing index
├── prompts/
│   ├── sql_prompt.txt       # SQL generation system prompt
│   ├── analysis_prompt.txt  # Analysis generation system prompt
│   └── analysis_style.txt   # Plot style guide
├── tests/
│   ├── test_router.py
│   ├── test_sql_gen.py
│   ├── test_executor.py
│   └── test_analysis.py
├── .env.example
├── requirements.txt
└── README.md

# Limitations 🟥
- The server connects to devmirror.lmfdb.xyz, which may only have partial coverage compared to the full LMFDB. Moreover, since the LMFDB itself is not fully comprehensive, some data may be unavailable.
- Queries are subject to API rate limits; responses may slow under heavy load.
-  Conductor is under active development. You may encounter occasional errors or unexpected behaviour — if you do, please open a GitHub issue to report it. 

# Acknowledgements 🌲
This work would be impossible without the work of hundreds of mathematicians in creating the LMFDB. See lmfdb.org/acknowledgment for the full list of contributors.
