"""Public data behind the figures of the portfolio page "Medical Assistant", saved to data/ so that the figures
render without the network.

- PubMed counts from NCBI E-utilities: articles per year of publication, 1990-2018, all of them and those with the
  publication types "Randomized Controlled Trial" and "Meta-Analysis"; and, for metformin in type 2 diabetes up to
  2018, the number of articles of each publication type. Counts are as indexed on the day of the fetch.
- A piece of Hetionet v1.0 (Himmelstein et al., eLife 2017), a biomedical knowledge graph published as a Neo4j
  database: type 2 diabetes with the compounds that treat it, the symptoms it presents and the diseases it
  resembles. The original content of Hetionet is CC0; its edges come from public resources credited in the paper.

    uv run python articles/medical_assistant/fetch.py --hetionet <folder with hetionet-v1.0-nodes.tsv and
        hetionet-v1.0-edges.sif.gz from https://github.com/hetio/hetionet/raw/v1.0.0/hetnet/tsv/>
"""

import argparse
import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
YEARS = range(1990, 2019)
TYPES = {"all": "", "randomized controlled trials": " AND randomized controlled trial[pt]",
         "meta-analyses": " AND meta-analysis[pt]"}
EVIDENCE = ["Meta-Analysis", "Randomized Controlled Trial", "Clinical Trial", "Observational Study",
            "Comparative Study", "Review", "Case Reports"]
QUESTION = 'metformin[tiab] AND "diabetes mellitus, type 2"[mh] AND 1990:2018[dp]'
DISEASE = "Disease::DOID:9352"  # type 2 diabetes mellitus
DRUGS = ["metformin", "sitagliptin", "linagliptin", "pioglitazone", "glimepiride", "gliclazide", "acarbose"]
T2D = '"diabetes mellitus, type 2"[mh]'
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def count(term):
    query = urllib.parse.urlencode({"db": "pubmed", "term": term, "rettype": "count", "retmode": "json"})
    with urllib.request.urlopen(f"{EUTILS}?{query}", timeout=60) as response:
        time.sleep(0.4)  # NCBI asks for at most three requests a second without a key
        return int(json.load(response)["esearchresult"]["count"])


def pubmed():
    per_year = {name: {y: count(f"{y}[dp]{extra}") for y in YEARS} for name, extra in TYPES.items()}
    evidence = {pt: count(f"{QUESTION} AND {pt}[pt]") for pt in EVIDENCE}
    evidence["any type"] = count(QUESTION)
    return {"per_year": per_year, "evidence": evidence, "question": QUESTION,
            "fetched": time.strftime("%Y-%m-%d")}


def assistant():
    """Numbers for the drawing of the assistant's screen: for each drug, randomized trials and meta-analyses in type
    2 diabetes up to 2018; and three trials or meta-analyses printed in 2018 with metformin in the title."""
    drugs = {d: {"trials": count(f"{d}[tiab] AND {T2D} AND randomized controlled trial[pt] AND 1990:2018[dp]"),
                 "meta": count(f"{d}[tiab] AND {T2D} AND meta-analysis[pt] AND 1990:2018[dp]")} for d in DRUGS}
    term = f"metformin[tiab] AND {T2D} AND (meta-analysis[pt] OR randomized controlled trial[pt]) AND 2018[dp]"
    query = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmax": 60, "sort": "pub_date",
                                    "retmode": "json"})
    with urllib.request.urlopen(f"{EUTILS}?{query}", timeout=60) as response:
        ids = json.load(response)["esearchresult"]["idlist"]
    time.sleep(0.4)
    query = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    with urllib.request.urlopen(f"{ESUMMARY}?{query}", timeout=60) as response:
        summaries = json.load(response)["result"]
    papers = [{"pmid": i, "title": summaries[i]["title"], "journal": summaries[i]["source"],
               "date": summaries[i]["pubdate"], "types": summaries[i]["pubtype"]} for i in ids
              if summaries[i]["pubdate"].startswith("2018") and "metformin" in summaries[i]["title"].lower()]
    return {"drugs": drugs, "papers": papers[:3]}


def hetionet(folder):
    names, kinds = {}, {}
    with open(folder / "hetionet-v1.0-nodes.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            node, name, kind = line.rstrip("\n").split("\t")
            names[node], kinds[node] = name, kind
    edges = []
    with gzip.open(folder / "hetionet-v1.0-edges.sif.gz", "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            source, kind, target = line.rstrip("\n").split("\t")
            if DISEASE in (source, target) and kind in ("CtD", "CpD", "DpS", "DrD"):
                edges.append((source, kind, target))
    nodes = {n for e in edges for n in (e[0], e[2])}
    return {"disease": DISEASE, "nodes": {n: {"name": names[n], "kind": kinds[n]} for n in nodes},
            "edges": edges}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hetionet", type=Path, required=True)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    (DATA / "hetionet_t2d.json").write_text(json.dumps(hetionet(args.hetionet), indent=1))
    (DATA / "pubmed_counts.json").write_text(json.dumps(pubmed(), indent=1))
    (DATA / "assistant.json").write_text(json.dumps(assistant(), indent=1))
    print("wrote", DATA / "hetionet_t2d.json", DATA / "pubmed_counts.json", DATA / "assistant.json")


if __name__ == "__main__":
    main()
