def grade(**kwargs) -> dict:
    """
    Grade the Kaiming He 2022 conference paper compilation task.

    Notes:
      - Titles are treated as unique paper identifiers in the hidden reference.
      - For homepage matching, http/https differences are ignored after URL normalization.
      - For author name matching, accented characters (e.g. á) and ASCII equivalents (e.g. a) are treated as equivalent.
    """
    from pathlib import Path
    from urllib.parse import urlsplit, urlunsplit
    import csv
    import io
    import re
    import unicodedata

    ALL_CRITERIA = [
        "output_exists",
        "tsv_header_valid",
        "rows_parseable",
        "paper_recall",
        "paper_precision",
        "paper_f1",
        "row_sorting_correct",
        "conference_accuracy",
        "authors_accuracy",
        "abstract_accuracy",
        "author_links_format_valid",
        "author_links_accuracy",
        "github_commit_id_format_valid",
        "github_commit_id_accuracy",
        "tex_files_created",
        "tex_exact_match_ratio",
        "output_dir_clean",
        "hard_constraint_pass",
        "overall_score",
    ]

    ZERO = {k: 0.0 for k in ALL_CRITERIA}
    scores = dict(ZERO)
    
    workspace = Path("/tmp_workspace")
    gt_dir = workspace / "gt"
    pred_path = workspace / "results" / "2022.tsv"
    pred_dir = workspace / "results"
    gt_path = gt_dir / "gt.tsv"
    gt_tex_dir = gt_dir / "gt_tex"
    if not (gt_tex_dir.exists() and gt_tex_dir.is_dir()):
        gt_tex_dir = gt_dir

    if not gt_path.exists():
        return ZERO

    def normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text))
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize_casefold_text(text: str) -> str:
        return normalize_text(text).casefold()

    def normalize_author_name(name: str) -> str:
        """Treat accented chars (e.g. á) and ASCII equivalents (e.g. a) as equivalent."""
        s = normalize_text(name)
        nfd = unicodedata.normalize("NFD", s)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    def normalize_conference(text: str) -> str:
        text = normalize_casefold_text(text)
        text = re.sub(r"[^a-z0-9]+", "", text)
        return text

    def normalize_url(text: str) -> str:
        text = normalize_text(text)
        if normalize_casefold_text(text) == "not found":
            return "not found"
        try:
            parsed = urlsplit(text)
            netloc = parsed.netloc.lower()
            path = re.sub(r"/+", "/", parsed.path or "/")
            if path != "/":
                path = path.rstrip("/")
            else:
                path = "/"
            # Ignore http/https difference when comparing homepages.
            return urlunsplit(("", netloc, path, "", ""))
        except Exception:
            return re.sub(r"^https?://", "", text.rstrip("/"), flags=re.IGNORECASE)

    def normalize_commit_id(text: str) -> str:
        text = normalize_casefold_text(text)
        if text == "not found":
            return "not found"
        text = re.sub(r"\s+", "", text)
        return text

    def is_valid_commit_id(text: str) -> bool:
        value = normalize_commit_id(text)
        return value == "not found" or re.fullmatch(r"[0-9a-f]{7,40}", value) is not None

    def commit_id_matches(pred: str, gt: str) -> bool:
        pred_norm = normalize_commit_id(pred)
        gt_norm = normalize_commit_id(gt)
        if pred_norm == "not found" or gt_norm == "not found":
            return pred_norm == gt_norm
        return pred_norm.startswith(gt_norm) or gt_norm.startswith(pred_norm)

    def parse_author_list(field: str):
        names = [normalize_text(x) for x in str(field).split(",")]
        return [x for x in names if x]

    def parse_author_links(field: str):
        items = []
        chunks = [normalize_text(x) for x in str(field).split(",")]
        chunks = [x for x in chunks if x]
        for chunk in chunks:
            if ": " not in chunk:
                return None
            author, value = chunk.split(": ", 1)
            author = normalize_text(author)
            value = normalize_text(value)
            if not author or not value:
                return None
            items.append((author, value))
        return items

    def parse_tex_filename(filename: str):
        if not filename.endswith(".tex"):
            return None, None
        stem = filename[:-4]
        m = re.match(r"^(.*)_v(\d+)$", stem)
        if m:
            return m.group(1), int(m.group(2))
        return stem, None

    def normalize_tex(text: str) -> str:
        text = str(text).replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def cleanliness_score(directory: Path, allowed_names: set[str]) -> float:
        """Score agent output directory cleanliness."""
        if not directory.exists() or not directory.is_dir():
            return 0.0
        agent_output = [p.name for p in directory.iterdir()]
        extras = [n for n in agent_output if n not in allowed_names]
        return round(max(0.0, 1.0 - 0.1 * len(extras)), 4)

    scores["output_exists"] = 1.0 if pred_path.exists() and pred_path.is_file() else 0.0

    if not gt_path.exists() or not gt_path.is_file() or not gt_dir.exists():
        return scores

    expected_header = ["Conference", "Title", "Authors", "Abstract", "Author links", "GitHub commit id"]
    try:
        gt_text = gt_path.read_text(encoding="utf-8")
        gt_reader = csv.DictReader(io.StringIO(gt_text), delimiter="\t")
        gt_fieldnames = gt_reader.fieldnames or []
    except Exception:
        return ZERO

    if gt_fieldnames != expected_header:
        return ZERO

    gt_by_title = {}
    total_gt_authors = 0
    try:
        gt_reader = csv.DictReader(io.StringIO(gt_text), delimiter="\t")
        for row in gt_reader:
            if not isinstance(row, dict):
                return ZERO
            if set(row.keys()) != set(expected_header):
                return ZERO

            conference = row.get("Conference", "")
            title = row.get("Title", "")
            authors_field = row.get("Authors", "")
            abstract = row.get("Abstract", "")
            author_links_field = row.get("Author links", "")
            github_commit_id = row.get("GitHub commit id", "")

            if not all(normalize_text(x) for x in [conference, title, authors_field, abstract, author_links_field, github_commit_id]):
                return ZERO

            authors = parse_author_list(authors_field)
            links = parse_author_links(author_links_field)
            if len(authors) == 0 or links is None or len(links) != len(authors):
                return ZERO
            if not is_valid_commit_id(github_commit_id):
                return ZERO

            link_authors = [normalize_author_name(a) for a, _ in links]
            if link_authors != [normalize_author_name(a) for a in authors]:
                return ZERO

            title_key = normalize_casefold_text(title)
            if not title_key or title_key in gt_by_title:
                return ZERO

            gt_by_title[title_key] = {
                "conference": conference,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "author_links": links,
                "github_commit_id": github_commit_id,
            }
            total_gt_authors += len(authors)
    except Exception:
        return ZERO

    if len(gt_by_title) == 0:
        return ZERO

    gt_tex_files = {}
    gt_tex_names = set()
    try:
        for path in gt_tex_dir.iterdir():
            if not path.is_file() or path.suffix.lower() != ".tex":
                continue
            base_title, version = parse_tex_filename(path.name)
            if not base_title:
                return ZERO
            title_key = normalize_casefold_text(base_title)
            if title_key not in gt_by_title:
                return ZERO
            if path.name in gt_tex_files:
                return ZERO
            gt_tex_files[path.name] = {
                "title_key": title_key,
                "version": version,
                "content": normalize_tex(path.read_text(encoding="utf-8", errors="ignore")),
            }
            gt_tex_names.add(path.name)
    except Exception:
        return ZERO

    if len(gt_tex_files) == 0:
        return ZERO

    allowed_output_names = {"2022.tsv"} | gt_tex_names
    scores["output_dir_clean"] = cleanliness_score(pred_dir, allowed_output_names)

    if not pred_path.exists() or not pred_path.is_file():
        return scores

    try:
        raw_text = pred_path.read_text(encoding="utf-8")
    except Exception:
        return scores

    try:
        reader = csv.DictReader(io.StringIO(raw_text), delimiter="\t")
        fieldnames = reader.fieldnames or []
    except Exception:
        return scores

    scores["tsv_header_valid"] = 1.0 if fieldnames == expected_header else 0.0
    if scores["tsv_header_valid"] == 0.0:
        return scores

    pred_rows = []
    rows_parseable = True
    author_links_format_valid = True
    github_commit_id_format_valid = True
    duplicate_titles = False
    pred_by_title = {}

    try:
        reader = csv.DictReader(io.StringIO(raw_text), delimiter="\t")
        for row in reader:
            if not isinstance(row, dict):
                rows_parseable = False
                break

            if set(row.keys()) != set(expected_header):
                rows_parseable = False
                break

            conference = row.get("Conference", "")
            title = row.get("Title", "")
            authors_field = row.get("Authors", "")
            abstract = row.get("Abstract", "")
            author_links_field = row.get("Author links", "")
            github_commit_id = row.get("GitHub commit id", "")

            if not all(normalize_text(x) for x in [conference, title, authors_field, abstract, author_links_field, github_commit_id]):
                rows_parseable = False
                break

            authors = parse_author_list(authors_field)
            links = parse_author_links(author_links_field)
            if len(authors) == 0:
                rows_parseable = False
                break

            if links is None or len(links) != len(authors):
                author_links_format_valid = False
            else:
                link_authors = [normalize_author_name(a) for a, _ in links]
                if link_authors != [normalize_author_name(a) for a in authors]:
                    author_links_format_valid = False

            if not is_valid_commit_id(github_commit_id):
                github_commit_id_format_valid = False

            title_key = normalize_casefold_text(title)
            if title_key in pred_by_title:
                duplicate_titles = True
            pred_by_title[title_key] = {
                "conference": conference,
                "title": title,
                "authors_field": authors_field,
                "authors": authors,
                "abstract": abstract,
                "author_links_field": author_links_field,
                "author_links": links,
                "github_commit_id": github_commit_id,
            }
            pred_rows.append((conference, title))
    except Exception:
        rows_parseable = False

    scores["rows_parseable"] = 1.0 if rows_parseable else 0.0
    scores["author_links_format_valid"] = 1.0 if author_links_format_valid and rows_parseable else 0.0
    scores["github_commit_id_format_valid"] = 1.0 if github_commit_id_format_valid and rows_parseable else 0.0

    if scores["rows_parseable"] == 0.0:
        return scores

    gt_titles = set(gt_by_title.keys())
    pred_titles = set(pred_by_title.keys())
    matched_titles = gt_titles.intersection(pred_titles)

    recall = len(matched_titles) / len(gt_titles) if gt_titles else 0.0
    precision = len(matched_titles) / len(pred_titles) if pred_titles else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

    if duplicate_titles:
        precision = 0.0
        f1 = 0.0

    scores["paper_recall"] = round(recall, 4)
    scores["paper_precision"] = round(precision, 4)
    scores["paper_f1"] = round(f1, 4)

    created_tex = 0
    exact_tex = 0
    for tex_name, gt_tex in gt_tex_files.items():
        pred_tex_path = pred_dir / tex_name
        if pred_tex_path.exists() and pred_tex_path.is_file():
            created_tex += 1
            try:
                pred_tex = normalize_tex(pred_tex_path.read_text(encoding="utf-8", errors="ignore"))
                if pred_tex == gt_tex["content"]:
                    exact_tex += 1
            except Exception:
                pass

    total_gt_tex = len(gt_tex_files)
    scores["tex_files_created"] = round(created_tex / total_gt_tex, 4) if total_gt_tex else 0.0
    scores["tex_exact_match_ratio"] = round(exact_tex / total_gt_tex, 4) if total_gt_tex else 0.0

    sorted_rows = sorted(
        pred_rows,
        key=lambda x: (normalize_conference(x[0]), normalize_casefold_text(x[1])),
    )
    scores["row_sorting_correct"] = 1.0 if pred_rows == sorted_rows and not duplicate_titles else 0.0

    conference_correct = 0
    authors_correct = 0
    abstract_correct = 0
    author_links_correct = 0
    github_commit_id_correct = 0

    for title_key, gt in gt_by_title.items():
        pred = pred_by_title.get(title_key)
        if pred is None:
            continue

        if normalize_conference(pred["conference"]) == normalize_conference(gt["conference"]):
            conference_correct += 1

        gt_authors = [normalize_author_name(x) for x in gt["authors"]]
        pred_authors = [normalize_author_name(x) for x in pred["authors"]]
        if pred_authors == gt_authors:
            authors_correct += 1

        if normalize_casefold_text(pred["abstract"]) == normalize_casefold_text(gt["abstract"]):
            abstract_correct += 1

        gt_links = gt["author_links"]
        pred_links = pred["author_links"] or []
        if len(pred_links) == len(gt_links):
            for (pred_author, pred_value), (gt_author, gt_value) in zip(pred_links, gt_links):
                if normalize_author_name(pred_author) != normalize_author_name(gt_author):
                    continue
                if normalize_url(pred_value) == normalize_url(gt_value):
                    author_links_correct += 1

        if commit_id_matches(pred["github_commit_id"], gt["github_commit_id"]):
            github_commit_id_correct += 1

    total_gt = len(gt_by_title)
    scores["conference_accuracy"] = round(conference_correct / total_gt, 4) if total_gt else 0.0
    scores["authors_accuracy"] = round(authors_correct / total_gt, 4) if total_gt else 0.0
    scores["abstract_accuracy"] = round(abstract_correct / total_gt, 4) if total_gt else 0.0
    scores["author_links_accuracy"] = (
        round(author_links_correct / total_gt_authors, 4) if total_gt_authors else 0.0
    )
    scores["github_commit_id_accuracy"] = round(github_commit_id_correct / total_gt, 4) if total_gt else 0.0

    hard_checks = [
        "output_exists",
        "tsv_header_valid",
        "rows_parseable",
    ]
    hard_pass = all(scores[k] == 1.0 for k in hard_checks)
    scores["hard_constraint_pass"] = 1.0 if hard_pass else 0.0

    if not hard_pass:
        scores["overall_score"] = 0.0
        return scores

    base_score = round(
        0.05 * scores["paper_f1"]
        + 0.05 * scores["row_sorting_correct"]
        + 0.05 * scores["conference_accuracy"]
        + 0.05 * scores["authors_accuracy"]
        + 0.10 * scores["abstract_accuracy"]
        + 0.15 * scores["author_links_accuracy"]
        + 0.25 * scores["github_commit_id_accuracy"]
        + 0.10 * scores["tex_files_created"]
        + 0.20 * scores["tex_exact_match_ratio"],
        4,
    )
    scores["overall_score"] = round(base_score * scores["output_dir_clean"], 4)
    return scores
