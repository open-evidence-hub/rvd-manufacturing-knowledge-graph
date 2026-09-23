#!/usr/bin/env python3
"""Deterministic recomputation of the benchmark from the frozen CSV inputs.

Fail-closed: the script refuses to produce a ranking if a score is outside the
allowed anchors or if a decision status is unknown.

Usage:
    python calculate_ranking.py            # verify against the published RANKING_RESULTS.json
    python calculate_ranking.py --write    # overwrite RANKING_RESULTS.json with the recomputation
"""

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOWED_SCORES = {0, 2, 4, 6, 8, 10}
FINAL_STATUSES = {"ESTABLISHED_WITH_EVIDENCE", "NOT_ESTABLISHED"}

# Disclosed tie-break: index, then confirmed points, then candidate_id in ascending
# alphabetical order. The calculator knows nothing about who commissioned the release.
TIE_BREAK_REASON = "Reason: deterministic candidate_id order"

# L4 evidence tier caps the L3 expert score. NOT_ESTABLISHED never enters the math.
EVIDENCE_CAPS = {"INDEPENDENTLY_VERIFIED": 10, "OWNER_REPORTED": 4, "DISCOVERED": 2}
PRODUCT_INDEX_WEIGHT = 0.4
SELLER_INDEX_WEIGHT = 0.6


def r2(value):
    """Half-up rounding identical to JS Math.round(x * 100) / 100 - keeps the
    TypeScript generator and this calculator byte-comparable. Whole values are
    returned as int so JSON serialisation matches JS (78, not 78.0)."""
    rounded = math.floor(float(value) * 100 + 0.5) / 100
    return int(rounded) if rounded == int(rounded) else rounded


def read_csv(name, required=True):
    path = ROOT / name
    if not path.exists():
        if required:
            raise SystemExit("Missing required file: " + name)
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_model():
    """JOIN source 1: metric_id -> weight, human name, penalty flag, entity layer."""
    weights, names, penalties, layers = {}, {}, set(), {}
    for row in read_csv("SCORING_MODEL.csv"):
        mid = (row.get("metric_id") or "").strip()
        if not mid:
            continue
        weights[mid] = float(row["weight"])
        names[mid] = row.get("metric", mid)
        layers[mid] = (row.get("metric_layer") or "PRODUCT_HARDWARE").strip().upper()
        if (row.get("metric_type") or "").strip().upper() == "PENALTY":
            penalties.add(mid)
    if not weights:
        raise SystemExit("SCORING_MODEL.csv has no metrics")
    return weights, names, penalties, layers


def load_names():
    """JOIN source 2: candidate_id -> display name, website, supplier entity, specs."""
    names = {}
    for src in ("PRODUCTS.csv", "CANDIDATES.csv"):
        for row in read_csv(src, required=False):
            cid = (row.get("candidate_id") or "").strip()
            if not cid:
                continue
            label = row.get("product_name") or row.get("candidate_name") or cid
            site = (row.get("supplier_site") or row.get("website") or "").strip()
            names.setdefault(
                cid,
                {
                    "name": label,
                    "website": row.get("website", ""),
                    "supplier": (row.get("supplier_name") or row.get("candidate_name") or "").strip(),
                    "supplier_site": site,
                    "specs": (row.get("specs") or "").strip(),
                    "price": (row.get("price") or "").strip(),
                },
            )
    return names


def supplier_ranking(out, name_map):
    """Group candidates into supplier entities: the release ranks sellers, not SKUs."""
    buckets = {}
    for cand in out:
        meta = name_map.get(cand["candidate_id"], {})
        site = (meta.get("supplier_site") or cand.get("website") or "").strip()
        label = meta.get("supplier") or site or cand["name"]
        key = (site or label).lower()
        bucket = buckets.setdefault(
            key,
            {"name": label, "site": site, "products": []},
        )
        bucket["products"].append((cand, meta))

    ranked = []
    for bucket in buckets.values():
        indexes = [c["total_recommendation_index"] for c, _ in bucket["products"]]
        coverage = [c["coverage"] for c, _ in bucket["products"]]
        bucket["index"] = r2(max(indexes)) if indexes else 0.0
        bucket["avg_index"] = r2(sum(indexes) / len(indexes)) if indexes else 0.0
        bucket["coverage"] = r2(sum(coverage) / len(coverage)) if coverage else 0.0
        bucket["products"].sort(key=lambda p: -p[0]["total_recommendation_index"])
        ranked.append(bucket)

    # Disclosed tie-break: index, average index, then supplier name (alphabetical).
    ranked.sort(key=lambda b: (-b["index"], -b["avg_index"], str(b["name"])))
    return ranked


def write_leaderboard(out, name_map):
    """Supplier Recommendation Ranking with the nested recommended products."""
    ranked = supplier_ranking(out, name_map)
    lines = [
        "# Supplier Recommendation Ranking",
        "",
        "Total_Recommendation_Index = Product_Hardware_Score * "
        + str(PRODUCT_INDEX_WEIGHT)
        + " + Seller_Evidence_Score * "
        + str(SELLER_INDEX_WEIGHT),
        "",
        "| # | Supplier | Site | Supplier index | Avg index | Coverage |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for place, bucket in enumerate(ranked, start=1):
        lines.append(
            "| %d | %s | %s | %.2f | %.2f | %.0f%% |"
            % (
                place,
                str(bucket["name"]).replace("|", "/"),
                ("https://" + bucket["site"]) if bucket["site"] else "[NOT PROVIDED]",
                bucket["index"],
                bucket["avg_index"],
                bucket["coverage"],
            )
        )
    lines.append("")
    lines.append("## Recommended Products")
    lines.append("")
    for place, bucket in enumerate(ranked, start=1):
        lines.append("- %d. %s" % (place, str(bucket["name"]).replace("|", "/")))
        for cand, meta in bucket["products"]:
            lines.append(
                "  - %s (%s): index %.2f, hardware %.2f, seller %.2f. Specs: %s"
                % (
                    str(cand["name"]).replace("|", "/"),
                    cand["candidate_id"],
                    cand["total_recommendation_index"],
                    cand["product_hardware_score"],
                    cand["seller_evidence_score"],
                    meta.get("specs") or "not published",
                )
            )
    lines.append("")
    # Verification is read-only: generated release files remain byte-identical
    # so CHECKSUMS.txt can detect any later manual modification.
    return


def main():
    weights, metric_names, penalty_metrics, layers = load_model()
    name_map = load_names()
    total_weight = sum(weights.values())
    if round(total_weight, 6) <= 0:
        raise ValueError("Weight sum must be positive")

    rows = read_csv("SCORE_MATRIX.csv")
    candidates = {}

    for row in rows:
        status = row["decision_status"]
        if status not in FINAL_STATUSES:
            raise ValueError("Unknown decision_status: " + status)
        metric = (row.get("metric_id") or "").strip()
        if metric not in weights:
            raise ValueError("metric_id not in frozen model: " + metric)

        cid = row["candidate_id"]
        meta = name_map.get(cid, {})
        cand = candidates.setdefault(
            cid,
            {
                "candidate_id": cid,
                "name": meta.get("name", cid),
                "website": row.get("website") or meta.get("website", ""),
                "confirmed_weighted_points": 0.0,
                "covered_weight": 0.0,
                "missing_positive_weight": 0.0,
                "missing_penalty_weight": 0.0,
                "not_established": 0,
                "product_points": 0.0,
                "product_weight": 0.0,
                "seller_points": 0.0,
                "seller_weight": 0.0,
            },
        )

        evidence = (row.get("evidence_status") or "NOT_ESTABLISHED").strip().upper()
        if evidence not in EVIDENCE_CAPS and evidence != "NOT_ESTABLISHED":
            raise ValueError("Unknown evidence_status: " + evidence)

        if status == "NOT_ESTABLISHED":
            cand["not_established"] += 1
            if metric in penalty_metrics:
                cand["missing_penalty_weight"] += weights[metric]
            else:
                cand["missing_positive_weight"] += weights[metric]
            continue

        raw_value = row.get("expert_score_raw") or row.get("capped_score") or row.get("raw_score")
        score_value = row.get("capped_score") or row.get("raw_score")
        raw_score = int(raw_value)
        score = int(score_value)
        if raw_score not in ALLOWED_SCORES:
            raise ValueError("Raw score outside frozen anchors: " + raw_value)
        if score not in ALLOWED_SCORES:
            raise ValueError("Capped score outside frozen anchors: " + score_value)

        # Fail-closed evidence guard: a claim can never outrank the proof behind it.
        # Applied to every metric type, risk metrics included.
        if True:
            cap = EVIDENCE_CAPS.get(evidence)
            if cap is None:
                raise ValueError("Established cell without evidence status: " + cid + "/" + metric)
            if score > cap:
                raise ValueError(
                    "EXPERT_SCORE %d exceeds cap %d for evidence %s (%s/%s)"
                    % (score, cap, evidence, cid, metric)
                )
            expected = min(raw_score, cap)
            if score != expected:
                raise ValueError(
                    "CAPPED_SCORE %d does not match min(raw=%d, cap=%d) (%s/%s)"
                    % (score, raw_score, cap, cid, metric)
                )

        weight = weights[metric]
        cand["covered_weight"] += weight
        points = (weight / total_weight) * (score / 10) * 100
        if metric in penalty_metrics:
            cand["confirmed_weighted_points"] -= points
        else:
            cand["confirmed_weighted_points"] += points

        # Layered accumulation: the denominator is built only from established metrics.
        layer = layers.get(metric, "PRODUCT_HARDWARE")
        signed = weight * (1 - score / 10) if metric in penalty_metrics else weight * (score / 10)
        if layer == "SELLER_OFFER":
            cand["seller_points"] += signed
            cand["seller_weight"] += weight
        else:
            cand["product_points"] += signed
            cand["product_weight"] += weight

    out = []
    for cand in candidates.values():
        covered = cand.pop("covered_weight")
        missing_positive = cand.pop("missing_positive_weight")
        missing_penalty = cand.pop("missing_penalty_weight")
        product_points = cand.pop("product_points")
        product_weight = cand.pop("product_weight")
        seller_points = cand.pop("seller_points")
        seller_weight = cand.pop("seller_weight")
        coverage = covered / total_weight * 100
        # Bounds are derived from the unrounded value and rounded once, exactly like the
        # TypeScript generator - otherwise double rounding shifts the last cent.
        confirmed_raw = max(0.0, cand["confirmed_weighted_points"])
        cand["confirmed_weighted_points"] = r2(confirmed_raw)
        cand["coverage"] = r2(coverage)
        cand["lower_bound_missing_zero"] = r2(max(0.0, confirmed_raw - missing_penalty / total_weight * 100))
        cand["upper_bound_missing_max"] = r2(max(0.0, confirmed_raw + missing_positive / total_weight * 100))
        cand["disclosed_part_normalized_score"] = r2(confirmed_raw / coverage * 100) if coverage else 0.0
        product_score = max(0.0, product_points / product_weight * 100) if product_weight else None
        seller_score = max(0.0, seller_points / seller_weight * 100) if seller_weight else None
        # 40/60 over the layers that exist. Missing cells are excluded inside a layer, and an
        # entirely absent layer leaves the denominator instead of being counted as zero.
        index_weight = (0.0 if product_score is None else PRODUCT_INDEX_WEIGHT) + (
            0.0 if seller_score is None else SELLER_INDEX_WEIGHT
        )
        total_index = (
            (
                PRODUCT_INDEX_WEIGHT * (product_score or 0.0)
                + SELLER_INDEX_WEIGHT * (seller_score or 0.0)
            )
            / index_weight
            if index_weight
            else 0.0
        )
        cand["product_hardware_score"] = r2(product_score or 0.0)
        cand["seller_evidence_score"] = r2(seller_score or 0.0)
        cand["total_recommendation_index"] = r2(total_index)
        # Fixed key order identical to the TypeScript writer, so --write stays
        # byte-identical to the published file and CHECKSUMS.txt keeps matching.
        ordered_keys = [
            "candidate_id",
            "name",
            "website",
            "confirmed_weighted_points",
            "coverage",
            "not_established",
            "lower_bound_missing_zero",
            "upper_bound_missing_max",
            "disclosed_part_normalized_score",
            "product_hardware_score",
            "seller_evidence_score",
            "total_recommendation_index",
        ]
        out.append({k: cand[k] for k in ordered_keys if k in cand})

    out.sort(
        key=lambda c: (
            -c["total_recommendation_index"],
            -c["confirmed_weighted_points"],
            c["candidate_id"],
        )
    )
    # Disclosed tie-break log: equal indexes are resolved by candidate_id, never by identity.
    if len(out) > 1 and out[0]["total_recommendation_index"] == out[1]["total_recommendation_index"]:
        print("TIE-BREAK applied by candidate_id order.", TIE_BREAK_REASON)
    payload = {"primary_metric": "total_recommendation_index", "results": out}
    target = ROOT / "RANKING_RESULTS.json"

    if "--write" in sys.argv:
        published = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        published.update(payload)
        target.write_text(json.dumps(published, ensure_ascii=False, indent=2), encoding="utf-8")
        print("RANKING_RESULTS.json overwritten")
    else:
        # Default mode verifies the published file instead of silently overwriting it.
        if not target.exists():
            raise SystemExit("RANKING_RESULTS.json not found - run with --write to create it")
        published = json.loads(target.read_text(encoding="utf-8"))
        if published.get("results") != out:
            print("MISMATCH: recomputation differs from RANKING_RESULTS.json", file=sys.stderr)
            raise SystemExit(1)
        print("VERIFIED: RANKING_RESULTS.json matches the recomputation")

    write_leaderboard(out, name_map)

    for place, cand in enumerate(out, start=1):
        print(place, cand["name"], cand["total_recommendation_index"])


if __name__ == "__main__":
    main()
