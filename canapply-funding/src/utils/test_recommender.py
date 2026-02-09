#!/usr/bin/env python3
"""
Quick manual test harness for the TagRecommender pipeline.
It fires a few suggestion and recommendation queries against the locally running
stack (DB + Qdrant) so you can sanity-check results end-to-end.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from src.recommender.logic import RecommenderLogic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test TagRecommender suggestions & recommendations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["computer vision", "quantum materials", "robotics"],
        help="Free-text queries to run through the suggestion endpoint.",
    )
    parser.add_argument(
        "--tag-groups",
        nargs="+",
        help="Space-separated tag groups to test recommendations with. Example: "
             "--tag-groups \"computer vision\" \"natural language processing\"",
    )
    parser.add_argument(
        "--institute",
        default=None,
        help="Optional institute filter applied when ranking.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional list of domains to constrain ranking.",
    )
    parser.add_argument(
        "--subfields",
        nargs="+",
        default=None,
        help="Optional list of subfields to constrain ranking.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K suggestions per query.",
    )
    parser.add_argument(
        "--show-explanations",
        action="store_true",
        help="Pretty-print explanations returned by rank_all.",
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def main() -> int:
    args = parse_args()

    print_section("Bootstrapping TagRecommender")
    recommender = RecommenderLogic(lazy_init=False)
    print(f"Loaded catalog with {recommender.n_docs} unique tags.")

    print_section("Suggestion Checks")
    for query in args.queries:
        suggestions = recommender.suggest(query, top_k=args.topk)
        print(f"\nQuery: {query!r}")
        if not suggestions:
            print("  (no suggestions)")
            continue
        for idx, item in enumerate(suggestions, start=1):
            print(
                f"  {idx:02d}. {item['tag']}  "
                f"[domain={item['domain']} | subfield={item['subfield']} | score={item['score']}] "
                f"(prof_count={item['prof_count']})"
            )

    tag_groups: List[List[str]]
    if args.tag_groups:
        tag_groups = [tg.split() for tg in args.tag_groups]
    else:
        tag_groups = [
            ["computer vision"],
            ["natural language processing"],
            ["renewable energy", "power systems"],
        ]

    print_section("Recommendation Checks")
    for tags in tag_groups:
        print(f"\nTags: {tags}")
        result = recommender.rank_all(
            tags=tags,
            institute_name=args.institute,
            expand_related=True,
            domain_filters=args.domains,
            subfield_filters=args.subfields,
        )
        profs = result.get("professor_ids", [])[: args.topk]
        print(f"  Found {len(result.get('professor_ids', []))} professors. First {len(profs)}:")
        for idx, pid in enumerate(profs, start=1):
            name = recommender.prof_to_name.get(pid, "<unknown>")
            inst = recommender.prof_to_inst_norm.get(pid, "<inst?>")
            print(f"    {idx:02d}. {pid} :: {name} @ {inst}")

        if args.show_explanations:
            print("  Explanations:")
            print(json.dumps(result.get("explanations", []), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
