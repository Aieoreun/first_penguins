"""Collect current ungated LLM decisions for PUBLIC-only gate OOF diagnostics.

All bundled requests are sent to /guard without candidate answers. Original IDs
and labels stay in this local recorder and never enter the request payload.
"""
import argparse
import json

from jh_compare import ROOT, load_public, run_variant

BLOCKED = {"harmful_content", "academic_dishonesty", "jailbreak_injection", "off_topic"}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env",override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",required=True)
    parser.add_argument("--variant",choices=("jh_adaptive","jh_r2_joint_ungated","jh_r2_joint_block_ungated",
                                            "jh_r2_dense_ungated"),default="jh_adaptive")
    parser.add_argument("--embedding-mode", choices=("off", "env"), default="off")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("Invalid name")
    root = ROOT / "experiments/.cache" / ("jh_gate_fallback_" + args.name)
    root.mkdir(exist_ok=False)
    items = [{"id":r["id"],"endpoint":"guard","messages":r["messages"],
              "labels":{"category":r["labels"]["category"],
                        "decision":"block" if r["labels"]["category"] in BLOCKED else "allow"}}
             for r in load_public()]
    (root / "derived_inputs.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in items))
    args.workers = args.request_slots = 64
    args.judge_workers = 1
    args.no_judge = True
    args.collect_only = False
    args.endpoints = ["guard"]
    run_variant(args,args.variant,1,items,root)
    for name in ("manifest.json","result.json"):
        path = root / (args.variant + "_1") / name
        data = json.loads(path.read_text())
        data["scope"] = "public_ungated_input_fallback_for_oof_diagnostic"
        data["no_tutor_or_output_score"] = True
        if args.variant == "jh_r2_dense_ungated":
            data["example_scope"] = "full_public_examples_excluding_normalized_identical_last_turn_not_fold_isolated"
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
