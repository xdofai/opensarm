from __future__ import annotations

import argparse
import hashlib
import os

DEFAULT_ROOT = ""


def generate_seed_from_usr(usr: str) -> int:
    hash_obj = hashlib.sha256(usr.encode("utf-8"))
    seed = int.from_bytes(hash_obj.digest()[:8], byteorder="big")
    return seed % (2**31)


def main():
    parser = argparse.ArgumentParser(description="Run rollout episode labeling web server")
    parser.add_argument(
        "--task_name",
        required=True,
        help="Task name used as the data/<task_name> subdir (shared across runs for resume support)",
    )
    parser.add_argument("--usr", help="Annotator/User id (optional in headless mode)")
    parser.add_argument(
        "--mode", choices=["new", "resume"], default="resume",
        help="new: create plan; resume: resume existing plan"
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Dataset root containing episode_* dirs")
    parser.add_argument("--host", default="0.0.0.0", help="Host for web server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8888, help="Port for web server (default: 8888)")
    args = parser.parse_args()

    os.environ["ROLLOUT_LABEL_TASK_NAME"] = args.task_name

    from src.io_utils import CKPT_DIR, PLANS_DIR, RESULTS_DIR, read_json
    from src.plan import make_plan, save_plan
    from src.web_server import LabelWebServer

    def _reset_progress_if_same_usr(usr: str, plan_id: str):
        results_path = RESULTS_DIR / f"{usr}_{plan_id}.jsonl"
        ckpt_path = CKPT_DIR / f"{usr}_{plan_id}.json"

        removed_any = False
        if results_path.exists():
            results_path.unlink()
            print(f"[fresh] Removed old results: {results_path}")
            removed_any = True
        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"[fresh] Removed old checkpoint: {ckpt_path}")
            removed_any = True
        if not removed_any:
            print("[fresh] No previous results/checkpoint to remove for this plan.")

    if not args.usr:
        app = LabelWebServer(plan_dict={}, usr="default", host=args.host, port=args.port, root=args.root)
        app.run()
        return

    usr = args.usr
    seed = generate_seed_from_usr(usr)

    if args.mode == "new":
        existing_results = sorted(RESULTS_DIR.glob(f"{usr}_*.jsonl"))
        has_existing = any(f.stat().st_size > 0 for f in existing_results)
        if has_existing:
            count = sum(1 for f in existing_results for _ in f.open())
            print(f"\n[WARNING] User '{usr}' already has {count} labeled episode(s).")
            print("Starting over will erase all previous progress.")
            resp = input("Press Enter to confirm, or Ctrl+C to cancel: ")

        print(f"[seed] Generated seed {seed} from user '{usr}'")
        plan = make_plan(args.root, seed=seed)
        plan_path = save_plan(plan, usr)
        print(f"[plan] Saved plan to: {plan_path}")
        _reset_progress_if_same_usr(usr, plan.plan_id)
        plan_dict = read_json(plan_path)
    else:
        cand_plans = sorted(PLANS_DIR.glob(f"{usr}_*.json"))
        if not cand_plans:
            raise SystemExit(f"No existing plan for user '{usr}'. Run with --mode new first.")
        plan_path = cand_plans[-1]
        print(f"[resume] Using plan: {plan_path}")
        plan_dict = read_json(plan_path)

    app = LabelWebServer(plan_dict, usr=usr, host=args.host, port=args.port, root=args.root)
    app.run()


if __name__ == "__main__":
    main()
