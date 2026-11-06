"""评测集候选题的人工校验（命令行）。逐题展示问题、答案和证据上下文（实时读自本地 text.txt），记录通过 / 修改 / 丢弃。

用法：
    uv run python scripts/review_questions.py --book asoiaf-1 --dataset asoiaf_v0
    uv run python scripts/review_questions.py --book asoiaf-1 --dataset asoiaf_v0 --type multihop
    uv run python scripts/review_questions.py --book asoiaf-1 --dataset asoiaf_v0 --redo r2-plot-003-q1
    uv run python scripts/review_questions.py --book asoiaf-1 --dataset asoiaf_v0 --stats
每个操作立即追加到 data/eval_work/{dataset}/review.jsonl，随时可以 q 退出，下次从未校验的题继续。
"""

from __future__ import annotations

import argparse
import json
import sys

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.eval.qgen.review import (
    DROP_REASONS,
    ReviewLog,
    load_candidates,
    replay,
    respan_evidence,
    review_order,
    review_stats,
)

TYPE_NAMES = {"fact": "事实", "relation": "人物关系", "multihop": "多跳", "plot": "跨章节情节"}
HELP = (
    "[a] 通过  [a1/a2/a3] 通过并标难度  [e] 修改问题/答案  [r] 改证据  [d] 丢弃  [s] 跳过  [u] 撤销上一个操作  [q] 退出"
)


class Painter:
    def __init__(self, color: bool):
        self.color = color

    def hl(self, s: str) -> str:
        return f"\x1b[1;33m{s}\x1b[0m" if self.color else f"【{s}】"

    def dim(self, s: str) -> str:
        return f"\x1b[2m{s}\x1b[0m" if self.color else s

    def bold(self, s: str) -> str:
        return f"\x1b[1m{s}\x1b[0m" if self.color else s


def show(st, book: BookText, pos: int, total: int, paint: Painter, width: int) -> None:  # noqa: ANN001
    c = st.cand
    print("\n" + "=" * 80)
    src = "人工" if c.get("source") == "human" else "生成"
    print(
        f"[{pos}/{total}] {c['cand_id']}  题型：{TYPE_NAMES.get(c['type'], c['type'])}  来源：{src}  状态：{st.status}"
    )
    if c["flags"]:
        print(paint.bold(f"预筛标记：{', '.join(c['flags'])}"))
    print(paint.bold(f"问题：{st.question}"))
    print(paint.bold(f"答案：{st.answer}"))
    if c.get("note"):
        print(paint.dim(f"出题说明：{c['note']}"))
    for k, ev in enumerate(st.evidence, 1):
        ch = book.chapter(ev["chapter_idx"])
        before, mid, after = book.context(ev["char_start"], ev["char_end"], width)
        flag = "" if ev.get("method") == "exact" else f"（{ev.get('method')} {ev.get('score')}）"
        print(f"\n  证据 {k}：第 {ch.chapter_idx} 章「{ch.title}」{('POV ' + ch.pov) if ch.pov else ''}{flag}")
        text = (before + paint.hl(mid) + after).replace("\n", "\n    ")
        print("    " + text)
    print()


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--type", choices=list(TYPE_NAMES), help="只校验某一题型")
    parser.add_argument("--redo", help="重新校验指定的候选题（即使已经校验过）")
    parser.add_argument("--stats", action="store_true", help="只打印校验统计")
    parser.add_argument("--width", type=int, default=150, help="证据前后各展示多少字")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    book = BookText.load(s.paths.books_dir / args.book)
    work = s.paths.eval_work_dir / args.dataset
    log = ReviewLog(work / "review.jsonl")
    cands = load_candidates(work)
    if not cands:
        print("没有可校验的候选题，先运行 scripts/gen_questions.py import")
        return 1
    paint = Painter(color=not args.no_color)

    def stats_and_exit() -> int:
        states = replay(cands, log.read())
        print(json.dumps(review_stats(states), ensure_ascii=False, indent=2))
        return 0

    if args.stats:
        return stats_and_exit()

    order = review_order(cands)
    if args.type:
        type_of = {c["cand_id"]: c["type"] for c in cands}
        order = [cid for cid in order if type_of[cid] == args.type]
    session: list[str] = []  # 本次会话中操作过的题，供撤销
    skipped: set[str] = set()
    forced = [args.redo] if args.redo else []

    while True:
        states = replay(cands, log.read())
        if forced:
            cid = forced[0]
            if cid not in states:
                print(f"找不到候选题 {cid}")
                return 1
        else:
            pending = [c for c in order if states[c].status == "pending" and c not in skipped]
            if not pending:
                print("全部校验完毕。")
                return stats_and_exit()
            cid = pending[0]
        st = states[cid]
        done = sum(1 for c in order if states[c].status != "pending")
        show(st, book, done + 1, len(order), paint, args.width)
        cmd = ask(HELP + "\n> ").lower()

        if cmd == "q":
            return stats_and_exit()
        if cmd == "s":
            skipped.add(cid)
            forced = []
            continue
        if cmd == "u":
            if not session:
                print("本次会话还没有可撤销的操作。")
                continue
            last = session.pop()
            log.append({"cand_id": last, "action": "undo"})
            forced = [last]
            continue
        if cmd in ("a", "a1", "a2", "a3"):
            rec = {"cand_id": cid, "action": "accept"}
            if len(cmd) == 2:
                rec["difficulty"] = int(cmd[1])
            log.append(rec)
        elif cmd == "e":
            q = ask("新问题（回车保持不变）：")
            a = ask("新答案（回车保持不变）：")
            d = ask("难度 1/2/3（回车跳过）：")
            rec = {"cand_id": cid, "action": "edit"}
            if q:
                rec["question"] = q
            if a:
                rec["answer"] = a
            if d in ("1", "2", "3"):
                rec["difficulty"] = int(d)
            log.append(rec)
        elif cmd == "r":
            which = ask(f"改第几段证据（1–{len(st.evidence)}；回车 = 追加一段；-k = 删除第 k 段）：")
            if which.startswith("-") and which[1:].isdigit():
                k = int(which[1:]) - 1
                if 0 <= k < len(st.evidence) and len(st.evidence) > 1:
                    new_ev = [e for i, e in enumerate(st.evidence) if i != k]
                    log.append({"cand_id": cid, "action": "respan", "evidence": new_ev})
                else:
                    print("无效的编号（至少保留一段证据）。")
                forced = [cid]
                session.append(cid)
                continue
            index = int(which) - 1 if which.isdigit() and 1 <= int(which) <= len(st.evidence) else None
            default_ch = st.evidence[index]["chapter_idx"] if index is not None else st.evidence[0]["chapter_idx"]
            quote = ask("原文片段（从上面的上下文中复制）：")
            ch_in = ask(f"所在章节号（回车 = {default_ch}）：")
            ch_idx = int(ch_in) if ch_in.lstrip("-").isdigit() else default_ch
            try:
                new_ev = respan_evidence(
                    book, st.evidence, index, quote, ch_idx, fuzzy_threshold=s.evalgen.fuzzy_threshold
                )
            except KeyError:
                new_ev = None
            if new_ev is None:
                print("没能在该章节中定位到这段原文，证据未修改。")
            else:
                log.append({"cand_id": cid, "action": "respan", "evidence": new_ev})
                session.append(cid)
            forced = [cid]  # 改完证据回到本题再确认
            continue
        elif cmd == "d":
            print("丢弃原因：" + "  ".join(f"[{k}] {v}" for k, v in DROP_REASONS.items()))
            reason = DROP_REASONS.get(ask("> "), "other")
            log.append({"cand_id": cid, "action": "drop", "reason": reason})
        else:
            print("无效的指令。")
            forced = [cid] if forced else []
            continue
        session.append(cid)
        forced = []


if __name__ == "__main__":
    raise SystemExit(main())
