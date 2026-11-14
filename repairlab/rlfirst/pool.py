"""The generated-bug pool, its development sets, and their admission.

Training pool: generated bugs in the 108 opened programs (pilot, supervised training,
first final test, earlier RL), at most 30 per program and no kind above 15% of the pool,
plus the 108 original hand-written bugs. Development, seen programs: one generated bug
per training program, held out with every mutation at the same site. Development, unseen
programs: one generated bug and the original bug of each second-final-test program.
"""

import ast
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from ..curriculum.final_test import final_tasks
from ..curriculum.splits import normalized_tokens
from ..curriculum.training_programs import training_programs
from ..final2.admission import reject_final2
from ..final2.programs import final2_tasks
from ..rl.programs import rl_programs
from ..tasks import Case, canonical, pilot_tasks, sha256
from .mutate import HAND_WRITTEN, KINDS, TWO_BUGS, candidates, combine

SEED = 20261109
PER_PROGRAM = 30
KIND_SHARE = 0.15
MAX_GENERATED = 2000
TWO_BUG_DRAWS = 40


def training_sources():
    """The 108 opened programs, with the split each came from."""
    rows = [("pilot", t) for t in pilot_tasks()]
    rows += [("training", p.task) for p in training_programs()]
    rows += [("final1", t) for t in final_tasks()]
    rows += [("rl", t) for t in rl_programs()]
    return rows


def ordered_cases(task):
    """The fixed order for witnesses: authored public cases, then the private domain."""
    return list(task.public_cases) + list(task.private_cases)


def _key(text):
    return tuple(normalized_tokens(text))


def evaluate(executor, task, text):
    """Bug behavior over the whole domain, or None if it does not finish within limits."""
    cases = ordered_cases(task)
    result = executor.run(text, cases, reveal=True)
    if result["status"] != "graded":
        return None
    return [row["passed"] for row in result["cases"]]


def admitted_singles(executor, task, workers=8):
    """Single mutations that keep the signature, change some answer, and always finish."""
    found = candidates(task.reference, task.signature)
    texts = list(found)
    with ThreadPoolExecutor(workers) as pool:
        behaviors = list(pool.map(lambda text: evaluate(executor, task, text), texts))
    return [
        {"text": text, "mutation": found[text], "passed": behavior}
        for text, behavior in zip(texts, behaviors)
        if behavior is not None and not all(behavior)
    ]


def two_bug_candidates(executor, task, singles, rng, draws=TWO_BUG_DRAWS):
    pairs = [
        (a, b)
        for i, a in enumerate(singles)
        for b in singles[i + 1 :]
        if a["mutation"].kind != b["mutation"].kind
    ]
    rng.shuffle(pairs)
    made, seen = [], set()
    for a, b in pairs:
        if len(made) >= draws:
            break
        combined = combine(task.reference, task.signature, a["mutation"], b["mutation"])
        if combined is None or combined[1] in seen:
            continue
        mutation, text = combined
        seen.add(text)
        behavior = evaluate(executor, task, text)
        if behavior is None or all(behavior):
            continue
        made.append({"text": text, "mutation": mutation, "passed": behavior})
    return made


def witness_public(task, passed):
    """Public cases for a bug: authored ones, plus the first failing domain input if none fails."""
    public = list(task.public_cases)
    order = ordered_cases(task)
    first = passed.index(False)
    if first >= len(public):
        public.append(order[first])
    keys = {canonical(c.args) for c in public}
    hidden = [c for c in order if canonical(c.args) not in keys]
    return tuple(public), tuple(hidden)


def make_task(task, ident, kind, text, passed, source_split):
    public, hidden = witness_public(task, passed)
    return replace(
        task,
        id=ident,
        bug_category=kind,
        buggy=text,
        public_cases=public,
        private_cases=hidden,
    ), {"program": task.id, "source_split": source_split, "kind": kind}


def _ident(program, text):
    return f"{program}~{sha256(text)[:10]}"


def build(executor, log=print):
    """Select the pool and development sets. Deterministic for a given executor's verdicts."""
    rng = random.Random(SEED)
    reference_keys = {_key(t.reference) for _, t in training_sources()} | {
        _key(t.reference) for t in final2_tasks()
    }
    original_keys = {_key(t.buggy) for _, t in training_sources()} | {
        _key(t.buggy) for t in final2_tasks()
    }
    seen_keys = set(reference_keys | original_keys)

    def distinct(rows):
        kept = []
        for row in rows:
            key = _key(row["text"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            kept.append(row)
        return kept

    per_program, holdouts, stats = {}, [], Counter()
    for source_split, task in training_sources():
        singles = distinct(admitted_singles(executor, task))
        stats["admitted_singles"] += len(singles)
        if not singles:
            per_program[task.id] = (source_split, task, [])
            log(f"{task.id}: no admitted mutation")
            continue
        held = rng.choice(singles)
        holdouts.append((source_split, task, held))
        site = held["mutation"].edits
        rest = [
            s
            for s in singles
            if s is not held
            and not any(
                a.start < b.end and b.start < a.end for a in s["mutation"].edits for b in site
            )
        ]
        pairs = distinct(two_bug_candidates(executor, task, rest, rng))
        stats["admitted_two_bugs"] += len(pairs)
        per_program[task.id] = (source_split, task, rest + pairs)
        log(f"{task.id}: {len(singles)} singles, {len(pairs)} two-bug")

    # Round-robin over kinds within each program, preferring new failure patterns.
    chosen = {}
    for program, (source_split, task, rows) in per_program.items():
        by_kind = defaultdict(list)
        for row in rows:
            by_kind[row["mutation"].kind].append(row)
        for kind_rows in by_kind.values():
            rng.shuffle(kind_rows)
        kinds = sorted(by_kind)
        rng.shuffle(kinds)
        picked, patterns = [], set()
        while len(picked) < PER_PROGRAM and any(by_kind.values()):
            for kind in kinds:
                if len(picked) >= PER_PROGRAM or not by_kind[kind]:
                    continue
                options = by_kind[kind]
                fresh = [r for r in options if tuple(r["passed"]) not in patterns]
                row = (fresh or options)[0]
                options.remove(row)
                picked.append(row)
                patterns.add(tuple(row["passed"]))
        chosen[program] = picked

    # Cap every kind at 15% of the pool, trimming from the programs with the most of it.
    while True:
        total = sum(len(v) for v in chosen.values()) + len(per_program)
        cap = int(KIND_SHARE * total)
        counts = Counter(r["mutation"].kind for v in chosen.values() for r in v)
        over = [k for k, n in counts.items() if n > cap]
        generated = sum(counts.values())
        if not over and generated <= MAX_GENERATED:
            break
        kind = over[0] if over else counts.most_common(1)[0][0]
        program = max(chosen, key=lambda p: (sum(r["mutation"].kind == kind for r in chosen[p]), p))
        victim = [r for r in chosen[program] if r["mutation"].kind == kind][-1]
        chosen[program].remove(victim)

    pool, meta = [], {}
    for program, (source_split, task, _) in per_program.items():
        original_passed = evaluate(executor, task, task.buggy)
        if original_passed is None or all(original_passed):
            raise ValueError("Original bug no longer fails: " + task.id)
        built, info = make_task(
            task,
            _ident(task.id, task.buggy),
            HAND_WRITTEN,
            task.buggy,
            original_passed,
            source_split,
        )
        pool.append(built)
        meta[built.id] = {**info, "description": "original hand-written bug"}
        for row in chosen[program]:
            built, info = make_task(
                task,
                _ident(task.id, row["text"]),
                row["mutation"].kind,
                row["text"],
                row["passed"],
                source_split,
            )
            pool.append(built)
            meta[built.id] = {
                **info,
                "description": row["mutation"].description,
                "edits": [[e.start, e.end, e.text] for e in row["mutation"].edits],
            }

    dev_seen = []
    for source_split, task, row in holdouts:
        built, info = make_task(
            task,
            _ident(task.id, row["text"]),
            row["mutation"].kind,
            row["text"],
            row["passed"],
            source_split,
        )
        dev_seen.append(built)
        meta[built.id] = {**info, "description": row["mutation"].description, "split": "dev_seen"}

    dev_unseen = []
    for task in final2_tasks():
        singles = distinct(admitted_singles(executor, task))
        if singles:
            row = rng.choice(singles)
            built, info = make_task(
                task,
                _ident(task.id, row["text"]),
                row["mutation"].kind,
                row["text"],
                row["passed"],
                "final2",
            )
            dev_unseen.append(built)
            meta[built.id] = {
                **info,
                "description": row["mutation"].description,
                "split": "dev_unseen",
            }
        original = evaluate(executor, task, task.buggy)
        built, info = make_task(
            task, _ident(task.id, task.buggy), HAND_WRITTEN, task.buggy, original, "final2"
        )
        dev_unseen.append(built)
        meta[built.id] = {**info, "description": "original hand-written bug", "split": "dev_unseen"}

    check_separation(pool, dev_seen, dev_unseen)
    stats.update(
        pool=len(pool),
        dev_seen=len(dev_seen),
        dev_unseen=len(dev_unseen),
    )
    return {
        "pool": pool,
        "dev_seen": dev_seen,
        "dev_unseen": dev_unseen,
        "meta": meta,
        "stats": stats,
    }


def reject_evaluation_material(tasks):
    """Refuse second- or third-test content in any training or model-facing path.

    Returns whether the third sealed test existed to check against; it is written
    separately, and every later launcher checks again.
    """
    reject_final2(tasks)
    try:
        from ..final3.admission import reject_final3
    except ImportError:
        return False
    reject_final3(tasks)
    return True


def check_separation(pool, dev_seen, dev_unseen):
    """Training material never contains a development or sealed bug, by content."""
    reject_evaluation_material(pool + dev_seen)
    pool_bugs = {t.buggy for t in pool}
    if pool_bugs & {t.buggy for t in dev_seen + dev_unseen}:
        raise ValueError("A development bug is in the training pool")
    unseen_programs = {t.reference for t in final2_tasks()}
    if any(t.reference in unseen_programs for t in pool + dev_seen):
        raise ValueError("A second-final-test program is in training material")
    for t in pool + dev_seen + dev_unseen:
        ast.parse(t.buggy)


def kind_counts(tasks):
    return dict(sorted(Counter(t.bug_category for t in tasks).items()))


def summary(selected):
    pool = selected["pool"]
    return {
        "seed": SEED,
        "per_program_cap": PER_PROGRAM,
        "kind_share_cap": KIND_SHARE,
        "kinds": list(KINDS) + [HAND_WRITTEN, TWO_BUGS],
        "pool_tasks": len(pool),
        "pool_by_kind": kind_counts(pool),
        "pool_by_split": dict(Counter(selected["meta"][t.id]["source_split"] for t in pool)),
        "dev_seen_tasks": len(selected["dev_seen"]),
        "dev_seen_by_kind": kind_counts(selected["dev_seen"]),
        "dev_unseen_tasks": len(selected["dev_unseen"]),
        "dev_unseen_by_kind": kind_counts(selected["dev_unseen"]),
        "programs_without_mutation": sorted(
            p
            for p in {selected["meta"][t.id]["program"] for t in pool}
            if not any(
                selected["meta"][t.id]["program"] == p and t.bug_category != HAND_WRITTEN
                for t in pool
            )
        ),
        "stats": dict(selected["stats"]),
    }


def record(task):
    return {
        "task_id": task.id,
        "family": task.family,
        "kind": task.bug_category,
        "specification": task.specification,
        "signature": task.signature,
        "buggy": task.buggy,
        "reference": task.reference,
        "public": [c.record() for c in task.public_cases],
        "hidden": [c.record() for c in task.private_cases],
        "task_sha256": task.fingerprint(),
    }


def from_record(row, template):
    return replace(
        template,
        id=row["task_id"],
        bug_category=row["kind"],
        buggy=row["buggy"],
        public_cases=tuple(Case(c["args"], c["expected"]) for c in row["public"]),
        private_cases=tuple(Case(c["args"], c["expected"]) for c in row["hidden"]),
    )
