"""
end_to_end.py -- formalize the END-TO-END field detection rate from existing artifacts.
======================================================================================
Connects the two deployed passes already on disk into one end-to-end measurement, per
live module and aggregate. Reads ONLY existing artifacts (no new field run):

  field_modules/<tag>.json              -- self-contained: ground_truth_keys + predicted_key_blocks
  field_modules/<tag>.predictions.json  -- classifier-path funnel counters (candidates etc.), optional

Definitions (documented so the paper can state them):
  * A ground-truth key is DETECTED if some predicted block's value_hex equals it, or one
    contains the other (DER blocks embed the key; handles PKCS#8 vs inner-key framing).
  * A detection is STRUCTURE-VERIFIED if the matching block's type begins "DER" or names a
    Curve25519/Ed25519 scalar; otherwise it is a STATISTICAL flag (AES/raw length+entropy).
  * Predicted blocks (deduped by value_hex) that match NO ground-truth key are FALSE
    POSITIVES = analyst review actions.
  * Recall = detected / GT.  Actionable precision = detected_keys / (detected_keys + FP_blocks).
  * Key types: asymmetric (RSA/EC/ECDSA/Ed/X25519) vs symmetric (AES/ChaCha/HMAC).

USAGE
-----
    python end_to_end.py --indir field_modules --tex end_to_end.tex
Paste the printed JSON / \input the .tex; numbers go into the paper only after this real run.
"""
import argparse, glob, json, math, os, re


def norm(h):
    return re.sub(r"[^0-9a-f]", "", (h or "").lower().replace("0x", ""))


def is_asym(algo):
    a = (algo or "").upper()
    return any(t in a for t in ("RSA", "EC", "ECDSA", "ED25519", "X25519", "CURVE", "DSA"))


def struct_verified(block_type):
    t = (block_type or "").upper()
    return t.startswith("DER") or "25519" in t or "CURVE" in t or "ED25519" in t


def match(gt_hex, pred_hex):
    return gt_hex == pred_hex or (gt_hex and pred_hex and (gt_hex in pred_hex or pred_hex in gt_hex))


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (round(100*(c-h)/d, 1), round(100*(c+h)/d, 1))


def load_funnel(indir, tag):
    f = os.path.join(indir, f"{tag}.predictions.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f, encoding="utf-8"))
    fn = d.get("funnel") or {}
    return fn.get("std_len_candidates") or fn.get("candidates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="field_modules")
    ap.add_argument("--tex", default="end_to_end.tex")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "*.json")))
    files = [f for f in files if not f.endswith(".groundtruth.json")
             and not f.endswith(".predictions.json")]
    rows, T = [], dict(gt=0, det=0, ver=0, stat_det=0, struct_extra=0, stat_fp=0,
                       asym=0, asym_det=0, sym=0, sym_det=0, cand=0)

    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        gt = d.get("ground_truth_keys") or []
        preds = d.get("predicted_key_blocks") or []
        if not gt:
            continue   # keep modules with GT even if 0 detections (they belong in the denominator)
        tag = os.path.splitext(os.path.basename(f))[0]
        # dedupe predicted blocks by value
        pblocks = {}
        for b in preds:
            pblocks.setdefault(norm(b.get("value_hex")), b.get("type"))
        det = ver = asym = asym_det = sym = sym_det = 0
        matched_pred = set()
        for g in gt:
            gh = norm(g.get("value_hex"))
            asy = is_asym(g.get("algo"))
            asym += asy; sym += (not asy)
            hit = None
            for ph in pblocks:
                if match(gh, ph):
                    hit = ph; break
            if hit is not None:
                det += 1; matched_pred.add(hit)
                asym_det += asy; sym_det += (not asy)
                if struct_verified(pblocks[hit]):
                    ver += 1
        stat_det = det - ver
        # Path-separated FP. Structure-verified blocks (DER/Curve25519) that match no GT key
        # are THEMSELVES valid keys (extra, not non-key FP) -> struct_extra. Statistical
        # (AES/raw) blocks that match no GT key are potential non-key FP -> stat_fp.
        struct_extra = stat_fp = 0
        for ph, ptype in pblocks.items():
            if not ph or ph in matched_pred or any(match(norm(g.get("value_hex")), ph) for g in gt):
                continue
            if struct_verified(ptype):
                struct_extra += 1
            else:
                stat_fp += 1
        n = len(gt)
        stat_prec = round(100*stat_det/(stat_det+stat_fp), 1) if (stat_det+stat_fp) else None
        cand = load_funnel(a.indir, tag)
        rows.append({"module": d.get("module", tag), "gt": n, "detected": det,
                     "struct_det": ver, "stat_det": stat_det, "struct_extra": struct_extra,
                     "stat_fp": stat_fp, "recall_pct": round(100*det/n, 1),
                     "stat_prec_pct": stat_prec, "candidates": cand,
                     "asym": f"{asym_det}/{asym}", "sym": f"{sym_det}/{sym}"})
        for k, v in [("gt", n), ("det", det), ("ver", ver), ("stat_det", stat_det),
                     ("struct_extra", struct_extra), ("stat_fp", stat_fp),
                     ("asym", asym), ("asym_det", asym_det), ("sym", sym), ("sym_det", sym_det),
                     ("cand", cand or 0)]:
            T[k] += v

    lo, hi = wilson(T["det"], T["gt"])
    summary = {"rows": rows, "totals": T,
               "recall_pct": round(100*T["det"]/max(T["gt"], 1), 1), "recall_CI95": [lo, hi],
               "struct_path_keys": T["ver"], "stat_path_keys": T["stat_det"],
               "stat_path_fp": T["stat_fp"],
               "stat_precision_pct": round(100*T["stat_det"]/max(T["stat_det"]+T["stat_fp"], 1), 1),
               "asym_recall_pct": round(100*T["asym_det"]/max(T["asym"], 1), 1),
               "sym_recall_pct": round(100*T["sym_det"]/max(T["sym"], 1), 1)}
    print(json.dumps(summary, indent=2))

    body = "\n".join(
        f"{r['module']} & {r['gt']} & {r['struct_det']} & {r['stat_det']} & {r['stat_fp']} & "
        f"{r['recall_pct']}\\% \\\\" for r in rows)
    tot = (f"\\textbf{{Total}} & \\textbf{{{T['gt']}}} & \\textbf{{{T['ver']}}} & "
           f"\\textbf{{{T['stat_det']}}} & \\textbf{{{T['stat_fp']}}} & "
           f"\\textbf{{{round(100*T['det']/max(T['gt'],1),1)}\\%}} \\\\")
    tex = ("% Auto-generated by end_to_end.py -- VERIFY then \\input.\n"
           "\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}\n"
           f"\\caption{{End-to-end blind field detection per live module, path-separated. "
           f"\\emph{{Struct.}}: keys confirmed by a cryptographic invariant (DER private key / Curve25519), "
           f"a path with \\emph{{zero non-key false positives by construction}}. \\emph{{Stat.}}: keys recovered "
           f"by the statistical AES/length path, which has its own false positives (\\emph{{St.\\,FP}}). Overall "
           f"recall {round(100*T['det']/max(T['gt'],1),1)}\\% (95\\% Wilson CI [{lo}\\%, {hi}\\%], "
           f"{T['det']}/{T['gt']} keys); by key type, asymmetric "
           f"{round(100*T['asym_det']/max(T['asym'],1),1)}\\% vs.\\ symmetric "
           f"{round(100*T['sym_det']/max(T['sym'],1),1)}\\%.}}\n"
           "\\label{tab:endtoend}\n"
           "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
           "\\textbf{Module} & \\textbf{Keys} & \\textbf{Struct.} & \\textbf{Stat.} & \\textbf{St.\\,FP} & "
           "\\textbf{Recall} \\\\\n\\midrule\n"
           f"{body}\n\\midrule\n{tot}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    open(a.tex, "w", encoding="utf-8").write(tex)
    print(f"\nWrote {a.tex}")


if __name__ == "__main__":
    main()
