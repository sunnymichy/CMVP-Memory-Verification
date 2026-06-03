"""
fp_diagnose.py  --  explain field false positives by KEY IDENTITY, not raw bytes
================================================================================
field_eval.py matches a predicted block to ground truth by raw-byte containment. But the
SAME logical key can be materialized in a different DER encoding (PKCS#8 vs PKCS#1/SEC1,
with/without parameters), so a byte match fails even though it is the same key. This script
re-checks every byte-level false positive at the cryptographic IDENTITY level:
  RSA -> public modulus n;  EC -> SubjectPublicKeyInfo;  Ed25519/X25519 -> raw public key.
It reports, per module, how many byte-level FPs are actually the SAME key in a different
encoding (a ground-truth-coverage artifact) versus GENUINE extra detections, and the
encoding-adjusted (identity-level) recall / actionable precision.

USAGE
-----
    python fp_diagnose.py --indir field_modules
"""
import argparse, glob, json, os


def _key_identity(k):
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, ed25519, x25519
    from cryptography.hazmat.primitives import serialization as s
    if isinstance(k, rsa.RSAPrivateKey):
        return ("rsa", k.public_key().public_numbers().n)
    if isinstance(k, ec.EllipticCurvePrivateKey):
        return ("ec", k.public_key().public_bytes(s.Encoding.DER, s.PublicFormat.SubjectPublicKeyInfo))
    if isinstance(k, ed25519.Ed25519PrivateKey):
        return ("ed25519", k.public_key().public_bytes(s.Encoding.Raw, s.PublicFormat.Raw))
    if isinstance(k, x25519.X25519PrivateKey):
        return ("x25519", k.public_key().public_bytes(s.Encoding.Raw, s.PublicFormat.Raw))
    return None


def identity(hexstr):
    """Identity of a private-key blob (DER, or raw 32-byte Curve25519/Ed25519 seed)."""
    try:
        b = bytes.fromhex(hexstr)
    except Exception:
        return ("bad", hexstr)
    try:
        from cryptography.hazmat.primitives.serialization import load_der_private_key
        k = load_der_private_key(b, password=None)
        idn = _key_identity(k)
        if idn:
            return idn
    except Exception:
        pass
    if len(b) == 32:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization as s
            pub = Ed25519PrivateKey.from_private_bytes(b).public_key().public_bytes(
                s.Encoding.Raw, s.PublicFormat.Raw)
            return ("ed25519", pub)
        except Exception:
            pass
    return ("raw", b)


def _byte_match(g, b):
    g, b = g.lower(), b.lower()
    return g == b or g in b or b in g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="field_modules")
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(args.indir, "*.json")))
             if not f.endswith(".groundtruth.json") and not f.endswith(".predictions.json")]

    tot = {"keys": 0, "det_byte": 0, "fp_byte": 0, "det_id": 0, "fp_id": 0,
           "fp_same_key": 0, "fp_genuine": 0}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        if "ground_truth_keys" not in d or "predicted_key_blocks" not in d:
            continue
        gts = d["ground_truth_keys"]
        preds = d["predicted_key_blocks"]
        gt_ids = {}
        for i, g in enumerate(gts):
            gt_ids[i] = identity(g.get("value_hex", ""))
        gt_id_set = set(gt_ids.values())

        # byte-level (as field_eval)
        det_byte_ids, matched_blocks = set(), set()
        for gi, g in enumerate(gts):
            for bi, p in enumerate(preds):
                if _byte_match(g.get("value_hex", ""), p.get("value_hex", "")):
                    det_byte_ids.add(gi); matched_blocks.add(bi)
        fp_blocks = [bi for bi in range(len(preds)) if bi not in matched_blocks]

        # identity-level
        det_id_keys = set()
        for gi, gid in gt_ids.items():
            if any(identity(p.get("value_hex", "")) == gid for p in preds):
                det_id_keys.add(gi)
        fp_same, fp_genuine = 0, 0
        details = []
        for bi in fp_blocks:
            pid = identity(preds[bi].get("value_hex", ""))
            same = pid in gt_id_set
            if same:
                fp_same += 1
            else:
                fp_genuine += 1
            details.append((preds[bi].get("addr", "?"), preds[bi].get("type", "?"),
                            pid[0], "SAME-KEY (diff encoding)" if same else "genuine-extra"))

        n = len(gts)
        print(f"\n== {d.get('module','?')} ==")
        print(f"  GT keys: {n}  predicted: {len(preds)}")
        print(f"  byte-level: detected {len(det_byte_ids)}, FP {len(fp_blocks)}")
        print(f"  identity-level: detected {len(det_id_keys)}, "
              f"genuine FP {fp_genuine}  (byte-FPs that are same-key-diff-encoding: {fp_same})")
        for ad, ty, kind, verdict in details:
            print(f"      FP {ad} [{ty}/{kind}] -> {verdict}")

        tot["keys"] += n
        tot["det_byte"] += len(det_byte_ids); tot["fp_byte"] += len(fp_blocks)
        tot["det_id"] += len(det_id_keys); tot["fp_id"] += fp_genuine
        tot["fp_same_key"] += fp_same; tot["fp_genuine"] += fp_genuine

    def prec(tp, fp):
        return 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    print("\n=================== OVERALL ===================")
    print(f"  byte-level:     recall {100*tot['det_byte']/tot['keys']:.1f}%  "
          f"actionable precision {prec(tot['det_byte'], tot['fp_byte']):.1f}%  (FP {tot['fp_byte']})")
    print(f"  identity-level: recall {100*tot['det_id']/tot['keys']:.1f}%  "
          f"actionable precision {prec(tot['det_id'], tot['fp_id']):.1f}%  (genuine FP {tot['fp_id']})")
    print(f"  of {tot['fp_byte']} byte-level FPs: {tot['fp_same_key']} were the same key in a "
          f"different encoding, {tot['fp_genuine']} genuine extra.")


if __name__ == "__main__":
    main()
