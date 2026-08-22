import argparse
import csv
import getpass
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import conf

from exact_sync.v1.api.image_sets_api import ImageSetsApi
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.image_registration_api import ImageRegistrationApi
from exact_sync.v1.api.products_api import ProductsApi
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.configuration import Configuration

DEFAULT_USERNAME = "alha7503"
EXACT_HOST = "https://exact.hs-flensburg.de"
RULE = "─" * 72
DEFAULT_STATS_CSV = conf.PROJECT_ROOT / "data" / "muromi_wsi_cross_stain_pairs.csv"

SCANNER_CODES = frozenset({
    "3DHP1", "3DHP250", "AAT2", "ACS2", "ASSCS", "ATHS60", "DEHS60", "FZAZ1",
    "NLHOL", "NLHS360", "NLHXR", "OVS1", "UKHS60", "ZAZ1",
})
_ROI_SUFFIX_RE = re.compile(r"\s*\(ROI[^)]*\)\s*$", re.IGNORECASE)
_SECTION_JOINED_RE = re.compile(r"^F\d+\+F\d+$", re.IGNORECASE)
_SECTION_SINGLE_RE = re.compile(r"^F\d+$", re.IGNORECASE)


def _exact_client() -> ApiClient:
    configuration = Configuration()
    configuration.username = os.getenv("EXACT_USERNAME") or DEFAULT_USERNAME
    configuration.password = os.getenv("EXACT_PASSWORD") or getpass.getpass("EXACT password: ")
    configuration.host = EXACT_HOST
    return ApiClient(configuration)


def _local_image_ids() -> set[int]:
    return {
        int(p.stem)
        for p in conf.IMAGE_DIR.glob("*.data")
        if p.stem.isdigit()
    }


def _match_labels(registrations, available_ids: set[int]) -> list[dict]:
    labels = []
    for reg in registrations:
        src_id = reg.source_image
        tgt_id = reg.target_image
        if src_id in available_ids and tgt_id in available_ids:
            labels.append(
                {
                    "source_image_id": src_id,
                    "target_image_id": tgt_id,
                    "registration_error": reg.registration_error,
                    "transformation_matrix": reg.transformation_matrix,
                }
            )
    labels.sort(key=lambda e: (e["target_image_id"], e["source_image_id"]))
    return labels


def _fetch_image_meta(images_api: ImagesApi, image_ids: set[int]) -> dict[int, object]:
    by_id: dict[int, object] = {}
    for image_id in sorted(image_ids):
        try:
            by_id[image_id] = images_api.retrieve_image(id=image_id)
        except Exception as exc:
            print(f"[WARN] retrieve_image({image_id}) failed: {exc}")
    return by_id


def _fetch_set_meta(image_sets_api: ImageSetsApi, set_ids: set[int]) -> dict[int, object]:
    by_id: dict[int, object] = {}
    for set_id in sorted(set_ids):
        try:
            by_id[set_id] = image_sets_api.retrieve_image_set(id=set_id)
        except Exception as exc:
            print(f"[WARN] retrieve_image_set({set_id}) failed: {exc}")
    return by_id


def _fetch_product_names(products_api: ProductsApi, product_ids: set[int]) -> dict[int, str]:
    names: dict[int, str] = {}
    for product_id in sorted(product_ids):
        try:
            product = products_api.retrieve_product(id=product_id)
            names[product_id] = getattr(product, "name", None) or f"product_{product_id}"
        except Exception as exc:
            print(f"[WARN] retrieve_product({product_id}) failed: {exc}")
            names[product_id] = f"product_{product_id}"
    return names


def _set_id(img) -> int | None:
    raw = getattr(img, "image_set", None)
    if raw is None or isinstance(raw, str):
        return None
    return int(raw)


def _banner(title: str) -> None:
    print()
    print(RULE)
    print(f"  {title}")
    print(RULE)


def _short(text: str, width: int) -> str:
    text = text or "?"
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def is_roi_imageset(name: str) -> bool:
    return "(ROI" in name


def parse_stain(name: str) -> str:
    has_phh3 = "PHH3" in name
    has_he = bool(re.search(r"(?<![A-Z0-9])HE(?![A-Z0-9])", name))
    if has_phh3 and has_he:
        raise ValueError(f"imageset name matches both HE and PHH3: {name!r}")
    if has_phh3:
        return "PHH3"
    if has_he:
        return "HE"
    raise ValueError(f"imageset name matches neither HE nor PHH3: {name!r}")


def parse_scanner(name: str) -> str:
    stripped = _ROI_SUFFIX_RE.sub("", name).strip()
    if not stripped:
        raise ValueError(f"imageset name empty after stripping ROI suffix: {name!r}")
    token = stripped.split()[-1]
    if token not in SCANNER_CODES:
        raise ValueError(
            f"unrecognised scanner token {token!r} in imageset name {name!r} "
            f"(expected one of {sorted(SCANNER_CODES)})"
        )
    return token


def parse_case_section(name: str) -> tuple[str, str]:
    """
    Derive (case, section) from an imageset name.

    Handles the irregular naming on this server explicitly: stain-before-section
    (FU6 HE F1 …), combined sections (F1 + F2 / F1+F2), VMU0 LUC vs VMU0 MCT
    cohorts, and MCT … Berlin (CB) / (CB) location tags.
    """
    core = _ROI_SUFFIX_RE.sub("", name).strip()
    stain = parse_stain(name)
    scanner = parse_scanner(name)
    tokens = [
        t for t in core.split()
        if t != stain and t != scanner
        and t.lower() != "berlin"
        and not (t.startswith("(") and t.endswith(")"))
    ]
    if not tokens:
        raise ValueError(f"no case/section tokens left in imageset name: {name!r}")

    section: str | None = None
    case_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if section is None and _SECTION_JOINED_RE.fullmatch(t):
            section = t.upper().replace(" ", "")
            i += 1
            continue
        if (
            section is None
            and _SECTION_SINGLE_RE.fullmatch(t)
            and i + 2 < len(tokens)
            and tokens[i + 1] == "+"
            and _SECTION_SINGLE_RE.fullmatch(tokens[i + 2])
        ):
            section = f"{t.upper()}+{tokens[i + 2].upper()}"
            i += 3
            continue
        if section is None and _SECTION_SINGLE_RE.fullmatch(t):
            section = t.upper()
            i += 1
            continue
        case_tokens.append(t)
        i += 1

    if section is None:
        raise ValueError(f"no section (F#) found in imageset name: {name!r}")
    if not case_tokens:
        raise ValueError(f"no case token left in imageset name: {name!r}")
    return " ".join(case_tokens), section


@dataclass(frozen=True)
class SetInfo:
    set_id: int
    name: str
    is_roi: bool
    stain: str
    scanner: str
    case: str
    section: str


@dataclass(frozen=True)
class LoadedCatalog:
    sets: list
    registrations: list
    image_to_set: dict[int, int]
    set_by_id: dict[int, object]
    reported_set_count: object
    reported_reg_count: object


def _load_catalog(
    image_sets_api: ImageSetsApi, image_registration_api: ImageRegistrationApi
) -> LoadedCatalog:
    sets_resp = image_sets_api.list_image_sets(pagination=False, limit=1000)
    sets = list(sets_resp.results)
    image_to_set: dict[int, int] = {}
    set_by_id: dict[int, object] = {}
    for s in sets:
        sid = int(s.id)
        set_by_id[sid] = s
        for image_id in getattr(s, "images", None) or []:
            image_to_set[int(image_id)] = sid

    regs_resp = image_registration_api.list_image_registrations_with_http_info(limit=4000)
    registrations = list(regs_resp[0].results)
    return LoadedCatalog(
        sets=sets,
        registrations=registrations,
        image_to_set=image_to_set,
        set_by_id=set_by_id,
        reported_set_count=getattr(sets_resp, "count", "?"),
        reported_reg_count=getattr(regs_resp[0], "count", None),
    )


def _classify_imageset(set_id: int, name: str) -> SetInfo:
    case, section = parse_case_section(name)
    return SetInfo(
        set_id=set_id,
        name=name,
        is_roi=is_roi_imageset(name),
        stain=parse_stain(name),
        scanner=parse_scanner(name),
        case=case,
        section=section,
    )


def print_metadata(labels: list[dict], images_api: ImagesApi, image_sets_api: ImageSetsApi, products_api: ProductsApi) -> None:
    image_ids = {int(e["source_image_id"]) for e in labels} | {
        int(e["target_image_id"]) for e in labels
    }
    _banner(f"EXACT metadata  ·  {len(image_ids)} images  ·  {len(labels)} pairs")
    print("  Fetching image + image-set details…")
    images = _fetch_image_meta(images_api, image_ids)

    set_ids = {sid for img in images.values() if (sid := _set_id(img)) is not None}
    sets = _fetch_set_meta(image_sets_api, set_ids)

    product_ids: set[int] = set()
    for s in sets.values():
        for pid in getattr(s, "product_set", None) or []:
            product_ids.add(int(pid))
    products = _fetch_product_names(products_api, product_ids) if product_ids else {}

    def set_name(set_id: int | None) -> str:
        if set_id is None:
            return "?"
        s = sets.get(set_id)
        if s is None:
            return f"set_{set_id}"
        return getattr(s, "name", None) or f"set_{set_id}"

    def product_label(set_id: int | None) -> str:
        if set_id is None or set_id not in sets:
            return "—"
        ids = [int(p) for p in (getattr(sets[set_id], "product_set", None) or [])]
        if not ids:
            return "—"
        return ", ".join(products.get(i, str(i)) for i in ids)

    by_set: dict[int | None, list[int]] = defaultdict(list)
    for image_id, img in images.items():
        by_set[_set_id(img)].append(image_id)

    print()
    print("  IMAGE SETS (local coverage)")
    print(f"  {'set_id':>6}  {'local':>5}  {'in set':>6}  {'name':<28}  products")
    print(f"  {'------':>6}  {'-----':>5}  {'------':>6}  {'-'*28}  {'-'*24}")
    for set_id in sorted(by_set.keys(), key=lambda x: (x is None, x or 0)):
        ids = sorted(by_set[set_id])
        s = sets.get(set_id) if set_id is not None else None
        n_set = len(getattr(s, "images", None) or []) if s is not None else "?"
        sid_s = str(set_id) if set_id is not None else "?"
        print(
            f"  {sid_s:>6}  {len(ids):>5}  {str(n_set):>6}  "
            f"{_short(set_name(set_id), 28):<28}  {_short(product_label(set_id), 40)}"
        )
        id_chunks = [ids[i : i + 12] for i in range(0, len(ids), 12)]
        for i, chunk in enumerate(id_chunks):
            prefix = "  ids:" if i == 0 else "      "
            print(f"  {prefix}  {', '.join(str(x) for x in chunk)}")

    print()
    print("  REGISTRATION PAIRS")
    print(
        f"  {'src':>6} → {'tgt':<6}  {'err':>8}  "
        f"{'source file':<34}  {'target file':<34}  sets"
    )
    print(
        f"  {'------':>6}   {'------':<6}  {'--------':>8}  "
        f"{'-'*34}  {'-'*34}  {'-'*28}"
    )
    for e in labels:
        src_id = int(e["source_image_id"])
        tgt_id = int(e["target_image_id"])
        src = images.get(src_id)
        tgt = images.get(tgt_id)
        src_sid = _set_id(src) if src is not None else None
        tgt_sid = _set_id(tgt) if tgt is not None else None
        src_nm = getattr(src, "filename", None) or getattr(src, "name", None) or "?"
        tgt_nm = getattr(tgt, "filename", None) or getattr(tgt, "name", None) or "?"
        err = e.get("registration_error")
        err_s = f"{err:.1f}" if isinstance(err, (int, float)) else "—"
        if src_sid == tgt_sid:
            sets_txt = set_name(src_sid)
        else:
            sets_txt = f"{set_name(src_sid)} → {set_name(tgt_sid)}  ✦ cross-set"
        print(
            f"  {src_id:>6} → {tgt_id:<6}  {err_s:>8}  "
            f"{_short(src_nm, 34):<34}  {_short(tgt_nm, 34):<34}  "
            f"{_short(sets_txt, 40)}"
        )

    print()
    print("  IMAGES")
    print(f"  {'id':>6}  {'size (W×H)':>15}  {'mpp':>7}  {'obj':>5}  {'set':<22}  filename")
    print(f"  {'------':>6}  {'-'*15}  {'-------':>7}  {'-----':>5}  {'-'*22}  {'-'*36}")
    for image_id in sorted(images):
        img = images[image_id]
        sid = _set_id(img)
        w = getattr(img, "width", None)
        h = getattr(img, "height", None)
        size = f"{w}×{h}" if w and h else "?"
        mpp = getattr(img, "mpp", None)
        mpp_s = f"{mpp:.4f}" if isinstance(mpp, (int, float)) else "?"
        obj = getattr(img, "objective_power", None)
        obj_s = f"{obj:.0f}×" if isinstance(obj, (int, float)) else "?"
        fn = getattr(img, "filename", None) or getattr(img, "name", None) or "?"
        print(
            f"  {image_id:>6}  {size:>15}  {mpp_s:>7}  {obj_s:>5}  "
            f"{_short(set_name(sid), 22):<22}  {fn}"
        )


def print_all_set_pairs(catalog: LoadedCatalog) -> None:
    _banner("all EXACT image sets + registration pairs")
    print("  listing image sets…")
    sets = catalog.sets
    print(f"  {len(sets)} image sets  (reported count {catalog.reported_set_count})")

    image_to_set = catalog.image_to_set
    set_by_id = catalog.set_by_id

    def set_name(set_id: int | None) -> str:
        if set_id is None:
            return "?"
        s = set_by_id.get(set_id)
        if s is None:
            return f"set_{set_id}"
        return getattr(s, "name", None) or f"set_{set_id}"

    print("  listing registrations…")
    registrations = catalog.registrations
    reported = catalog.reported_reg_count
    print(f"  {len(registrations)} registrations  (reported count {reported if reported is not None else '?'})")

    pair_counts: dict[tuple[int | None, int | None], int] = defaultdict(int)
    unique_pairs: dict[tuple[int | None, int | None], set[tuple[int, int]]] = defaultdict(set)
    unmatched_images: set[int] = set()
    for reg in registrations:
        src_id = int(reg.source_image)
        tgt_id = int(reg.target_image)
        src_set = image_to_set.get(src_id)
        tgt_set = image_to_set.get(tgt_id)
        if src_set is None:
            unmatched_images.add(src_id)
        if tgt_set is None:
            unmatched_images.add(tgt_id)
        pair_counts[(src_set, tgt_set)] += 1
        unique_pairs[(src_set, tgt_set)].add((src_id, tgt_id))

    print()
    print("  IMAGE SETS")
    print(f"  {'id':>6}  {'n_img':>5}  name")
    print(f"  {'------':>6}  {'-----':>5}  {'-'*48}")
    for s in sorted(sets, key=lambda x: (getattr(x, "name", None) or "").lower()):
        n_img = len(getattr(s, "images", None) or [])
        print(f"  {int(s.id):>6}  {n_img:>5}  {getattr(s, 'name', None) or '?'}")

    print()
    print("  REGISTRATION PAIRS BY SET (source → target)")
    print(f"  {'n':>5}  {'uniq':>5}  source set → target set")
    print(f"  {'-----':>5}  {'-----':>5}  {'-'*56}")
    rows = sorted(
        pair_counts.items(),
        key=lambda kv: (-kv[1], set_name(kv[0][0]), set_name(kv[0][1])),
    )
    for (src_set, tgt_set), n in rows:
        n_uniq = len(unique_pairs[(src_set, tgt_set)])
        cross = "  ✦ cross-set" if src_set != tgt_set else ""
        print(
            f"  {n:>5}  {n_uniq:>5}  "
            f"{set_name(src_set)} → {set_name(tgt_set)}{cross}"
        )

    print()
    print(f"  total registration records  {len(registrations)}")
    print(f"  unique (src,tgt) pairs      {sum(len(v) for v in unique_pairs.values())}")
    print(f"  images not in any listed set {len(unmatched_images)}")


def print_registration_stats(catalog: LoadedCatalog, csv_path: Path) -> None:
    _banner("registration stats  ·  WSI / ROI · stain · scanner")
    image_to_set = catalog.image_to_set
    registrations = catalog.registrations

    used_set_ids: set[int] = set()
    for reg in registrations:
        for image_id in (int(reg.source_image), int(reg.target_image)):
            sid = image_to_set.get(image_id)
            if sid is not None:
                used_set_ids.add(sid)
    set_info = {
        sid: _classify_imageset(
            sid, getattr(catalog.set_by_id[sid], "name", None) or f"set_{sid}"
        )
        for sid in sorted(used_set_ids)
    }

    n_total = len(registrations)
    n_unresolved = 0
    n_roi = 0
    n_wsi = 0
    n_cross = 0
    n_cross_same_scanner = 0
    n_cross_cross_scanner = 0
    n_same_he = 0
    n_same_phh3 = 0
    cross_rows: list[dict] = []
    cross_wsi_ids: set[int] = set()
    cross_phh3_ids: set[int] = set()

    for reg in registrations:
        src_id = int(reg.source_image)
        tgt_id = int(reg.target_image)
        src_sid = image_to_set.get(src_id)
        tgt_sid = image_to_set.get(tgt_id)
        if src_sid is None or tgt_sid is None:
            n_unresolved += 1
            continue
        src = set_info[src_sid]
        tgt = set_info[tgt_sid]
        if src.is_roi or tgt.is_roi:
            n_roi += 1
            continue
        n_wsi += 1
        stains = {src.stain, tgt.stain}
        if stains == {"HE", "PHH3"}:
            n_cross += 1
            if src.scanner == tgt.scanner:
                n_cross_same_scanner += 1
            else:
                n_cross_cross_scanner += 1
            if src.case != tgt.case or src.section != tgt.section:
                case, section = src.case, src.section
                print(
                    f"  [WARN] case/section mismatch "
                    f"{src.name!r} ({src.case}/{src.section}) vs "
                    f"{tgt.name!r} ({tgt.case}/{tgt.section}) — using source"
                )
            else:
                case, section = src.case, src.section
            cross_rows.append({
                "src_id": src_id,
                "tgt_id": tgt_id,
                "src_imageset": src.name,
                "tgt_imageset": tgt.name,
                "case": case,
                "section": section,
                "src_scanner": src.scanner,
                "tgt_scanner": tgt.scanner,
            })
            cross_wsi_ids.add(src_id)
            cross_wsi_ids.add(tgt_id)
            if src.stain == "PHH3":
                cross_phh3_ids.add(src_id)
            if tgt.stain == "PHH3":
                cross_phh3_ids.add(tgt_id)
        elif stains == {"HE"}:
            n_same_he += 1
        elif stains == {"PHH3"}:
            n_same_phh3 += 1
        else:
            raise ValueError(f"unexpected stain pair {stains} for {src.name!r} → {tgt.name!r}")

    n_phh3 = len(cross_phh3_ids)
    reuse = (n_cross / n_phh3) if n_phh3 else float("nan")

    print(f"  classified imagesets                  {len(set_info)}")
    print(f"  total records                         {n_total}")
    if n_unresolved:
        print(f"  unresolved (image not in any set)     {n_unresolved}")
    print(f"  WSI-only records                      {n_wsi}")
    print(f"    cross-stain (HE ↔ PHH3)             {n_cross}")
    print(f"      same scanner                      {n_cross_same_scanner}")
    print(f"      cross scanner                     {n_cross_cross_scanner}")
    print(f"    same-stain                          {n_same_he + n_same_phh3}")
    print(f"      HE ↔ HE                           {n_same_he}")
    print(f"      PHH3 ↔ PHH3                       {n_same_phh3}")
    print(f"  records with ≥1 ROI side              {n_roi}")
    print(f"  unique WSIs in cross-stain WSI-only   {len(cross_wsi_ids)}")
    print(f"  distinct PHH3 WSIs in that set        {n_phh3}")
    if n_phh3:
        print(f"  reuse factor (pairs / PHH3 WSIs)      {reuse:.3f}")
    else:
        print("  reuse factor (pairs / PHH3 WSIs)      n/a")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "src_id", "tgt_id", "src_imageset", "tgt_imageset",
        "case", "section", "src_scanner", "tgt_scanner",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cross_rows)
    print()
    print(f"  wrote {len(cross_rows)} WSI-only cross-stain pairs → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EXACT registration labels for local images.")
    parser.add_argument(
        "--meta",
        action="store_true",
        help="print image / image-set metadata to the console",
    )
    parser.add_argument(
        "--meta-only",
        action="store_true",
        help="print metadata and skip writing the labels JSON",
    )
    parser.add_argument(
        "--all-sets",
        action="store_true",
        help="list every image set and count registration pairs per set pair (ignores local files)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="break down registrations by WSI/ROI, stain and scanner; write WSI cross-stain CSV",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=DEFAULT_STATS_CSV,
        help=f"output path for WSI-only cross-stain pair CSV (default: {DEFAULT_STATS_CSV})",
    )
    args = parser.parse_args()
    print_meta = bool(args.meta or args.meta_only)
    write_labels = not bool(args.meta_only)

    _banner("fetch_labels")
    print(f"  system   {conf.SYSTEM_PREFIX}")
    print(f"  images   {conf.IMAGE_DIR}")
    print(f"  labels   {conf.LABELS_PATH}")

    client = _exact_client()
    image_registration_api = ImageRegistrationApi(client)
    images_api = ImagesApi(client)
    image_sets_api = ImageSetsApi(client)
    products_api = ProductsApi(client)

    if args.all_sets or args.stats:
        if not args.all_sets:
            print("  listing image sets + registrations…")
        catalog = _load_catalog(image_sets_api, image_registration_api)
        if args.all_sets:
            print_all_set_pairs(catalog)
        if args.stats:
            print_registration_stats(catalog, Path(args.stats_csv))
        return

    if not conf.IMAGE_DIR.exists():
        print(f"\n  ERROR  image directory not found: {conf.IMAGE_DIR}")
        sys.exit(1)

    available_ids = _local_image_ids()
    if not available_ids:
        print(f"\n  ERROR  no *.data files in {conf.IMAGE_DIR}")
        sys.exit(1)

    print(f"  local    {len(available_ids)} *.data files")

    print("  server   listing registrations…")
    response = image_registration_api.list_image_registrations_with_http_info(limit=4000)
    registrations = response[0].results
    labels = _match_labels(registrations, available_ids)
    print(f"  server   {len(registrations)} registrations  →  {len(labels)} match local")

    if not labels:
        print("\n  WARNING  no matching pairs found.")
        return

    if print_meta:
        print_metadata(labels, images_api, image_sets_api, products_api)

    if not write_labels:
        print()
        print(RULE)
        print("  meta-only · labels file not written")
        print(RULE)
        return

    conf.LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(conf.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)

    print()
    print(RULE)
    print(f"  wrote {len(labels)} pairs → {conf.LABELS_PATH}")
    print(RULE)


if __name__ == "__main__":
    main()
