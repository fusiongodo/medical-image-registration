#!/usr/bin/env bash
#
# Operability smoke check for the medical-image-registration Svelte app.
#
# Hits the live HTTP endpoints for a couple of pairs/depths and asserts each
# returns HTTP 200 plus a minimal response shape. This is the repeatable
# green/red signal run after every refactor milestone (see REFACTOR_CHECKLIST.md).
#
# The dev server must already be running (npm run dev). Pair 0 must have cached
# candidates for depths 3+ (data/c2f_cache/0_d3.json etc.).
#
# Usage: scripts/smoke.sh [base_url]        # default http://localhost:5173
set -u

BASE="${1:-http://localhost:5173}"
PASS=0
FAIL=0

# check_json <name> <url> <python-bool-expr over dict `d`>
check_json() {
	local name="$1" url="$2" expr="$3" tmp code
	tmp="$(mktemp)"
	code="$(curl -s -o "$tmp" -w '%{http_code}' "$url")"
	if [ "$code" != "200" ]; then
		echo "FAIL  $name  (http $code)  $url"
		FAIL=$((FAIL + 1))
		rm -f "$tmp"
		return
	fi
	if python3 -c "import json; d=json.load(open('$tmp')); assert ($expr)" 2>/dev/null; then
		echo "PASS  $name"
		PASS=$((PASS + 1))
	else
		echo "FAIL  $name  (bad shape)  $url"
		FAIL=$((FAIL + 1))
	fi
	rm -f "$tmp"
}

# check_png <name> <url>
check_png() {
	local name="$1" url="$2" out code ctype size
	out="$(curl -s -o /dev/null -w '%{http_code} %{content_type} %{size_download}' "$url")"
	code="$(printf '%s' "$out" | awk '{print $1}')"
	ctype="$(printf '%s' "$out" | awk '{print $2}')"
	size="$(printf '%s' "$out" | awk '{print $3}')"
	if [ "$code" = "200" ] && [ "${ctype%%;*}" = "image/png" ] && [ "${size:-0}" -gt 1000 ]; then
		echo "PASS  $name  (${size}b)"
		PASS=$((PASS + 1))
	else
		echo "FAIL  $name  (http $code $ctype ${size}b)  $url"
		FAIL=$((FAIL + 1))
	fi
}

echo "Smoke check against $BASE"
echo "----------------------------------------"

check_json "tiles pair0 L3"      "$BASE/api/live-crop/tiles?pair=0&level=3" "d.get('ok') is True and isinstance(d.get('tiles'), list) and len(d['tiles']) > 0"
check_json "tiles pair1 L1"      "$BASE/api/live-crop/tiles?pair=1&level=1" "d.get('ok') is True and isinstance(d.get('tiles'), list)"

check_png  "tile  pair0 L3 he"   "$BASE/api/live-crop/tile?pair=0&level=3&x=1&y=0&side=he"
check_png  "tile  pair0 L3 ihc"  "$BASE/api/live-crop/tile?pair=0&level=3&x=1&y=0&side=ihc"

check_json "candidates pair0 L3" "$BASE/api/c2f/candidates?pair=0&depth=3" "d.get('cached') is True and isinstance(d.get('candidates'), list) and len(d['candidates']) > 0"
check_json "candidates pair0 L4" "$BASE/api/c2f/candidates?pair=0&depth=4" "d.get('cached') is True and isinstance(d.get('candidates'), list)"

check_json "refit pair0 L3 keep" "$BASE/api/c2f/refit?pair=0&depth=3&keep=0.95" "'tiles' in d and 'tau' in d and isinstance(d['tiles'], list)"

check_json "field pair0 L4"      "$BASE/api/field?pair=0&depth=4" "isinstance(d, dict)"
check_json "annotations p0 L3"   "$BASE/api/annotations?pair=0&depth=3" "isinstance(d, dict)"
check_json "field-set pair0"     "$BASE/api/c2f/field-set?pair=0" "isinstance(d.get('sets'), list)"

echo "----------------------------------------"
echo "PASS=$PASS  FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
