<script lang="ts">
	import { goto, invalidateAll } from '$app/navigation';
	import { deriveStatus, nextDepthForPair, MAX_DEPTH, type ValidationStore } from '$lib/types';
	import type { TileMeta } from '$lib/types';
	import OverlayCanvas from '$lib/OverlayCanvas.svelte';
	import LnccScore from '$lib/LnccScore.svelte';
	import PointCanvas from '$lib/PointCanvas.svelte';
	import DisplacedOverlay from '$lib/DisplacedOverlay.svelte';

	let {
		data
	} = $props<{
		data: {
			pairId: number;
			depth: number;
			tiles: TileMeta[];
			validation: ValidationStore;
			smooth: boolean;
			smoothAvailable: boolean;
		};
	}>();

	let submitting = $state(false);
	let patchSize = $state(11);
	let effectivePatchSize = $state(11);
	let patchDebounce: ReturnType<typeof setTimeout> | null = null;

	function onPatchInput(e: Event) {
		patchSize = (e.target as HTMLInputElement).valueAsNumber;
		if (patchDebounce) clearTimeout(patchDebounce);
		patchDebounce = setTimeout(() => { effectivePatchSize = patchSize; }, 400);
	}
	let sortByScore = $state(false);
	let sortByFactor = $state(false);
	let scores = $state<Map<string, number>>(new Map());
	let autoLnccScores = $state<Map<string, number>>(new Map());
	let cachedScores = $state<Record<string, { lncc?: number; sq?: number }>>({});
	let pendingEntries: Record<string, { lncc?: number; sq?: number }> = {};
	let saveTimer: ReturnType<typeof setTimeout> | null = null;

	interface TileMetrics { lncc2: number; lncc2_auto: number; delta_px: number; factor_auto: number; }
	let tileMetrics = $state<Map<string, TileMetrics>>(new Map());

	let tileKeypoints = $state<Map<string, number[][]>>(new Map());
	let showKeypoints = $state(false);

	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		let stale = false;
		fetch(`/api/tile-keypoints?pair=${pair}&depth=${depth}`)
			.then((r) => r.json())
			.then((fetched: Record<string, number[][]>) => {
				if (stale) return;
				tileKeypoints = new Map(Object.entries(fetched));
			});
		return () => { stale = true; tileKeypoints = new Map(); };
	});

	// Fetch cached scores — reads only pair/depth/effectivePatchSize, never scores
	$effect(() => {
		const pair = data.pairId, depth = data.depth, ps = effectivePatchSize;
		let stale = false;
		fetch(`/api/scores?pair=${pair}&depth=${depth}&patchSize=${ps}`)
			.then((r) => r.json())
			.then((fetched: Record<string, { lncc?: number; sq?: number }>) => {
				if (stale) return;
				cachedScores = fetched;
			});
		return () => { stale = true; cachedScores = {}; };
	});

	// Seed sort-map from cache — reads only cachedScores, never scores
	$effect(() => {
		const cached = cachedScores;
		const nextSq = new Map<string, number>();
		for (const [tile, v] of Object.entries(cached)) {
			if (v.sq !== undefined) nextSq.set(tile, v.sq);
		}
		scores = nextSq;
	});

	function recordScore(tile: string, s: number) {
		scores = new Map(scores.set(tile, s));
	}

	function recordAutoLnccScore(tile: string, s: number) {
		autoLnccScores = new Map(autoLnccScores.set(tile, s));
	}

	function queueScore(tile: string, type: 'sq', value: number) {
		pendingEntries[tile] = { ...pendingEntries[tile], [type]: value };
		if (saveTimer) clearTimeout(saveTimer);
		saveTimer = setTimeout(flushScores, 2000);
	}

	async function flushScores() {
		const entries = pendingEntries;
		if (Object.keys(entries).length === 0) return;
		pendingEntries = {};
		await fetch('/api/scores', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ pair_id: data.pairId, depth: data.depth, patchSize: effectivePatchSize, entries })
		});
	}

	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		let stale = false;
		fetch(`/api/tile-metrics?pair=${pair}&depth=${depth}`)
			.then((r) => r.json())
			.then((fetched: Record<string, TileMetrics>) => {
				if (stale) return;
				tileMetrics = new Map(Object.entries(fetched));
				for (const [tile, m] of tileMetrics) {
					scores = new Map(scores.set(tile, m.lncc2));
					autoLnccScores = new Map(autoLnccScores.set(tile, m.lncc2_auto));
				}
			});
		return () => { stale = true; tileMetrics = new Map(); };
	});

	const sortedTiles = $derived.by(() => {
		const factorOf = (tile: string): number | undefined => {
			const m = tileMetrics.get(tile);
			if (m) return m.factor_auto;
			const sq = scores.get(tile), asq = autoLnccScores.get(tile);
			return sq !== undefined && asq !== undefined && sq > 0 ? asq / sq : undefined;
		};
		if (sortByFactor) {
			const factored = data.tiles
				.filter((t: TileMeta) => factorOf(t.tile) !== undefined)
				.sort((a: TileMeta, b: TileMeta) => (factorOf(b.tile) ?? 0) - (factorOf(a.tile) ?? 0));
			const rest = data.tiles.filter((t: TileMeta) => factorOf(t.tile) === undefined);
			return [...factored, ...rest];
		}
		const map = sortByScore ? scores : null;
		if (!map) return data.tiles;
		const scored = data.tiles
			.filter((t: TileMeta) => map.has(t.tile))
			.sort((a: TileMeta, b: TileMeta) => (map.get(b.tile) ?? 0) - (map.get(a.tile) ?? 0));
		const unscored = data.tiles.filter((t: TileMeta) => !map.has(t.tile));
		return [...scored, ...unscored];
	});

	interface Point { x: number; y: number; }
	interface TileAnnotation { hePoints: Point[]; ihcPoints: Point[]; }

	let activeRow = $state<string | null>(null);
	let displayOrder = $state<TileMeta[]>([]);

	$effect(() => {
		if (activeRow === null) displayOrder = sortedTiles;
	});
	let annotations = $state<Record<string, TileAnnotation>>({});

	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		let stale = false;
		fetch(`/api/annotations?pair=${pair}&depth=${depth}`)
			.then((r) => r.json())
			.then((fetched: Record<string, TileAnnotation>) => {
				if (stale) return;
				annotations = fetched;
			});
		return () => { stale = true; annotations = {}; activeRow = null; };
	});

	function addPoint(tile: string, side: 'he' | 'ihc', x: number, y: number) {
		const prev = annotations[tile] ?? { hePoints: [], ihcPoints: [] };
		const sideKey = side === 'he' ? 'hePoints' : 'ihcPoints';
		const updated: TileAnnotation = {
			...prev,
			[sideKey]: prev[sideKey].length < 2 ? [...prev[sideKey], { x, y }] : [{ x, y }]
		};
		annotations = { ...annotations, [tile]: updated };
		fetch('/api/annotations', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				pair_id: data.pairId,
				depth: data.depth,
				tile,
				hePoints: updated.hePoints,
				ihcPoints: updated.ihcPoints
			})
		});
	}

	// ── Auto-displacement (per tile, Python-written) ─────────────────────────
	interface AutoDisp { dx: number; dy: number; }
	let autoDisps = $state<Map<string, AutoDisp>>(new Map());
	let autoDispRefreshKey = $state(0);

	$effect(() => {
		autoDispRefreshKey; // tracked so Refresh button re-runs this effect
		const pair = data.pairId, depth = data.depth;
		let stale = false;
		fetch(`/api/python-displacement?pair=${pair}&depth=${depth}`)
			.then((r) => r.json())
			.then((fetched: Record<string, AutoDisp>) => {
				if (stale) return;
				autoDisps = new Map(Object.entries(fetched));
			});
		return () => { stale = true; autoDisps = new Map(); };
	});

	const autoTargets = $derived(displayOrder.slice(0, 5));

	const alignCommand = $derived(
		`python setup/auto-alignment/align.py ${data.pairId} ${data.depth}` +
		(autoTargets.length > 0 ? ' ' + autoTargets.map((t: TileMeta) => t.tile).join(' ') : '')
	);

	const alignAllCommand = $derived(
		`python setup/auto-alignment/align.py ${data.pairId} ${data.depth}`
	);

	let pollingActive = $state(false);
	let pollingInterval: ReturnType<typeof setInterval> | null = null;

	$effect(() => {
		if (!pollingActive) {
			if (pollingInterval) { clearInterval(pollingInterval); pollingInterval = null; }
			return;
		}
		pollingInterval = setInterval(() => { autoDispRefreshKey++; }, 2000);
		return () => { if (pollingInterval) { clearInterval(pollingInterval); pollingInterval = null; } };
	});

	const levelCorrelation = null as { r: number; n: number } | null; /* disabled
	$derived.by(() => {
		const xs: number[] = [];
		const ys: number[] = [];
		for (const t of data.tiles) {
			const sq  = scores.get(t.tile);
			const dsq = displScores.get(t.tile);
			const ann = annotations[t.tile];
			if (sq === undefined || dsq === undefined || sq <= 0 || !ann) continue;
			const pairs = Math.min(ann.hePoints.length, ann.ihcPoints.length);
			if (pairs === 0) continue;
			let dx = 0, dy = 0;
			for (let i = 0; i < pairs; i++) {
				dx += ann.hePoints[i].x - ann.ihcPoints[i].x;
				dy += ann.hePoints[i].y - ann.ihcPoints[i].y;
			}
			xs.push(Math.sqrt((dx / pairs) ** 2 + (dy / pairs) ** 2));
			ys.push(dsq / sq);
		}
		const n = xs.length;
		if (n < 2) return null;
		const mx = xs.reduce((a, b) => a + b, 0) / n;
		const my = ys.reduce((a, b) => a + b, 0) / n;
		let num = 0, dx2 = 0, dy2 = 0;
		for (let i = 0; i < n; i++) {
			const ex = xs[i] - mx, ey = ys[i] - my;
			num += ex * ey; dx2 += ex * ex; dy2 += ey * ey;
		}
		const den = Math.sqrt(dx2 * dy2);
		return den > 0 ? { r: num / den, n } : null;
	}); */

	const status = $derived(deriveStatus(data.validation, data.pairId));
	const alreadyEvaluated = $derived(
		data.validation[String(data.pairId)]?.[String(data.depth)] !== undefined
	);
	const currentDecision = $derived(
		data.validation[String(data.pairId)]?.[String(data.depth)]
	);

	async function decide(valid: boolean) {
		submitting = true;
		await fetch('/api/validation', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ pair_id: data.pairId, depth: data.depth, valid })
		});
		await invalidateAll();
		submitting = false;

		if (valid && data.depth < MAX_DEPTH) {
			goto(`/${data.pairId}/${data.depth + 1}`);
		}
	}

	async function reset() {
		submitting = true;
		await fetch('/api/validation', {
			method: 'DELETE',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ pair_id: data.pairId, depth: data.depth })
		});
		await invalidateAll();
		submitting = false;
	}

	function lnccColor(s: number): string {
		const t = Math.max(0, Math.min(1, s));
		return t < 0.5
			? `rgb(255,${Math.round(t * 2 * 255)},0)`
			: `rgb(${Math.round((1 - (t - 0.5) * 2) * 255)},255,0)`;
	}

	function depthLabel(d: number) {
		const grid = Math.pow(2, d);
		return `Level ${d} · ${grid}×${grid} grid`;
	}
</script>

<div class="viewer">
	<header>
		<div class="breadcrumb">
			<span class="pair-label">Pair {data.pairId}</span>
			<span class="sep">·</span>
			<span class="depth-label">{depthLabel(data.depth)}</span>
			<span class="tile-count">{data.tiles.length} tiles</span>
		</div>

		<label class="sort-control">
			<input type="checkbox" bind:checked={sortByScore} onchange={() => { if (sortByScore) sortByFactor = false; }} />
			<span>Sort by LNCC²</span>
		</label>
		<label class="sort-control">
			<input type="checkbox" bind:checked={sortByFactor} onchange={() => { if (sortByFactor) sortByScore = false; }} />
			<span>Sort by Factor</span>
		</label>
		<label class="sort-control">
			<input type="checkbox" bind:checked={showKeypoints} />
			<span>Keypoints</span>
		</label>
		{#if data.smooth}
			<a class="source-badge source-smooth" href="/{data.pairId}/{data.depth}">Standard</a>
		{:else if data.smoothAvailable}
			<a class="source-badge source-standard" href="/{data.pairId}/{data.depth}?source=smooth">Smooth IHC</a>
		{/if}

		<label class="patch-control">
			<span class="patch-label">Patch</span>
			<input type="range" min="3" max="51" step="2" value={patchSize} oninput={onPatchInput} />
			<span class="patch-value">{patchSize}px</span>
		</label>

		<div class="auto-disp-controls">
			<div class="auto-disp-control">
				<code class="align-cmd">{alignCommand}</code>
				<button class="btn btn-ghost btn-sm" onclick={() => navigator.clipboard.writeText(alignCommand)} title="Copy top-5 command">⎘</button>
				<button class="btn btn-auto" onclick={() => { autoDispRefreshKey++; }}>↺ Refresh</button>
			</div>
			<div class="auto-disp-control">
				{#if autoDisps.size > 0}
					<span class="auto-disp-badge" class:done-badge={autoDisps.size === data.tiles.length}>
						{autoDisps.size} / {data.tiles.length}{autoDisps.size === data.tiles.length ? ' ✓' : ''}
					</span>
				{/if}
				<code class="align-cmd">{alignAllCommand}</code>
				<button class="btn btn-ghost btn-sm" onclick={() => navigator.clipboard.writeText(alignAllCommand)} title="Copy all-tiles command">⎘</button>
				<button
					class="btn btn-auto"
					class:btn-polling={pollingActive}
					onclick={() => { pollingActive = !pollingActive; if (pollingActive) autoDispRefreshKey++; }}>
					{pollingActive ? '⏹ Stop' : '⟳ Watch'}
				</button>
			</div>
		</div>

		<div class="depth-nav">
			{#each Array.from({ length: MAX_DEPTH + 1 }, (_, i) => i) as d}
				{@const dv = data.validation[String(data.pairId)]?.[String(d)]}
				<a
					href={`/${data.pairId}/${d}`}
					class="depth-pip"
					class:current={d === data.depth}
					class:evaluated={dv !== undefined}
					class:passed={dv === true}
					class:failed={dv === false}
					title={depthLabel(d)}
				>
					{d}
				</a>
			{/each}
		</div>
	</header>

	{#if levelCorrelation !== null}
		<div class="agg-bar">
			<span class="agg-label">corr(|Δ|, Factor)</span>
			<span class="agg-value" class:agg-pos={levelCorrelation.r > 0} class:agg-neg={levelCorrelation.r < 0}>
				{levelCorrelation.r.toFixed(3)}
			</span>
			<span class="agg-n">n = {levelCorrelation.n}</span>
		</div>
	{/if}

	{#if data.tiles.length === 0}
		<div class="empty">No tiles found for this pair / depth.</div>
	{:else}
		<div class="scroll-wrap">
			<div class="tile-grid">
			<span class="col-header sticky-header"></span>
			<span class="col-header sticky-header">HE norm</span>
			<span class="col-header sticky-header">{data.smooth ? 'IHC smooth' : 'IHC norm'}</span>
			<span class="col-header sticky-header">Overlay</span>
			<span class="col-header sticky-header">{data.smooth ? 'Pre-aligned' : 'Auto overlay'}</span>
			<span class="col-header sticky-header">LNCC²</span>
			<span class="col-header sticky-header">|Δ| auto</span>
			<span class="col-header sticky-header">LNCC² auto</span>
			<span class="col-header sticky-header">Factor auto</span>

	{#each displayOrder as t (`${data.depth}-${t.tile}`)}
			{@const heSrc  = `/api/image?path=${encodeURIComponent(t.he)}`}
			{@const ihcSrc = `/api/image?path=${encodeURIComponent(t.ihc)}`}
			{@const isActive = activeRow === t.tile}
			{@const ann = annotations[t.tile] ?? { hePoints: [], ihcPoints: [] }}
		{@const tileAutoDisp = autoDisps.get(t.tile)}
		{@const m = tileMetrics.get(t.tile)}
		{@const sq = m?.lncc2 ?? scores.get(t.tile)}
		{@const asq = m?.lncc2_auto ?? autoLnccScores.get(t.tile)}
		{@const tileKps = showKeypoints ? (tileKeypoints.get(t.tile) ?? []) : []}
		<span
			class="tile-id"
			class:tile-id-active={isActive}
			onclick={() => { activeRow = isActive ? null : t.tile; }}
			role="button"
			tabindex="0"
			onkeydown={(e) => e.key === 'Enter' && (activeRow = isActive ? null : t.tile)}
		>{t.tile}</span>
		<PointCanvas src={heSrc} active={isActive} points={ann.hePoints}
			keypoints={tileKps}
			onpoint={(x, y) => addPoint(t.tile, 'he', x, y)} />
		<PointCanvas src={ihcSrc} active={isActive} points={ann.ihcPoints}
			onpoint={(x, y) => addPoint(t.tile, 'ihc', x, y)} />
		<OverlayCanvas {heSrc} {ihcSrc} />
		{#if data.smooth}
			<DisplacedOverlay {heSrc} {ihcSrc} dx={0} dy={0} keypoints={tileKps} />
		{:else if tileAutoDisp !== undefined}
			<DisplacedOverlay {heSrc} {ihcSrc} dx={tileAutoDisp.dx} dy={tileAutoDisp.dy}
				keypoints={tileKps} />
		{:else}
			<div class="factor-cell"><span class="factor-placeholder">…</span></div>
		{/if}
				{#if m}
					<div class="score-cell-pre" style:background={lnccColor(m.lncc2)}>
						<span class="value">{m.lncc2.toFixed(3)}</span>
					</div>
				{:else}
					<LnccScore {heSrc} {ihcSrc} patchSize={effectivePatchSize} squared={true}
						cachedScore={cachedScores[t.tile]?.sq}
						onscore={(s) => { recordScore(t.tile, s); queueScore(t.tile, 'sq', s); }} />
				{/if}
				<div class="factor-cell">
					{#if m}
						{m.delta_px.toFixed(1)}
					{:else if tileAutoDisp !== undefined}
						{Math.sqrt(tileAutoDisp.dx ** 2 + tileAutoDisp.dy ** 2).toFixed(1)}
					{:else}
						<span class="factor-placeholder">…</span>
					{/if}
				</div>
				{#if m}
					<div class="score-cell-pre" style:background={lnccColor(m.lncc2_auto)}>
						<span class="value">{m.lncc2_auto.toFixed(3)}</span>
					</div>
				{:else}
					<LnccScore {heSrc} {ihcSrc} patchSize={effectivePatchSize} squared={true}
						displaced={true} displaceX={tileAutoDisp?.dx} displaceY={tileAutoDisp?.dy}
						onscore={(s) => recordAutoLnccScore(t.tile, s)} />
				{/if}
				<div class="factor-cell" class:factor-positive={asq !== undefined && sq !== undefined && asq > sq}>
					{#if asq !== undefined && sq !== undefined && sq > 0}
						{(asq / sq).toFixed(3)}
					{:else}
						<span class="factor-placeholder">…</span>
					{/if}
				</div>
		{/each}
			</div>
		</div>
	{/if}

	<footer>
		{#if alreadyEvaluated}
			<div class="result-row">
				<div class="result-badge" class:badge-pass={currentDecision} class:badge-fail={!currentDecision}>
					{currentDecision ? '✓ Valid' : '✗ Invalid'}
				</div>
				<button class="btn btn-ghost" onclick={reset} disabled={submitting}>Reset</button>
			</div>
			{#if !currentDecision}
				<div class="final-level">
					Final level: <strong>{status.finalLevel !== null ? status.finalLevel : '—'}</strong>
				</div>
			{/if}
		{:else}
			<div class="decision-row">
				<button
					class="btn btn-pass"
					onclick={() => decide(true)}
					disabled={submitting || data.tiles.length === 0}
				>
					✓ Level Valid
				</button>
				<button
					class="btn btn-fail"
					onclick={() => decide(false)}
					disabled={submitting || data.tiles.length === 0}
				>
					✗ Level Invalid
				</button>
			</div>
			{#if data.depth === MAX_DEPTH}
				<p class="hint">This is the deepest level ({MAX_DEPTH}).</p>
			{/if}
		{/if}
	</footer>
</div>

<style>
	.viewer {
		display: flex;
		flex-direction: column;
		height: 100%;
		overflow: hidden;
	}

	header {
		padding: 14px 20px 12px;
		border-bottom: 1px solid #2a2d3a;
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-shrink: 0;
		gap: 16px;
	}

	.breadcrumb {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.pair-label {
		font-weight: 700;
		font-size: 1.05rem;
		color: #e8eaf0;
	}

	.sep {
		color: #4b5563;
	}

	.depth-label {
		color: #9ca3af;
		font-size: 0.9rem;
	}

	.tile-count {
		font-size: 0.75rem;
		color: #6b7280;
		background: #1e2130;
		padding: 2px 8px;
		border-radius: 10px;
	}

	.sort-control {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 0.78rem;
		color: #9ca3af;
		cursor: pointer;
		flex-shrink: 0;
	}

	.sort-control input[type='checkbox'] {
		accent-color: #6366f1;
		width: 14px;
		height: 14px;
		cursor: pointer;
	}

	.source-badge {
		font-size: 0.75rem;
		font-weight: 700;
		border-radius: 6px;
		padding: 4px 10px;
		text-decoration: none;
		flex-shrink: 0;
		border: 1px solid;
	}

	.source-standard {
		color: #6ee7b7;
		border-color: #065f46;
		background: #022c22;
	}

	.source-standard:hover {
		background: #064e3b;
		border-color: #10b981;
	}

	.source-smooth {
		color: #fcd34d;
		border-color: #92400e;
		background: #1c1407;
	}

	.source-smooth:hover {
		background: #292007;
		border-color: #d97706;
	}

	.patch-control {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-shrink: 0;
	}

	.patch-label {
		font-size: 0.7rem;
		font-weight: 600;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #6b7280;
	}

	.patch-control input[type='range'] {
		width: 100px;
		accent-color: #6366f1;
		cursor: pointer;
	}

	.patch-value {
		font-size: 0.75rem;
		font-variant-numeric: tabular-nums;
		color: #9ca3af;
		min-width: 28px;
	}

	.depth-nav {
		display: flex;
		gap: 4px;
	}

	.depth-pip {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 28px;
		height: 28px;
		border-radius: 6px;
		font-size: 0.78rem;
		font-weight: 600;
		text-decoration: none;
		color: #6b7280;
		background: #1a1d27;
		border: 1px solid #2a2d3a;
		transition: background 0.1s, color 0.1s, border-color 0.1s;
	}

	.depth-pip:hover {
		background: #1e2130;
		color: #e8eaf0;
	}

	.depth-pip.current {
		border-color: #6366f1;
		color: #a5b4fc;
		background: #1e2130;
	}

	.depth-pip.passed {
		border-color: #15803d;
		color: #22c55e;
		background: #0d2218;
	}

	.depth-pip.failed {
		border-color: #991b1b;
		color: #ef4444;
		background: #2a0e0e;
	}

	.depth-pip.current.passed {
		border-color: #22c55e;
	}

	.depth-pip.current.failed {
		border-color: #ef4444;
	}

	.empty {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		color: #6b7280;
		font-size: 0.9rem;
	}

	.agg-bar {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 6px 20px;
		border-bottom: 1px solid #2a2d3a;
		background: #13161f;
		flex-shrink: 0;
	}

	.agg-label {
		font-size: 0.7rem;
		font-weight: 600;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #6b7280;
	}

	.agg-value {
		font-size: 0.88rem;
		font-weight: 700;
		font-variant-numeric: tabular-nums;
		color: #9ca3af;
	}

	.agg-value.agg-pos { color: #22c55e; }
	.agg-value.agg-neg { color: #ef4444; }

	.agg-n {
		font-size: 0.7rem;
		color: #4b5563;
	}

	.scroll-wrap {
		flex: 1;
		overflow: auto;
	}

	.scroll-wrap::-webkit-scrollbar {
		width: 6px;
		height: 6px;
	}
	.scroll-wrap::-webkit-scrollbar-thumb {
		background: #2a2d3a;
		border-radius: 3px;
	}

	.tile-grid {
		display: grid;
		grid-template-columns: 48px repeat(4, auto) repeat(4, 80px);
		column-gap: 12px;
		row-gap: 12px;
		padding: 0 20px 20px;
		min-width: max-content;
		align-items: center;
	}

	.col-header {
		font-size: 0.65rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		color: #6b7280;
		text-transform: uppercase;
		padding: 8px 0 4px;
	}

	.sticky-header {
		position: sticky;
		top: 0;
		background: #0f1117;
		z-index: 1;
	}

	.tile-id {
		font-size: 0.65rem;
		color: #4b5563;
		text-align: right;
		cursor: pointer;
		user-select: none;
		padding: 4px 4px 4px 0;
		border-radius: 3px;
		transition: color 0.1s;
	}

	.tile-id:hover {
		color: #9ca3af;
	}

	.tile-id-active {
		color: #a5b4fc;
		font-weight: 700;
	}

	.score-cell-pre {
		height: 180px;
		width: 80px;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.score-cell-pre .value {
		font-size: 0.78rem;
		font-weight: 700;
		color: #000;
		text-shadow: 0 1px 2px rgba(255,255,255,0.4);
		font-variant-numeric: tabular-nums;
	}

	.factor-cell {
		height: 180px;
		width: 80px;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #1a1d27;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 0.78rem;
		font-weight: 700;
		font-variant-numeric: tabular-nums;
		color: #ef4444;
		transition: background 0.3s, color 0.3s;
	}

	.factor-cell.factor-positive {
		color: #22c55e;
	}

	.factor-placeholder {
		color: #4b5563;
		font-weight: 400;
	}

	footer {
		border-top: 1px solid #2a2d3a;
		padding: 14px 20px;
		flex-shrink: 0;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.decision-row,
	.result-row {
		display: flex;
		gap: 10px;
		align-items: center;
	}

	.btn {
		padding: 9px 22px;
		border: none;
		border-radius: 7px;
		font-size: 0.88rem;
		font-weight: 600;
		cursor: pointer;
		transition: opacity 0.15s, filter 0.15s;
	}

	.btn:disabled {
		opacity: 0.45;
		cursor: default;
	}

	.btn:not(:disabled):hover {
		filter: brightness(1.15);
	}

	.btn-pass {
		background: #166534;
		color: #bbf7d0;
	}

	.btn-fail {
		background: #7f1d1d;
		color: #fecaca;
	}

	.auto-disp-controls {
		display: flex;
		flex-direction: column;
		gap: 4px;
		flex-shrink: 0;
	}

	.auto-disp-control {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.auto-disp-badge {
		font-size: 0.75rem;
		font-weight: 700;
		font-variant-numeric: tabular-nums;
		color: #a5b4fc;
		background: #1e2130;
		border: 1px solid #4338ca;
		border-radius: 6px;
		padding: 4px 10px;
	}

	.done-badge {
		color: #86efac;
		background: #0d2218;
		border-color: #15803d;
	}

	.btn-polling {
		border-color: #d97706;
		color: #fcd34d;
		background: #1c1407;
	}

	.align-cmd {
		font-family: ui-monospace, monospace;
		font-size: 0.7rem;
		color: #93c5fd;
		background: #0d1a2e;
		border: 1px solid #1d4ed8;
		border-radius: 5px;
		padding: 4px 8px;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		max-width: 340px;
	}

	.btn-sm {
		padding: 4px 8px;
		font-size: 0.8rem;
	}

	.btn-auto {
		background: #1e3a5f;
		color: #93c5fd;
		border: 1px solid #1d4ed8;
		padding: 6px 14px;
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
	}

	.btn-ghost {
		background: #1e2130;
		color: #9ca3af;
		border: 1px solid #2a2d3a;
		padding: 7px 14px;
		font-size: 0.8rem;
	}

	.result-badge {
		padding: 6px 16px;
		border-radius: 6px;
		font-weight: 700;
		font-size: 0.88rem;
	}

	.badge-pass {
		background: #0d2218;
		color: #22c55e;
		border: 1px solid #15803d;
	}

	.badge-fail {
		background: #2a0e0e;
		color: #ef4444;
		border: 1px solid #991b1b;
	}

	.final-level {
		font-size: 0.82rem;
		color: #9ca3af;
	}

	.final-level strong {
		color: #e8eaf0;
	}

	.hint {
		font-size: 0.78rem;
		color: #6b7280;
	}
</style>
