<script lang="ts">
	import { goto, invalidateAll } from '$app/navigation';
	import { deriveStatus, nextDepthForPair, MAX_DEPTH, type ValidationStore } from '$lib/types';
	import type { TileMeta } from '$lib/types';
	import OverlayCanvas from '$lib/OverlayCanvas.svelte';
	import PointCanvas from '$lib/PointCanvas.svelte';
	import DisplacedOverlay from '$lib/DisplacedOverlay.svelte';
	import LnccDistributionPanel from '$lib/LnccDistributionPanel.svelte';
	import C2fPanel from '$lib/C2fPanel.svelte';

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
	const PATCH_SIZES = [5, 10, 20, 30, 40, 50] as const;
	let patchSize = $state<5 | 10 | 20 | 30 | 40 | 50>(50);
	let sortByScore = $state(false);
	let sortByFactor = $state(false);
	interface PatchEntry { lncc2: number; lncc2_auto: number; factor_auto: number; }
	interface TileMetrics { delta_px: number; dx: number; dy: number; by_patch: Record<string, PatchEntry>; }
	let tileMetrics = $state<Map<string, TileMetrics>>(new Map());

	let tileKeypoints = $state<Map<string, number[][]>>(new Map());
	let showKeypoints = $state(false);
	let overlayEmphasis = $state<'he' | 'ihc' | null>(null);

	function toggleEmphasis(side: 'he' | 'ihc') {
		overlayEmphasis = overlayEmphasis === side ? null : side;
	}

	$effect(() => {
		function onKeyDown(e: KeyboardEvent) {
			if (!e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
			if (e.key === 'Q' || e.key === 'q') {
				e.preventDefault();
				overlayEmphasis = overlayEmphasis === 'he' ? null : 'he';
			} else if (e.key === 'W' || e.key === 'w') {
				e.preventDefault();
				overlayEmphasis = overlayEmphasis === 'ihc' ? null : 'ihc';
			}
		}
		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	});

	// ── Coarse-to-fine refinement (Phase 2b) ────────────────────────────────
	let refineMode = $state(false);
	let annotationVersion = $state(0);
	let busyTile = $state<string | null>(null);

	interface C2fCandidate { u: number; v: number; psr: number; }
	let c2fCandidates = $state<Map<string, C2fCandidate>>(new Map());

	interface RegAnnotation { type: 'approve' | 'correct' | 'exclude'; u: number; v: number; }
	let regAnnotations = $state<Map<string, RegAnnotation>>(new Map());

	const REFINE_LEVELS = [3, 4, 5] as const;
	const PREV_COMPLETION_THRESHOLD = 1.0;

	function seedStride(depth: number): number {
		// keep the refine set roughly constant at ~16 seeds per level
		return Math.pow(2, depth - 2); // level 3 -> 2, 4 -> 4, 5 -> 8
	}

	function isSeed(tile: string, depth: number): boolean {
		const [xi, yi] = tile.split('_').map((n) => parseInt(n, 10));
		const stride = seedStride(depth);
		return Number.isFinite(xi) && Number.isFinite(yi) && xi % stride === 0 && yi % stride === 0;
	}

	const seedTiles = $derived(data.tiles.filter((t: TileMeta) => isSeed(t.tile, data.depth)));
	const seedLocs = $derived(seedTiles.map((t: TileMeta) => t.tile));
	const seedDone = $derived(seedTiles.filter((t: TileMeta) => regAnnotations.has(t.tile)).length);

	$effect(() => {
		try {
			localStorage.setItem('mvrLastPair', String(data.pairId));
			localStorage.setItem('mvrLastDepth', String(data.depth));
		} catch {
			/* ignore storage errors */ }
	});

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

	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		let stale = false;
		fetch(`/api/tile-metrics?pair=${pair}&depth=${depth}`)
			.then((r) => r.json())
			.then((fetched: Record<string, TileMetrics>) => {
				if (stale) return;
				tileMetrics = new Map(Object.entries(fetched));
			});
		return () => { stale = true; tileMetrics = new Map(); };
	});

	const sortedTiles = $derived.by(() => {
		const factorOf = (tile: string): number | undefined => {
			const m = tileMetrics.get(tile);
			if (!m) return undefined;
			const entry = m.by_patch[String(patchSize)];
			return entry ? entry.factor_auto : undefined;
		};
		if (sortByFactor) {
			const factored = data.tiles
				.filter((t: TileMeta) => factorOf(t.tile) !== undefined)
				.sort((a: TileMeta, b: TileMeta) => (factorOf(b.tile) ?? 0) - (factorOf(a.tile) ?? 0));
			const rest = data.tiles.filter((t: TileMeta) => factorOf(t.tile) === undefined);
			return [...factored, ...rest];
		}
		if (!sortByScore) return data.tiles;
		const lncc2Of = (tile: string): number | undefined => {
			const entry = tileMetrics.get(tile)?.by_patch[String(patchSize)];
			return entry?.lncc2_auto;
		};
		const scored = data.tiles
			.filter((t: TileMeta) => lncc2Of(t.tile) !== undefined)
			.sort((a: TileMeta, b: TileMeta) => (lncc2Of(b.tile) ?? 0) - (lncc2Of(a.tile) ?? 0));
		const unscored = data.tiles.filter((t: TileMeta) => lncc2Of(t.tile) === undefined);
		return [...scored, ...unscored];
	});

	interface Point { x: number; y: number; }
	interface TileAnnotation { hePoints: Point[]; ihcPoints: Point[]; }

	let activeRow = $state<string | null>(null);
	let displayOrder = $state<TileMeta[]>([]);

	const baseOrder = $derived(refineMode ? seedTiles : sortedTiles);

	$effect(() => {
		if (activeRow === null) displayOrder = baseOrder;
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

	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		let stale = false;
		fetch(`/api/c2f/candidates?pair=${pair}&depth=${depth}`)
			.then((r) => r.json())
			.then((d: { cached?: boolean; candidates?: { tile_loc: string; u: number; v: number; psr: number }[] }) => {
				if (stale) return;
				const m = new Map<string, C2fCandidate>();
				if (d.cached && Array.isArray(d.candidates)) {
					for (const c of d.candidates) m.set(c.tile_loc, { u: c.u, v: c.v, psr: c.psr });
				}
				c2fCandidates = m;
			});
		return () => { stale = true; c2fCandidates = new Map(); };
	});

	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		annotationVersion; // re-fetch after each human action
		let stale = false;
		fetch(`/api/c2f/annotate?pair=${pair}&level=${depth}`)
			.then((r) => r.json())
			.then((fetched: Record<string, RegAnnotation>) => {
				if (stale) return;
				regAnnotations = new Map(Object.entries(fetched));
			});
		return () => { stale = true; };
	});

	// ── Previous-level state for refine gating ───────────────────────────────
	let prevTiles = $state<TileMeta[]>([]);
	let prevRegAnnotations = $state<Map<string, RegAnnotation>>(new Map());

	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		if (depth <= 3) { prevTiles = []; return; }
		let stale = false;
		fetch(`/api/tiles/${pair}/${depth - 1}`)
			.then((r) => r.json())
			.then((fetched: TileMeta[]) => {
				if (stale) return;
				prevTiles = fetched;
			});
		return () => { stale = true; prevTiles = []; };
	});

	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		if (depth <= 3) { prevRegAnnotations = new Map(); return; }
		let stale = false;
		fetch(`/api/c2f/annotate?pair=${pair}&level=${depth - 1}`)
			.then((r) => r.json())
			.then((fetched: Record<string, RegAnnotation>) => {
				if (stale) return;
				prevRegAnnotations = new Map(Object.entries(fetched));
			});
		return () => { stale = true; prevRegAnnotations = new Map(); };
	});

	const prevSeedTiles = $derived(
		data.depth > 3 ? prevTiles.filter((t: TileMeta) => isSeed(t.tile, data.depth - 1)) : []
	);
	const prevSeedDone = $derived(
		prevSeedTiles.filter((t: TileMeta) => prevRegAnnotations.has(t.tile)).length
	);
	const prevSeedCompletion = $derived(
		prevSeedTiles.length > 0 ? prevSeedDone / prevSeedTiles.length : 0
	);
	const canRefine = $derived(
		REFINE_LEVELS.includes(data.depth) &&
		(data.depth === 3 || prevSeedCompletion >= PREV_COMPLETION_THRESHOLD)
	);

	$effect(() => {
		if (!canRefine && refineMode) {
			refineMode = false;
			activeRow = null;
		}
	});

	// Persist refine-mode toggle across browser sessions
	let refineSaveReady = $state(false);
	$effect(() => {
		// Skip the initial run so we don't overwrite the stored value with the
		// default `false` before the restore effect has a chance to read it.
		if (!refineSaveReady) {
			refineSaveReady = true;
			return;
		}
		try {
			localStorage.setItem('mvrLastRefine', refineMode ? '1' : '0');
		} catch {
			/* ignore storage errors */ }
	});

	let restoredRefineFor = $state<{ pair: number; depth: number } | null>(null);
	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		// Only restore once we know refinement is allowed. If it is still locked
		// (e.g. waiting for previous-level data), keep restoredRefineFor unset so
		// the effect re-runs and restores as soon as canRefine becomes true.
		if (!canRefine) return;

		if (restoredRefineFor?.pair === pair && restoredRefineFor?.depth === depth) return;
		restoredRefineFor = { pair, depth };

		let stored = false;
		try {
			stored = localStorage.getItem('mvrLastRefine') === '1';
		} catch {
			/* ignore storage errors */ }

		if (stored) {
			refineMode = true;
		}
	});

	async function postAnnotate(tile: string, action: 'approve' | 'correct' | 'exclude' | 'clear', u = 0, v = 0) {
		busyTile = tile;
		try {
			const payload: Record<string, unknown> = { pair_id: data.pairId, level: data.depth, tile_loc: tile, action };
			if (action !== 'clear') {
				payload.u = u;
				payload.v = v;
			}
			await fetch('/api/c2f/annotate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload)
			});
			annotationVersion++;
		} finally {
			busyTile = null;
		}
	}

	function approveTile(tile: string) {
		const c = c2fCandidates.get(tile);
		if (!c) return;
		postAnnotate(tile, 'approve', c.u, c.v);
	}

	function approveTileUv(tile: string, u: number, v: number) {
		postAnnotate(tile, 'approve', u, v);
	}

	function correctTile(tile: string) {
		const ann = annotations[tile];
		if (!ann || ann.hePoints.length < 1 || ann.ihcPoints.length < 1) return;
		const he = ann.hePoints[ann.hePoints.length - 1];
		const ihc = ann.ihcPoints[ann.ihcPoints.length - 1];
		postAnnotate(tile, 'correct', he.x - ihc.x, he.y - ihc.y);
	}

	function excludeTile(tile: string) {
		postAnnotate(tile, 'exclude', 0, 0);
	}

	function clearTile(tile: string) {
		postAnnotate(tile, 'clear');
	}

	function manualDisplacement(tile: string): { dx: number; dy: number } | null {
		const ann = annotations[tile];
		if (!ann) return null;
		const pairs = Math.min(ann.hePoints.length, ann.ihcPoints.length);
		if (pairs === 0) return null;
		let dx = 0, dy = 0;
		for (let i = 0; i < pairs; i++) {
			dx += ann.hePoints[i].x - ann.ihcPoints[i].x;
			dy += ann.hePoints[i].y - ann.ihcPoints[i].y;
		}
		return { dx: dx / pairs, dy: dy / pairs };
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

	// ── Alignment job (run from header) ──────────────────────────────────────
	interface AlignJobState {
		running: boolean;
		step: string;
		error: string | null;
		finishedAt: number | null;
	}
	let alignJob = $state<AlignJobState | null>(null);
	let alignJobLocked = $state(false);
	let alignJobReason = $state<string | null>(null);

	$effect(() => {
		const pair = data.pairId, depth = data.depth;
		let stale = false;
		let lastFinishedAt: number | null = null;
		function sameState(a: AlignJobState | null, b: AlignJobState | null): boolean {
			if (a === b) return true;
			if (!a || !b) return false;
			return a.running === b.running && a.step === b.step && a.error === b.error && a.finishedAt === b.finishedAt;
		}
		function poll() {
			fetch(`/api/run-alignment?pair=${pair}&depth=${depth}`)
				.then((r) => r.json())
				.then((d: { state: AlignJobState | null; locked: boolean; reason: string | null }) => {
					if (stale) return;
					alignJobLocked = d.locked;
					alignJobReason = d.reason;

					if (!sameState(alignJob, d.state)) {
						alignJob = d.state;
					}

					const finishedAt = d.state?.finishedAt ?? null;
					if (finishedAt !== null && finishedAt !== lastFinishedAt) {
						lastFinishedAt = finishedAt;
						autoDispRefreshKey++;
						tileMetrics = new Map();
						fetch(`/api/tile-metrics?pair=${pair}&depth=${depth}`)
							.then((r) => r.json())
							.then((fetched: Record<string, TileMetrics>) => {
								if (stale) return;
								tileMetrics = new Map(Object.entries(fetched));
							});
					}
				});
		}
		poll();
		const interval = setInterval(poll, 2000);
		return () => {
			stale = true;
			clearInterval(interval);
			alignJob = null;
			alignJobLocked = false;
			alignJobReason = null;
		};
	});

	async function runAlignment() {
		const res = await fetch('/api/run-alignment', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ pair_id: data.pairId, depth: data.depth })
		});
		const d = await res.json();
		if (d.locked) {
			alignJobLocked = true;
			alignJobReason = d.reason;
		} else {
			alignJob = d.state ?? null;
		}
	}

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

<div class="scrollable">

<LnccDistributionPanel pairId={data.pairId} depth={data.depth} {patchSize} />
<C2fPanel pairId={data.pairId} depth={data.depth} {annotationVersion} seed={seedLocs} {tileMetrics} {patchSize} emphasis={overlayEmphasis} onToggleEmphasis={toggleEmphasis} onApprove={approveTileUv} onExclude={excludeTile} onClear={clearTile} />

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
		{#if REFINE_LEVELS.includes(data.depth)}
			<label
				class="sort-control refine-toggle"
				class:refine-locked={!canRefine}
				title={!canRefine
					? `Complete level ${data.depth - 1} refine set first (${prevSeedDone}/${prevSeedTiles.length})`
					: 'Toggle refine set'}
			>
				<input
					type="checkbox"
					bind:checked={refineMode}
					onchange={() => { activeRow = null; }}
					disabled={!canRefine}
				/>
				<span>Refine set{#if refineMode} · {seedDone}/{seedTiles.length}{/if}</span>
			</label>
			{#if !canRefine && data.depth > 3}
				<span class="refine-lock-hint">
					🔒 level {data.depth - 1} incomplete ({prevSeedDone}/{prevSeedTiles.length})
				</span>
			{/if}
		{:else}
			<span class="sort-control refine-disabled" title="Refinement available at levels 3–5">
				Refine set · n/a
			</span>
		{/if}
		{#if data.smooth}
			<a class="source-badge source-smooth" href="/{data.pairId}/{data.depth}">Standard</a>
		{:else if data.smoothAvailable}
			<a class="source-badge source-standard" href="/{data.pairId}/{data.depth}?source=smooth">Smooth IHC</a>
		{/if}

		<div class="patch-control">
			<span class="patch-label">Patch</span>
			<div class="patch-btns">
				{#each PATCH_SIZES as ps}
					<button
						class="patch-btn"
						class:active={patchSize === ps}
						onclick={() => { patchSize = ps; }}
					>{ps}px</button>
				{/each}
			</div>
		</div>

		<div class="auto-disp-controls">
			<div class="auto-disp-control">
				{#if alignJob?.running}
					<span class="auto-disp-badge running-badge" title={alignJob.step}>
						⏳ {alignJob.step}
					</span>
				{:else if alignJob?.error}
					<span class="auto-disp-badge error-badge" title={alignJob.error}>
						✗ failed
					</span>
				{:else if autoDisps.size === 0}
					<span class="auto-disp-badge pending-badge">FFT alignment not computed</span>
				{:else}
					<span class="auto-disp-badge" class:done-badge={autoDisps.size === data.tiles.length}>
						{autoDisps.size} / {data.tiles.length}{autoDisps.size === data.tiles.length ? ' ✓' : ''}
					</span>
				{/if}

				<button
					class="btn btn-auto"
					onclick={runAlignment}
					disabled={alignJobLocked || alignJob?.running || autoDisps.size === data.tiles.length}
					title={alignJobReason ?? 'Run FFT alignment for this level'}
				>
					{alignJob?.running ? 'Running…' : '▶ Run alignment'}
				</button>

				{#if alignJobLocked}
					<span class="refine-lock-hint">{alignJobReason}</span>
				{/if}
			</div>

			{#if autoDisps.size > 0}
				<div class="auto-disp-control">
					<code class="align-cmd">{alignCommand}</code>
					<button class="btn btn-ghost btn-sm" onclick={() => navigator.clipboard.writeText(alignCommand)} title="Copy top-5 command">⎘</button>
					<button class="btn btn-auto" onclick={() => { autoDispRefreshKey++; }}>↺ Refresh</button>
					<button
						class="btn btn-auto"
						class:btn-polling={pollingActive}
						onclick={() => { pollingActive = !pollingActive; if (pollingActive) autoDispRefreshKey++; }}>
						{pollingActive ? '⏹ Stop' : '⟳ Watch'}
					</button>
				</div>
			{/if}
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
		<div
			class="tile-grid"
				style:grid-template-columns={refineMode
					? '48px repeat(4, auto) repeat(4, 80px) 168px'
					: '48px repeat(4, auto) repeat(4, 80px)'}
			>
			<span class="col-header sticky-header"></span>
			<span class="col-header sticky-header">HE norm</span>
			<span class="col-header sticky-header">{data.smooth ? 'IHC smooth' : 'IHC norm'}</span>
			<span class="col-header sticky-header">Overlay</span>
			<span class="col-header sticky-header col-header-flex">
				{data.smooth ? 'Pre-aligned' : 'Auto overlay'}
				<span class="emph-pills">
					<button
						class="emph-pill"
						class:active={overlayEmphasis === 'he'}
						onclick={() => toggleEmphasis('he')}
						title="Highlight fixed HE (Shift+Q)"
					>HE</button>
					<button
						class="emph-pill"
						class:active={overlayEmphasis === 'ihc'}
						onclick={() => toggleEmphasis('ihc')}
						title="Highlight moving IHC (Shift+W)"
					>IHC</button>
				</span>
			</span>
			<span class="col-header sticky-header">LNCC²</span>
			<span class="col-header sticky-header">|Δ| auto</span>
			<span class="col-header sticky-header">LNCC² auto</span>
			<span class="col-header sticky-header">Factor auto</span>
			{#if refineMode}<span class="col-header sticky-header">Refine</span>{/if}

	{#each displayOrder as t (`${data.pairId}-${data.depth}-${t.tile}`)}
			{@const heSrc  = `/api/image?path=${encodeURIComponent(t.he)}`}
			{@const ihcSrc = `/api/image?path=${encodeURIComponent(t.ihc)}`}
			{@const isActive = activeRow === t.tile}
			{@const ann = annotations[t.tile] ?? { hePoints: [], ihcPoints: [] }}
		{@const tileAutoDisp = autoDisps.get(t.tile)}
	{@const m = tileMetrics.get(t.tile)}
	{@const entry = m?.by_patch[String(patchSize)]}
	{@const tileKps = showKeypoints ? (tileKeypoints.get(t.tile) ?? []) : []}
		<span
			class="tile-id"
			class:tile-id-active={isActive}
			onclick={() => { activeRow = isActive ? null : t.tile; }}
			role="button"
			tabindex="0"
			onkeydown={(e) => e.key === 'Enter' && (activeRow = isActive ? null : t.tile)}
		>{t.tile}</span>
		<PointCanvas src={heSrc} active={isActive || refineMode} points={ann.hePoints}
			keypoints={tileKps}
			onpoint={(x, y) => addPoint(t.tile, 'he', x, y)} />
		<PointCanvas src={ihcSrc} active={isActive || refineMode} points={ann.ihcPoints}
			onpoint={(x, y) => addPoint(t.tile, 'ihc', x, y)} />
		<OverlayCanvas {heSrc} {ihcSrc} />
		{@const manual = manualDisplacement(t.tile)}
		{@const corrDisp = regAnnotations.get(t.tile)}
		{@const overlayDx = data.smooth
			? 0
			: corrDisp && corrDisp.type === 'correct'
				? corrDisp.u
				: manual
					? manual.dx
					: tileAutoDisp !== undefined
						? tileAutoDisp.dx
						: 0}
		{@const overlayDy = data.smooth
			? 0
			: corrDisp && corrDisp.type === 'correct'
				? corrDisp.v
				: manual
					? manual.dy
					: tileAutoDisp !== undefined
						? tileAutoDisp.dy
						: 0}
		{#if data.smooth || corrDisp?.type === 'correct' || manual || tileAutoDisp !== undefined}
			<DisplacedOverlay {heSrc} {ihcSrc} dx={overlayDx} dy={overlayDy}
				keypoints={tileKps} emphasis={overlayEmphasis} />
		{:else}
			<div class="factor-cell align-pending-cell" title="Run FFT alignment to enable this overlay">
				<span class="align-pending">Align<br>pending</span>
			</div>
		{/if}
			{#if entry}
				<div class="score-cell-pre" style:background={lnccColor(entry.lncc2)}>
					<span class="value">{entry.lncc2.toFixed(3)}</span>
				</div>
			{:else}
				<div class="score-cell-pre"><span class="factor-placeholder">…</span></div>
			{/if}
			{@const arrowDisp = data.smooth
				? (tileAutoDisp !== undefined
					? tileAutoDisp
					: m && (m.dx !== 0 || m.dy !== 0)
						? { dx: m!.dx, dy: m!.dy }
						: null)
				: (corrDisp?.type === 'correct' || manual || tileAutoDisp !== undefined
					? { dx: overlayDx, dy: overlayDy }
					: null)}
			{#if arrowDisp}
			{@const dvx = arrowDisp.dx}
			{@const dvy = arrowDisp.dy}
			{@const mag = Math.sqrt(dvx ** 2 + dvy ** 2)}
			{@const MAX_LEN = 28}
			{@const scale = mag > 0 ? Math.min(MAX_LEN, mag) / mag : 0}
			{@const ax = 40 + dvx * scale}
			{@const ay = 76 + dvy * scale}
			<div class="factor-cell disp-cell">
				<svg width="80" height="100" class="disp-svg">
					<defs>
						<marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
							<path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
						</marker>
					</defs>
					<circle cx="40" cy="76" r="2.5" fill="#9ca3af" />
					{#if mag > 0.1}
						<line x1="40" y1="76" x2={ax} y2={ay}
							stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)" />
					{/if}
					<text x="40" y="96" text-anchor="middle" class="disp-label">{mag.toFixed(1)}px</text>
				</svg>
			</div>
		{:else}
			<div class="factor-cell disp-cell">
				<span class="factor-placeholder">…</span>
			</div>
		{/if}
			{#if entry}
				<div class="score-cell-pre" style:background={lnccColor(entry.lncc2_auto)}>
					<span class="value">{entry.lncc2_auto.toFixed(3)}</span>
				</div>
			{:else}
				<div class="score-cell-pre"><span class="factor-placeholder">…</span></div>
			{/if}
			<div class="factor-cell" class:factor-positive={entry !== undefined && entry.factor_auto > 1}>
				{#if entry}
					{entry.factor_auto.toFixed(3)}
				{:else}
					<span class="factor-placeholder">…</span>
				{/if}
			</div>
			{#if refineMode}
				{@const cand = c2fCandidates.get(t.tile)}
				{@const reg = regAnnotations.get(t.tile)}
				{@const canCorrect = ann.hePoints.length >= 1 && ann.ihcPoints.length >= 1}
				{@const busy = busyTile === t.tile}
				<div class="refine-cell">
					{#if reg}
						<span class="refine-status"
							class:st-approve={reg.type === 'approve'}
							class:st-correct={reg.type === 'correct'}
							class:st-exclude={reg.type === 'exclude'}>
							{reg.type === 'approve'
								? '✓ approved'
								: reg.type === 'correct'
									? '✎ corrected'
									: '⊘ excluded'}
						</span>
						{#if reg.type !== 'exclude'}
							<span class="refine-disp">Δ {Math.hypot(reg.u, reg.v).toFixed(1)}px</span>
						{/if}
						<button class="refine-btn ghost" onclick={() => clearTile(t.tile)} disabled={busy}>Clear</button>
					{:else}
						{#if cand}
							<button class="refine-btn approve" onclick={() => approveTile(t.tile)} disabled={busy}>
								Approve <span class="mini">{Math.hypot(cand.u, cand.v).toFixed(1)}px</span>
							</button>
						{:else}
							<span class="refine-hint">compute candidates</span>
						{/if}
						<button
							class="refine-btn correct"
							onclick={() => correctTile(t.tile)}
							disabled={!canCorrect || busy}
							title={canCorrect ? 'Use the placed HE/IHC point pair' : 'Click a landmark on HE and its match on IHC'}
						>Correct</button>
						<button
							class="refine-btn exclude"
							onclick={() => excludeTile(t.tile)}
							disabled={busy}
							title="Ignore this tile in the fit"
						>Exclude</button>
						{#if !canCorrect}
							<span class="refine-hint">click HE + IHC point</span>
						{/if}
					{/if}
				</div>
			{/if}
		{/each}
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
</div>

<style>
	.scrollable {
		flex: 1;
		overflow: auto;
	}

	.scrollable::-webkit-scrollbar {
		width: 12px;
		height: 12px;
	}
	.scrollable::-webkit-scrollbar-track {
		background: transparent;
	}
	.scrollable::-webkit-scrollbar-thumb {
		background: #3f4454;
		border-radius: 6px;
		border: 2px solid #0f1117;
	}
	.scrollable::-webkit-scrollbar-thumb:hover {
		background: #5b6277;
	}

	.viewer {
		display: flex;
		flex-direction: column;
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

	.patch-btns {
		display: flex;
		gap: 2px;
	}

	.patch-btn {
		all: unset;
		cursor: pointer;
		padding: 2px 7px;
		border-radius: 3px;
		font-size: 0.7rem;
		font-variant-numeric: tabular-nums;
		color: #6b7280;
		background: transparent;
		border: 1px solid transparent;
		transition: color 0.12s, background 0.12s;
	}

	.patch-btn:hover { color: #e8eaf0; }

	.patch-btn.active {
		background: #6366f1;
		color: #fff;
		border-color: #6366f1;
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

	.col-header-flex {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.emph-pills {
		display: inline-flex;
		gap: 2px;
	}

	.emph-pill {
		all: unset;
		cursor: pointer;
		font-size: 0.6rem;
		font-weight: 700;
		letter-spacing: 0.06em;
		color: #6b7280;
		background: #1a1d27;
		border: 1px solid #2a2d3a;
		border-radius: 3px;
		padding: 1px 5px;
	}

	.emph-pill:hover { color: #e8eaf0; }

	.emph-pill.active {
		color: #fff;
		background: #6366f1;
		border-color: #6366f1;
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

	.align-pending-cell {
		cursor: help;
	}

	.align-pending {
		font-size: 0.68rem;
		font-weight: 600;
		color: #6b7280;
		text-align: center;
		line-height: 1.3;
	}

	.disp-cell {
		padding: 0;
		overflow: visible;
	}

	.disp-svg { display: block; overflow: visible; }

	.disp-label {
		fill: #9ca3af;
		font-size: 10px;
		font-family: system-ui, sans-serif;
		font-variant-numeric: tabular-nums;
	}

	.refine-toggle input[type='checkbox'] { accent-color: #22c55e; }

	.refine-locked {
		opacity: 0.55;
		cursor: not-allowed;
	}

	.refine-locked input[type='checkbox'] {
		cursor: not-allowed;
		accent-color: #6b7280;
	}

	.refine-lock-hint {
		font-size: 0.7rem;
		color: #f87171;
		background: #2a0e0e;
		border: 1px solid #991b1b;
		border-radius: 4px;
		padding: 2px 6px;
		white-space: nowrap;
	}

	.refine-disabled {
		color: #4b5563;
		cursor: help;
	}

	.refine-cell {
		height: 180px;
		width: 168px;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #12151f;
		display: flex;
		flex-direction: column;
		align-items: stretch;
		justify-content: center;
		gap: 6px;
		padding: 8px;
	}

	.refine-status {
		font-size: 0.72rem;
		font-weight: 700;
		text-align: center;
		border-radius: 4px;
		padding: 3px 6px;
	}

	.refine-status.st-approve { color: #22c55e; background: #0d2218; border: 1px solid #15803d; }
	.refine-status.st-correct { color: #a5b4fc; background: #171a2e; border: 1px solid #4338ca; }
	.refine-status.st-exclude { color: #9ca3af; background: #1e1e22; border: 1px solid #4b5563; }

	.refine-disp {
		font-size: 0.72rem;
		color: #9ca3af;
		text-align: center;
		font-variant-numeric: tabular-nums;
	}

	.refine-btn {
		all: unset;
		cursor: pointer;
		text-align: center;
		font-size: 0.72rem;
		font-weight: 600;
		padding: 6px 8px;
		border-radius: 4px;
		border: 1px solid transparent;
	}

	.refine-btn.approve { background: #166534; color: #bbf7d0; }
	.refine-btn.correct { background: #1e2130; color: #a5b4fc; border-color: #4338ca; }
	.refine-btn.exclude { background: #374151; color: #e5e7eb; border-color: #4b5563; }
	.refine-btn.ghost   { background: #1e2130; color: #9ca3af; border-color: #2a2d3a; }
	.refine-btn:not(:disabled):hover { filter: brightness(1.15); }
	.refine-btn:disabled { opacity: 0.4; cursor: default; }

	.refine-btn .mini {
		font-weight: 400;
		font-variant-numeric: tabular-nums;
		opacity: 0.85;
	}

	.refine-hint {
		font-size: 0.62rem;
		color: #6b7280;
		text-align: center;
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

	.running-badge {
		color: #fcd34d;
		background: #1c1407;
		border-color: #d97706;
	}

	.error-badge {
		color: #f87171;
		background: #2a0e0e;
		border-color: #991b1b;
		cursor: help;
	}

	.pending-badge {
		color: #9ca3af;
		background: #1a1d27;
		border-color: #4b5563;
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
