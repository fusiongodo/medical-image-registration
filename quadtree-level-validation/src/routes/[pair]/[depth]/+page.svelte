<script lang="ts">
	import { goto, invalidateAll } from '$app/navigation';
	import { deriveStatus, nextDepthForPair, MAX_DEPTH, type ValidationStore } from '$lib/types';
	import type { TileMeta } from '$lib/types';
	import OverlayCanvas from '$lib/OverlayCanvas.svelte';
	import NormCanvas from '$lib/NormCanvas.svelte';
	import LnccScore from '$lib/LnccScore.svelte';

	let {
		data
	} = $props<{
		data: {
			pairId: number;
			depth: number;
			tiles: TileMeta[];
			validation: ValidationStore;
		};
	}>();

	let submitting = $state(false);
	let patchSize = $state(11);
	let sortByScore = $state(false);
	let scores = $state<Map<string, number>>(new Map());
	let cachedScores = $state<Record<string, { lncc?: number; sq?: number }>>({});
	let pendingEntries: Record<string, { lncc?: number; sq?: number }> = {};
	let saveTimer: ReturnType<typeof setTimeout> | null = null;

	// Fetch cached scores — reads only pair/depth/patchSize, never scores
	$effect(() => {
		const pair = data.pairId, depth = data.depth, ps = patchSize;
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
		const next = new Map<string, number>();
		for (const [tile, v] of Object.entries(cached)) {
			if (v.sq !== undefined) next.set(tile, v.sq);
		}
		scores = next;
	});

	function recordScore(tile: string, s: number) {
		scores = new Map(scores.set(tile, s));
	}

	function queueScore(tile: string, type: 'lncc' | 'sq', value: number) {
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
			body: JSON.stringify({ pair_id: data.pairId, depth: data.depth, patchSize, entries })
		});
	}

	const sortedTiles = $derived.by(() => {
		if (!sortByScore) return data.tiles;
		const scored = data.tiles
			.filter((t: TileMeta) => scores.has(t.tile))
			.sort((a: TileMeta, b: TileMeta) => (scores.get(b.tile) ?? 0) - (scores.get(a.tile) ?? 0));
		const unscored = data.tiles.filter((t: TileMeta) => !scores.has(t.tile));
		return [...scored, ...unscored];
	});

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
			<input type="checkbox" bind:checked={sortByScore} />
			<span>Sort by LNCC²</span>
		</label>

		<label class="patch-control">
			<span class="patch-label">Patch</span>
			<input type="range" min="3" max="51" step="2" bind:value={patchSize} />
			<span class="patch-value">{patchSize}px</span>
		</label>

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

	{#if data.tiles.length === 0}
		<div class="empty">No tiles found for this pair / depth.</div>
	{:else}
		<div class="scroll-wrap">
			<div class="tile-grid">
				<span class="col-header sticky-header"></span>
				<span class="col-header sticky-header">HE norm</span>
				<span class="col-header sticky-header">IHC norm</span>
				<span class="col-header sticky-header">Overlay</span>
				<span class="col-header sticky-header">Sobel overlay</span>
				<span class="col-header sticky-header">LNCC</span>
				<span class="col-header sticky-header">LNCC²</span>

				{#each sortedTiles as t (`${data.depth}-${t.tile}`)}
					{@const heSrc = `/api/image?path=${encodeURIComponent(t.he)}`}
					{@const ihcSrc = `/api/image?path=${encodeURIComponent(t.ihc)}`}
					<span class="tile-id">{t.tile}</span>
					<NormCanvas src={heSrc} />
					<NormCanvas src={ihcSrc} />
					<OverlayCanvas {heSrc} {ihcSrc} />
					<OverlayCanvas {heSrc} {ihcSrc} edges={true} />
					<LnccScore {heSrc} {ihcSrc} {patchSize}
						cachedScore={cachedScores[t.tile]?.lncc}
						onscore={(s) => queueScore(t.tile, 'lncc', s)} />
					<LnccScore {heSrc} {ihcSrc} {patchSize} squared={true}
						cachedScore={cachedScores[t.tile]?.sq}
						onscore={(s) => { recordScore(t.tile, s); queueScore(t.tile, 'sq', s); }} />
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
		grid-template-columns: 48px repeat(4, auto) 80px 80px;
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
