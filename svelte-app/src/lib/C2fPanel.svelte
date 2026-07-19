<script lang="ts">
	import C2fHeatmap from '$lib/C2fHeatmap.svelte';
	import C2fVectorField from '$lib/C2fVectorField.svelte';
	import DisplacedOverlay from '$lib/DisplacedOverlay.svelte';
	import DisplacementArrow from '$lib/DisplacementArrow.svelte';
	import OverlayCanvas from '$lib/OverlayCanvas.svelte';
	import { computeLNCC } from '$lib/imageUtils';

	let {
		pairId,
		depth,
		annotationVersion = 0,
		seed = [],
		tileMetrics = new Map(),
		patchSize = 50,
		emphasis = null,
		onToggleEmphasis,
		onApprove,
		onExclude,
		onClear,
		onReload,
		onComputed,
		onFlash
	}: {
		pairId: number;
		depth: number;
		annotationVersion?: number;
		seed?: string[];
		tileMetrics?: Map<string, TileMetrics>;
		patchSize?: number;
		emphasis?: 'he' | 'ihc' | null;
		onToggleEmphasis?: (side: 'he' | 'ihc') => void;
		onApprove?: (tile: string, u: number, v: number) => void;
		onExclude?: (tile: string) => void;
		onClear?: (tile: string) => void;
		onReload?: () => void;
		onComputed?: () => void;
		onFlash?: (msg: string, kind?: 'ok' | 'warn' | 'err') => void;
	} = $props();

	type Rating = 'bad' | 'ok' | 'good';

	interface FieldSet {
		id: string;
		name: string;
		saved_depth: number | null;
		tau?: number;
		n_human?: number;
		updated: number;
		rating?: Rating | null;
	}

	let sets = $state<FieldSet[]>([]);
	let activeSetId = $state<string | null>(null);
	let mainSetId = $state<string | null>(null);
	let selectedSetId = $state('');
	let setBusy = $state(false);
	let pendingSetId = $state<string | null>(null);

	const activeSet = $derived(sets.find((s) => s.id === activeSetId) ?? null);
	const pendingSet = $derived(sets.find((s) => s.id === pendingSetId) ?? null);
	const selectedSet = $derived(sets.find((s) => s.id === selectedSetId) ?? null);
	const RATINGS: Rating[] = ['bad', 'ok', 'good'];

	interface TileResult {
		tile_loc: string;
		psr: number;
		residual: number;
		kept: boolean;
		excluded?: boolean;
		annotated?: 'approve' | 'correct' | 'exclude' | null;
		dx: number;
		dy: number;
		ux: number;
		uy: number;
		prior_dx: number;
		prior_dy: number;
	}

	interface RefitData {
		tau: number;
		kept: number;
		rejected: number;
		excluded?: number;
		n_human?: number;
		mean_residual: number;
		tiles: TileResult[];
	}

	interface PatchEntry {
		lncc2: number;
		lncc2_auto: number;
		factor_auto: number;
	}

	interface TileMetrics {
		delta_px: number;
		dx: number;
		dy: number;
		by_patch: Record<string, PatchEntry>;
	}

	interface JobState {
		running: boolean;
		done: number;
		total: number;
		error: string | null;
		finishedAt: number | null;
	}

	// tau is exposed on a log slider: tau = 10 ** exp
	const EXP_MIN = -5;
	const EXP_MAX = -2.3; // ~5e-3
	let tauExp = $state(-4); // default tau = 1e-4
	const tau = $derived(Math.pow(10, tauExp));

	// keep-fraction gate: tau derived server-side as the (1 - exclude%) quantile
	// of the auto-tile residuals, so the exclude% directly targets the ratio of
	// tiles dropped from the spline.
	type TauMode = 'tau' | 'keep';
	let tauMode = $state<TauMode>('keep');
	const EXCLUDE_MIN = 0;
	const EXCLUDE_MAX = 50;
	let excludePct = $state(10);
	const keepFraction = $derived(1 - excludePct / 100);

	let open = $state(true);
	let cached = $state<boolean | null>(null);
	let refit = $state<RefitData | null>(null);
	let refitError = $state<string | null>(null);
	const effectiveTau = $derived(tauMode === 'keep' ? refit?.tau ?? null : tau);
	let job = $state<JobState | null>(null);
	let saving = $state(false);
	let savedAt = $state<number | null>(null);
	let hoveredTile = $state<string | null>(null);
	let selectedTile = $state<string | null>(null);

	type VectorMode = 'refinement' | 'prior' | 'result';
	let vectorMode = $state<VectorMode>('refinement');

	const previewTile = $derived(selectedTile ?? hoveredTile);
	const previewResult = $derived(
		previewTile ? refit?.tiles.find((t) => t.tile_loc === previewTile) ?? null : null
	);

	// Field-aware crops: the moving IHC is recropped from the raw WSI at the
	// given offset (baked into the crop, never translated), so large per-level
	// displacements still keep full intersection with the fixed HE tile.
	function liveCropSrc(side: 'he' | 'ihc', dx = 0, dy = 0): string {
		if (!previewTile) return '';
		const [x, y] = previewTile.split('_');
		let u = `/api/live-crop/tile?pair=${pairId}&level=${depth}&x=${x}&y=${y}&side=${side}`;
		if (dx !== 0 || dy !== 0) u += `&dx=${dx}&dy=${dy}`;
		return u;
	}

	const previewHeSrc = $derived(previewTile ? liveCropSrc('he') : '');
	// Base IHC recropped at the previous-level field prediction (prior).
	const previewIhcSrc = $derived(
		previewTile && previewResult
			? liveCropSrc('ihc', previewResult.prior_dx, previewResult.prior_dy)
			: ''
	);
	// Included IHC recropped at the full refined displacement (total).
	const previewIhcIncludedSrc = $derived(
		previewTile && previewResult ? liveCropSrc('ihc', previewResult.ux, previewResult.uy) : ''
	);

	function onSelect(tile: string | null) {
		selectedTile = selectedTile === tile ? null : tile;
	}

	$effect(() => {
		try {
			const stored = localStorage.getItem('mvrC2fOpen');
			if (stored !== null) open = stored === '1';
		} catch {
			/* ignore storage errors */ }
	});

	$effect(() => {
		try {
			localStorage.setItem('mvrC2fOpen', open ? '1' : '0');
		} catch {
			/* ignore storage errors */ }
	});

	$effect(() => {
		try {
			const m = localStorage.getItem('mvrC2fTauMode');
			if (m === 'tau' || m === 'keep') tauMode = m;
			const ex = localStorage.getItem('mvrC2fExclude');
			if (ex !== null && !Number.isNaN(Number(ex))) excludePct = Number(ex);
		} catch {
			/* ignore storage errors */ }
	});

	$effect(() => {
		try {
			localStorage.setItem('mvrC2fTauMode', tauMode);
			localStorage.setItem('mvrC2fExclude', String(excludePct));
		} catch {
			/* ignore storage errors */ }
	});

	let pollTimer: ReturnType<typeof setInterval> | null = null;
	let tauDebounce: ReturnType<typeof setTimeout> | null = null;

	async function checkCache() {
		const p = pairId, d = depth;
		refitError = null;
		try {
			const r = await fetch(`/api/c2f/candidates?pair=${p}&depth=${d}`);
			if (p !== pairId || d !== depth) return; // navigated away mid-flight
			const data = await r.json();
			cached = data.cached === true;
			if (cached) runRefit();
		} catch (err) {
			if (p !== pairId || d !== depth) return;
			refitError = err instanceof Error ? err.message : 'failed to check candidates';
		}
	}

	async function runRefit() {
		if (!cached) return;
		const p = pairId, d = depth;
		refitError = null;
		const q = tauMode === 'keep' ? `keep=${keepFraction}` : `tau=${tau}`;
		try {
			const r = await fetch(`/api/c2f/refit?pair=${p}&depth=${d}&${q}`);
			if (p !== pairId || d !== depth) return; // stale response for a prior pair/depth
			if (r.ok) {
				refit = await r.json();
			} else {
				refitError = (await r.text().catch(() => '')) || `refit failed (${r.status})`;
			}
		} catch (err) {
			if (p !== pairId || d !== depth) return;
			refitError = err instanceof Error ? err.message : 'refit request failed';
		}
	}

	function onTauInput(e: Event) {
		tauExp = (e.target as HTMLInputElement).valueAsNumber;
		if (tauDebounce) clearTimeout(tauDebounce);
		tauDebounce = setTimeout(runRefit, 300);
	}

	function onExcludeInput(e: Event) {
		excludePct = (e.target as HTMLInputElement).valueAsNumber;
		if (tauDebounce) clearTimeout(tauDebounce);
		tauDebounce = setTimeout(runRefit, 300);
	}

	function setTauMode(mode: TauMode) {
		if (tauMode === mode) return;
		tauMode = mode;
		runRefit();
	}

	async function startCompute() {
		const r = await fetch('/api/c2f/candidates', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ pair_id: pairId, depth })
		});
		const data = await r.json();
		job = data.state;
		if (job?.running) startPolling();
	}

	function startPolling() {
		if (pollTimer) return;
		pollTimer = setInterval(async () => {
			const r = await fetch(`/api/c2f/candidates/progress?pair=${pairId}&depth=${depth}`);
			job = await r.json();
			if (!job?.running) {
				clearInterval(pollTimer!);
				pollTimer = null;
				if (!job?.error) {
					cached = true;
					runRefit();
					onComputed?.();
				}
			}
		}, 1000);
	}

	async function saveField() {
		saving = true;
		try {
			const body =
				tauMode === 'keep'
					? { pair_id: pairId, depth, keep: keepFraction }
					: { pair_id: pairId, depth, tau };
			const r = await fetch('/api/c2f/save-field', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body)
			});
			if (r.ok) savedAt = Date.now();
		} finally {
			saving = false;
		}
	}

	async function loadSets() {
		try {
			const r = await fetch(`/api/c2f/field-set?pair=${pairId}`);
			if (!r.ok) return;
			const data = await r.json();
			sets = data.sets ?? [];
			activeSetId = data.active ?? null;
			mainSetId = data.main ?? null;
			selectedSetId = activeSetId ?? (sets[0]?.id ?? '');
		} catch {
			/* ignore */ }
	}

	async function rateSet(rating: Rating) {
		if (!selectedSetId) return;
		const res = await postSet({ action: 'rate', set_id: selectedSetId, rating });
		if (res?.ok) { await loadSets(); onReload?.(); }
	}

	async function pinMain() {
		if (!selectedSetId) return;
		const res = await postSet({ action: 'main', set_id: selectedSetId });
		if (res?.ok) { await loadSets(); onReload?.(); }
	}

	async function postSet(bodyExtra: Record<string, unknown>): Promise<{ ok?: boolean } | null> {
		setBusy = true;
		try {
			const r = await fetch('/api/c2f/field-set', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ pair_id: pairId, ...bodyExtra })
			});
			if (!r.ok) return null;
			return await r.json();
		} catch {
			return null;
		} finally {
			setBusy = false;
		}
	}

	async function reloadWorkspace() {
		cached = null;
		refit = null;
		savedAt = null;
		selectedTile = null;
		await checkCache();
		onReload?.();
	}

	async function saveSet() {
		const active = sets.find((s) => s.id === activeSetId);
		let name = active?.name;
		const setId = active?.id;
		if (!setId) {
			const entered = prompt('Name this field set:');
			if (!entered) return;
			name = entered;
		}
		const res = await postSet({ action: 'save', set_id: setId, name });
		if (res?.ok) await loadSets();
	}

	async function newSet() {
		const entered = prompt('Name the new field set:', 'attempt');
		if (!entered) { selectedSetId = activeSetId ?? ''; return; }
		if (!confirm('Start a new field set? The active annotations and field for this pair will be cleared (existing saved sets are kept).')) {
			selectedSetId = activeSetId ?? '';
			return;
		}
		const res = await postSet({ action: 'new', name: entered });
		if (res?.ok) { await loadSets(); await reloadWorkspace(); }
		else selectedSetId = activeSetId ?? '';
	}

	function onSelectChange(e: Event) {
		const id = (e.target as HTMLSelectElement).value;
		if (id === '__new__') {
			newSet();
			return;
		}
		selectedSetId = id;
		if (id && id !== activeSetId) pendingSetId = id;
	}

	function cancelLoad() {
		pendingSetId = null;
		selectedSetId = activeSetId ?? '';
	}

	$effect(() => {
		if (!pendingSetId) return;
		function onKey(e: KeyboardEvent) {
			if (e.key === 'Escape') cancelLoad();
		}
		window.addEventListener('keydown', onKey);
		return () => window.removeEventListener('keydown', onKey);
	});

	async function saveActive(): Promise<boolean> {
		let name = activeSet?.name;
		if (!name) {
			const entered = prompt('Name the current field set before switching:');
			if (!entered) return false;
			name = entered;
		}
		const res = await postSet({ action: 'save', name });
		return !!res?.ok;
	}

	async function confirmLoad(save: boolean) {
		const target = pendingSetId;
		pendingSetId = null;
		if (!target) return;
		if (save && !(await saveActive())) {
			selectedSetId = activeSetId ?? '';
			return;
		}
		const res = await postSet({ action: 'load', set_id: target });
		if (res?.ok) {
			await loadSets();
			await reloadWorkspace();
		} else {
			selectedSetId = activeSetId ?? '';
		}
	}

	async function renameSelectedSet() {
		if (!selectedSetId) return;
		const s = sets.find((x) => x.id === selectedSetId);
		const entered = prompt('Rename field set:', s?.name ?? '');
		if (!entered) return;
		const res = await postSet({ action: 'rename', set_id: selectedSetId, name: entered });
		if (res?.ok) await loadSets();
	}

	$effect(() => {
		if (open && cached === null) checkCache();
		return () => {
			if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
		};
	});

	// reset on pair/depth change
	$effect(() => {
		void pairId; void depth;
		cached = null;
		refit = null;
		refitError = null;
		job = null;
		savedAt = null;
		selectedTile = null;
		loadSets();
		if (open) checkCache();
	});

	// re-fit whenever a human action lands (approve / correct / clear)
	let lastVersion = 0;
	$effect(() => {
		const v = annotationVersion;
		if (v !== lastVersion) {
			lastVersion = v;
			if (cached) { savedAt = null; runRefit(); }
		}
	});

	function lnccColor(s: number): string {
		const t = Math.max(0, Math.min(1, s));
		return t < 0.5
			? `rgb(255,${Math.round(t * 2 * 255)},0)`
			: `rgb(${Math.round((1 - (t - 0.5) * 2) * 255)},255,0)`;
	}

	interface RowStats {
		label: string;
		dx: number;
		dy: number;
		lncc2: number;
		factor: number;
	}

	interface TileStats {
		lncc2: number;
		rows: RowStats[];
	}

	let tileStats = $state<TileStats | null>(null);

	const previewRows = $derived(
		previewResult
			? [
					{ label: 'Included', dx: previewResult.ux, dy: previewResult.uy },
					{ label: 'Excluded', dx: previewResult.prior_dx, dy: previewResult.prior_dy }
				]
			: []
	);

	async function loadNormalizedGray(src: string): Promise<{ data: Float32Array; w: number; h: number }> {
		const img = new Image();
		img.src = src;
		await new Promise<void>((resolve, reject) => {
			img.onload = () => resolve();
			img.onerror = () => reject(new Error(`failed to load ${src}`));
		});
		const c = document.createElement('canvas');
		c.width = img.naturalWidth;
		c.height = img.naturalHeight;
		const ctx = c.getContext('2d')!;
		ctx.drawImage(img, 0, 0);
		const raw = ctx.getImageData(0, 0, c.width, c.height).data;
		const n = c.width * c.height;
		const grayRaw = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			grayRaw[i] = (raw[i * 4] + raw[i * 4 + 1] + raw[i * 4 + 2]) / 3;
		}
		const mean = grayRaw.reduce((a, b) => a + b, 0) / n;
		let variance = 0;
		for (let i = 0; i < n; i++) {
			const d = grayRaw[i] - mean;
			variance += d * d;
		}
		const std = Math.sqrt(variance / n) || 1;
		const gray = new Float32Array(n);
		for (let i = 0; i < n; i++) {
			gray[i] = Math.min(255, Math.max(0, ((grayRaw[i] - mean) / std) * 64 + 128));
		}
		return { data: gray, w: c.width, h: c.height };
	}

	$effect(() => {
		if (!previewResult || !previewHeSrc || !previewIhcSrc || !previewIhcIncludedSrc) {
			tileStats = null;
			return;
		}
		const heSrc = previewHeSrc;
		const priorSrc = previewIhcSrc;
		const includedSrc = previewIhcIncludedSrc;
		const ps = patchSize;
		const ux = previewResult.ux;
		const uy = previewResult.uy;
		const pdx = previewResult.prior_dx;
		const pdy = previewResult.prior_dy;
		let cancelled = false;
		tileStats = null;
		Promise.all([
			loadNormalizedGray(heSrc),
			loadNormalizedGray(priorSrc),
			loadNormalizedGray(includedSrc)
		])
			.then(([he, prior, included]) => {
				if (cancelled) return;
				const lncc2 = computeLNCC(he.data, prior.data, he.w, he.h, ps, true);
				const lncc2Included = computeLNCC(he.data, included.data, he.w, he.h, ps, true);
				const lncc2Excluded = lncc2;
				const base = lncc2 > 1e-9 ? lncc2 : 0;
				tileStats = {
					lncc2,
					rows: [
						{ label: 'Included', dx: ux, dy: uy, lncc2: lncc2Included, factor: base ? lncc2Included / base : 0 },
						{ label: 'Excluded', dx: pdx, dy: pdy, lncc2: lncc2Excluded, factor: base ? lncc2Excluded / base : 0 }
					]
				};
			})
			.catch((err) => {
				if (cancelled) return;
				console.error('tile stats failed', err);
				tileStats = null;
			});
		return () => { cancelled = true; };
	});

	$effect(() => {
		function onKeyDown(e: KeyboardEvent) {
			if (!e.shiftKey) return;
			const target = e.target as HTMLElement | null;
			if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
			const key = e.key.toLowerCase();
			if (key !== 'a' && key !== 'x' && key !== 's') return;
			e.preventDefault();

			const tile = previewTile;
			if (!tile) {
				onFlash?.('Hover or select a tile first', 'warn');
				return;
			}
			const result = previewResult;
			if (!result) {
				onFlash?.(`No candidate for ${tile} — recompute candidates`, 'warn');
				return;
			}

			if (key === 'a') {
				if (result.annotated) onFlash?.(`${tile} already voted — Shift+S to clear`, 'warn');
				else onApprove?.(tile, result.ux, result.uy);
			} else if (key === 'x') {
				if (result.annotated) onFlash?.(`${tile} already voted — Shift+S to clear`, 'warn');
				else onExclude?.(tile);
			} else {
				if (result.annotated) onClear?.(tile);
				else onFlash?.(`${tile} has no vote to clear`, 'warn');
			}
		}
		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	});

</script>

<div class="panel">
	<button class="toggle" onclick={() => (open = !open)}>
		<span class="arrow">{open ? '▾' : '▸'}</span>
		Coarse-to-fine field
		{#if refit}
			<span class="summary-inline">
				· {refit.tiles.length} tiles · τ={refit.tau.toExponential(1)}{#if tauMode === 'keep'} (excl {excludePct}%){/if} · {refit.kept} kept / {refit.rejected} rejected
				{#if refit.n_human}· {refit.n_human} human{/if}
			</span>
		{/if}
	</button>

	{#if open}
		<div class="body">
			<div class="set-bar">
				<span class="set-label">Field set</span>
				<select class="set-select" value={selectedSetId} onchange={onSelectChange} disabled={setBusy}>
					{#if sets.length === 0}
						<option value="">— none saved —</option>
					{/if}
					{#each sets as s (s.id)}
						<option value={s.id}>
							{s.name}{s.id === activeSetId ? ' ●' : ''}{s.id === mainSetId ? ' ★' : ''}{s.saved_depth != null ? ` · L${s.saved_depth}` : ''}
						</option>
					{/each}
					<option value="__new__">＋ New field set…</option>
				</select>
				<button class="set-btn set-btn-primary" onclick={saveSet} disabled={setBusy}>Save</button>
				<button class="set-btn set-btn-ghost" onclick={renameSelectedSet} disabled={setBusy || !selectedSetId}>Rename</button>
			</div>

			<div class="rating-bar">
				<span class="set-label">Rating</span>
				<span class="rating-group">
					{#each RATINGS as r}
						<button
							class="rating-btn rating-{r}"
							class:active={selectedSet?.rating === r}
							onclick={() => rateSet(r)}
							disabled={setBusy || !selectedSetId}
						>{r === 'bad' ? 'Bad' : r === 'ok' ? 'OK' : 'Good'}</button>
					{/each}
				</span>
				<button
					class="set-btn set-btn-ghost main-btn"
					class:active={!!selectedSetId && selectedSetId === mainSetId}
					onclick={pinMain}
					disabled={setBusy || !selectedSetId}
					title="Pin this set as the pair's main set (its rating shows in the sidebar)"
				>{selectedSetId && selectedSetId === mainSetId ? '★ Main' : '☆ Main'}</button>
			</div>

			{#if pendingSetId}
				<div class="set-modal-backdrop">
					<div class="set-modal" role="dialog" aria-modal="true">
						<p class="set-modal-title">Load “{pendingSet?.name ?? pendingSetId}”</p>
						<p class="set-modal-text">
							Save changes to the current set{activeSet ? ` “${activeSet.name}”` : ''} before switching?
						</p>
						<div class="set-modal-actions">
							<button class="set-btn set-btn-primary" onclick={() => confirmLoad(true)} disabled={setBusy}>Save & continue</button>
							<button class="set-btn" onclick={() => confirmLoad(false)} disabled={setBusy}>Continue without saving</button>
							<button class="set-btn set-btn-ghost" onclick={cancelLoad} disabled={setBusy}>Cancel</button>
						</div>
					</div>
				</div>
			{/if}
			{#if cached === null}
				<span class="loading">loading…</span>
			{:else if !cached}
				<div class="controls-row">
					{#if job?.running}
						<span class="progress">Computing candidates… {job.done} / {job.total || '?'}</span>
					{:else}
						<button class="compute-btn" onclick={startCompute}>Compute candidates</button>
					{/if}
					{#if job?.error}<span class="err">{job.error}</span>{/if}
				</div>
			{:else if refitError}
				<div class="controls-row">
					<span class="err">Fit failed: {refitError}</span>
					<button class="compute-btn" onclick={() => runRefit()}>Retry fit</button>
					<button class="compute-btn" onclick={() => startCompute()}>Recompute candidates</button>
				</div>
			{:else if refit === null}
				<span class="loading">fitting…</span>
			{:else}
				<div class="top">
					<C2fHeatmap
						{depth}
						tiles={refit.tiles}
						tau={refit.tau}
						{seed}
						selected={selectedTile}
						onhover={(t) => hoveredTile = t}
						onselect={onSelect}
					/>
					<C2fVectorField
						{depth}
						tiles={refit.tiles}
						mode={vectorMode}
						selected={selectedTile}
						hovered={hoveredTile}
						onhover={(t) => hoveredTile = t}
						onselect={onSelect}
					/>
					<div class="controls">
						<div class="mode-group">
							<span class="mode-label">Gate by</span>
							<div class="mode-buttons">
								<button
									class="mode-btn"
									class:active={tauMode === 'tau'}
									onclick={() => setTauMode('tau')}
								>τ</button>
								<button
									class="mode-btn"
									class:active={tauMode === 'keep'}
									onclick={() => setTauMode('keep')}
								>exclude %</button>
							</div>
						</div>
						{#if tauMode === 'tau'}
							<label class="tau-control">
								<span class="tau-label">τ = {tau.toExponential(2)}</span>
								<input
									type="range"
									min={EXP_MIN}
									max={EXP_MAX}
									step="0.05"
									value={tauExp}
									oninput={onTauInput}
								/>
							</label>
						{:else}
							<label class="tau-control">
								<span class="tau-label">
									exclude {excludePct}%
									<span class="tau-derived">τ = {effectiveTau != null ? effectiveTau.toExponential(2) : '…'}</span>
								</span>
								<input
									type="range"
									min={EXCLUDE_MIN}
									max={EXCLUDE_MAX}
									step="1"
									value={excludePct}
									oninput={onExcludeInput}
								/>
							</label>
						{/if}
						<div class="mode-group">
							<span class="mode-label">Vector field</span>
							<div class="mode-buttons">
								<button
									class="mode-btn"
									class:active={vectorMode === 'refinement'}
									onclick={() => vectorMode = 'refinement'}
								>Refinement</button>
								<button
									class="mode-btn"
									class:active={vectorMode === 'prior'}
									onclick={() => vectorMode = 'prior'}
								>Prior</button>
								<button
									class="mode-btn"
									class:active={vectorMode === 'result'}
									onclick={() => vectorMode = 'result'}
								>Result</button>
							</div>
						</div>
						<div class="stat"><span class="k">kept</span><span class="v green">{refit.kept}</span></div>
						<div class="stat"><span class="k">rejected</span><span class="v red">{refit.rejected}</span></div>
						{#if refit.excluded}
							<div class="stat"><span class="k">excluded</span><span class="v grey">{refit.excluded}</span></div>
						{/if}
						<div class="stat"><span class="k">human</span><span class="v indigo">{refit.n_human ?? 0}</span></div>
						<div class="stat"><span class="k">mean res</span><span class="v">{refit.mean_residual.toExponential(2)}</span></div>
						<button class="save-btn" onclick={saveField} disabled={saving}>
							{saving ? 'Saving…' : 'Save field'}
						</button>
						{#if savedAt}<span class="saved">✓ saved</span>{/if}
					</div>

					</div>
				{#if previewResult}
					<div class="tile-row">
						<div class="tile-row-header">
							<span class="tile-row-title">
								{#if selectedTile}
									Selected {selectedTile}
								{:else}
									Hover {hoveredTile}
								{/if}
							</span>
							<div class="tile-actions">
								<span class="shortcut-hint">Shift+A approve · Shift+X exclude · Shift+S clear</span>
								{#if previewResult.annotated}
									<span class="ann-badge ann-badge-{previewResult.annotated}">
										{previewResult.annotated === 'correct'
											? 'Corrected'
											: previewResult.annotated === 'exclude'
												? 'Excluded'
												: 'Approved'}
									</span>
									<button
										class="action-btn"
										onclick={() => onClear?.(previewResult.tile_loc)}
									>Clear vote</button>
								{:else}
									<button
										class="action-btn"
										onclick={() => onApprove?.(previewResult.tile_loc, previewResult.ux, previewResult.uy)}
									>Approve tile</button>
									<button
										class="action-btn action-btn-grey"
										onclick={() => onExclude?.(previewResult.tile_loc)}
									>Exclude tile</button>
								{/if}
							</div>
						</div>
						<div
							class="tile-row-grid"
							style:grid-template-columns="48px repeat(2, auto) repeat(4, 80px)"
						>
							<span class="col-header"></span>
							<span class="col-header">Overlay</span>
							<span class="col-header col-header-flex">
								Refined overlay
								<span class="emph-pills">
									<button
										class="emph-pill"
										class:active={emphasis === 'he'}
										onclick={() => onToggleEmphasis?.('he')}
										title="Highlight fixed HE (Shift+Q)"
									>HE</button>
									<button
										class="emph-pill"
										class:active={emphasis === 'ihc'}
										onclick={() => onToggleEmphasis?.('ihc')}
										title="Highlight moving IHC (Shift+W)"
									>IHC</button>
								</span>
							</span>
							<span class="col-header">LNCC²</span>
							<span class="col-header">|Δ| refined</span>
							<span class="col-header">LNCC² refined</span>
							<span class="col-header">Factor refined</span>

							{#each previewRows as row (row.label)}
								{@const stats = tileStats?.rows.find((r) => r.label === row.label)}
								<span class="fr-label {row.label.toLowerCase()}">{row.label}</span>
								<div class="fr-overlay">
									{#key previewTile}
										<OverlayCanvas heSrc={previewHeSrc} ihcSrc={previewIhcSrc} />
									{/key}
								</div>
								<div class="fr-overlay">
									{#key `${previewTile}-${row.label}`}
										<DisplacedOverlay
											heSrc={previewHeSrc}
											ihcSrc={liveCropSrc('ihc', row.dx, row.dy)}
											dx={0}
											dy={0}
											emphasis={emphasis}
										/>
									{/key}
								</div>
								{#if stats}
									<div class="fr-score" style:background={lnccColor(tileStats?.lncc2 ?? 0)}>
										<span class="value">{(tileStats?.lncc2 ?? 0).toFixed(3)}</span>
									</div>
									<div class="fr-disp">
										<DisplacementArrow dx={row.dx} dy={row.dy} />
									</div>
									<div class="fr-score" style:background={lnccColor(stats.lncc2)}>
										<span class="value">{stats.lncc2.toFixed(3)}</span>
									</div>
									<div class="fr-factor" class:positive={stats.factor > 1}>
										{stats.factor.toFixed(3)}
									</div>
								{:else}
									<div class="fr-score"><span class="placeholder">…</span></div>
									<div class="fr-disp"><span class="placeholder">…</span></div>
									<div class="fr-score"><span class="placeholder">…</span></div>
									<div class="fr-factor"><span class="placeholder">…</span></div>
								{/if}
							{/each}
						</div>
					</div>
				{:else}
					<div class="tile-row-hint">Hover or click a heatmap tile to inspect it</div>
				{/if}
			{/if}
		</div>
	{/if}
</div>

<style>
	.panel {
		border-bottom: 1px solid #2a2d3a;
		background: #131520;
		flex-shrink: 0;
	}

	.toggle {
		all: unset;
		cursor: pointer;
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 6px 14px;
		font-size: 0.72rem;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: #6b7280;
		width: 100%;
	}
	.toggle:hover { color: #e8eaf0; }
	.arrow { font-size: 0.65rem; }
	.summary-inline {
		font-weight: 400;
		text-transform: none;
		letter-spacing: 0;
		color: #9ca3af;
	}

	.body { padding: 8px 14px 14px; }
	.loading { font-size: 0.8rem; color: #6b7280; }

	.set-bar {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
		padding-bottom: 10px;
		margin-bottom: 10px;
		border-bottom: 1px solid #2a2d3a;
	}

	.set-label {
		font-size: 0.65rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #6b7280;
	}

	.rating-bar {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
		padding-bottom: 10px;
		margin-bottom: 10px;
		border-bottom: 1px solid #2a2d3a;
	}

	.rating-group {
		display: inline-flex;
		gap: 4px;
	}

	.rating-btn {
		all: unset;
		cursor: pointer;
		padding: 3px 10px;
		border-radius: 5px;
		font-size: 0.72rem;
		color: #9ca3af;
		background: #1e2130;
		border: 1px solid #2a2d3a;
	}
	.rating-btn:hover { border-color: #6b7280; }
	.rating-btn:disabled { opacity: 0.4; cursor: default; }
	.rating-btn:disabled:hover { border-color: #2a2d3a; }
	.rating-btn.rating-bad.active { background: #ef4444; border-color: #ef4444; color: #fff; }
	.rating-btn.rating-ok.active { background: #f59e0b; border-color: #f59e0b; color: #1a1205; }
	.rating-btn.rating-good.active { background: #22c55e; border-color: #22c55e; color: #06210f; }

	.main-btn.active { color: #fbbf24; border-color: #fbbf24; }

	.set-select {
		all: unset;
		cursor: pointer;
		min-width: 180px;
		padding: 4px 8px;
		border-radius: 5px;
		font-size: 0.75rem;
		color: #e8eaf0;
		background: #0f1117;
		border: 1px solid #2a2d3a;
	}
	.set-select:hover { border-color: #3a3f52; }
	.set-select:focus { border-color: #6366f1; }
	.set-select:disabled { opacity: 0.5; cursor: default; }

	.set-btn {
		all: unset;
		cursor: pointer;
		font-size: 0.7rem;
		font-weight: 600;
		padding: 4px 9px;
		border-radius: 5px;
		border: 1px solid #4b5563;
		color: #e8eaf0;
		background: #1e2130;
	}
	.set-btn:hover { background: #2a2d3a; border-color: #6b7280; }
	.set-btn:disabled { opacity: 0.4; cursor: default; }
	.set-btn:disabled:hover { background: #1e2130; border-color: #4b5563; }

	.set-btn-primary {
		background: #6366f1;
		border-color: #6366f1;
		color: #fff;
	}
	.set-btn-primary:hover { background: #4f51c8; border-color: #4f51c8; }

	.set-btn-ghost { color: #9ca3af; }

	.set-modal-backdrop {
		position: fixed;
		inset: 0;
		z-index: 50;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(0, 0, 0, 0.55);
	}

	.set-modal {
		min-width: 320px;
		max-width: 420px;
		padding: 18px;
		border-radius: 10px;
		background: #181b23;
		border: 1px solid #2a2d3a;
		box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
	}

	.set-modal-title {
		font-size: 0.9rem;
		font-weight: 700;
		color: #e8eaf0;
		margin-bottom: 6px;
	}

	.set-modal-text {
		font-size: 0.78rem;
		color: #9ca3af;
		margin-bottom: 16px;
		line-height: 1.4;
	}

	.set-modal-actions {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
		justify-content: flex-end;
	}

	.controls-row { display: flex; align-items: center; gap: 12px; }

	.top { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }

	.controls {
		display: flex;
		flex-direction: column;
		gap: 8px;
		width: 200px;
		max-width: 220px;
		min-width: 180px;
		flex: 0 0 auto;
	}

	.tau-control { display: flex; flex-direction: column; gap: 4px; }
	.tau-label { font-size: 0.72rem; color: #9ca3af; font-variant-numeric: tabular-nums; }
	.tau-derived { margin-left: 6px; color: #6b7280; }
	.tau-control input[type='range'] { width: 100%; accent-color: #6366f1; cursor: pointer; }

	.stat {
		display: flex;
		justify-content: space-between;
		font-size: 0.75rem;
		font-variant-numeric: tabular-nums;
	}
	.stat .k { color: #6b7280; }
	.stat .v { color: #e8eaf0; }
	.stat .v.green { color: #22c55e; }
	.stat .v.red { color: #ef4444; }
	.stat .v.grey { color: #9ca3af; }
	.stat .v.indigo { color: #a5b4fc; }

	.compute-btn, .save-btn {
		all: unset;
		cursor: pointer;
		background: #6366f1;
		color: #fff;
		font-size: 0.72rem;
		font-weight: 600;
		padding: 6px 12px;
		border-radius: 4px;
		text-align: center;
		white-space: nowrap;
	}
	.compute-btn:hover, .save-btn:hover { background: #4f51c8; }
	.save-btn[disabled] { opacity: 0.6; cursor: default; }

	.progress { font-size: 0.72rem; color: #f59e0b; }
	.saved { font-size: 0.72rem; color: #22c55e; }
	.err { font-size: 0.7rem; color: #ef4444; }
	.emph-pills {
		display: inline-flex;
		gap: 4px;
	}

	.emph-pill {
		all: unset;
		cursor: pointer;
		font-size: 0.6rem;
		font-weight: 700;
		padding: 2px 6px;
		border-radius: 4px;
		border: 1px solid #4b5563;
		color: #9ca3af;
		background: #1e2130;
	}

	.emph-pill.active {
		border-color: #6366f1;
		color: #e0e7ff;
		background: #312e81;
	}

	.tile-row {
		margin-top: 14px;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.tile-row-header {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.tile-actions {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.ann-badge {
		font-size: 0.65rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 2px 6px;
		border-radius: 4px;
	}

	.ann-badge-approve {
		color: #facc15;
		background: rgba(250, 204, 21, 0.12);
		border: 1px solid rgba(250, 204, 21, 0.35);
	}

	.ann-badge-correct {
		color: #a5b4fc;
		background: rgba(165, 180, 252, 0.12);
		border: 1px solid rgba(165, 180, 252, 0.35);
	}

	.ann-badge-exclude {
		color: #9ca3af;
		background: rgba(161, 161, 170, 0.12);
		border: 1px solid rgba(161, 161, 170, 0.35);
	}

	.action-btn {
		all: unset;
		cursor: pointer;
		font-size: 0.68rem;
		font-weight: 600;
		padding: 4px 8px;
		border-radius: 4px;
		border: 1px solid #4b5563;
		color: #e8eaf0;
		background: #1e2130;
	}

	.action-btn:hover {
		background: #2a2d3a;
		border-color: #6b7280;
	}

	.action-btn-grey {
		color: #d1d5db;
		border-color: #6b7280;
	}

	.action-btn-grey:hover {
		background: #374151;
		border-color: #9ca3af;
	}

	.mode-group {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.mode-label {
		font-size: 0.65rem;
		font-weight: 700;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: #6b7280;
	}

	.mode-buttons {
		display: flex;
		gap: 4px;
	}

	.mode-btn {
		all: unset;
		cursor: pointer;
		font-size: 0.62rem;
		font-weight: 600;
		padding: 4px 6px;
		border-radius: 4px;
		border: 1px solid #4b5563;
		color: #9ca3af;
		background: #1e2130;
		flex: 1;
		text-align: center;
	}

	.mode-btn:hover {
		background: #2a2d3a;
		border-color: #6b7280;
	}

	.mode-btn.active {
		border-color: #6366f1;
		color: #e0e7ff;
		background: #312e81;
	}

	.shortcut-hint {
		font-size: 0.6rem;
		color: #6b7280;
		font-variant-numeric: tabular-nums;
	}

	.tile-row-title {
		font-size: 0.75rem;
		font-weight: 700;
		color: #e8eaf0;
	}

	.tile-row-grid {
		display: grid;
		column-gap: 12px;
		row-gap: 12px;
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

	.col-header-flex {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.fr-label {
		width: 48px;
		font-size: 0.65rem;
		font-weight: 700;
		color: #9ca3af;
		font-variant-numeric: tabular-nums;
		text-align: center;
		flex-shrink: 0;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.fr-label.included { color: #22c55e; }
	.fr-label.excluded { color: #f97316; }

	.fr-overlay {
		flex-shrink: 0;
	}

	.fr-overlay :global(.wrap),
	.fr-overlay :global(canvas) {
		height: 180px;
		width: 269px;
	}

	.fr-score {
		width: 80px;
		height: 180px;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 4px;
		font-size: 0.8rem;
		font-weight: 700;
		color: #0f1117;
		flex-shrink: 0;
	}

	.fr-score .value {
		background: rgba(15, 17, 23, 0.35);
		padding: 2px 6px;
		border-radius: 4px;
	}

	.fr-score .placeholder,
	.fr-factor .placeholder {
		color: #4b5563;
	}

	.fr-disp {
		width: 80px;
		height: 180px;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #0f1117;
		flex-shrink: 0;
	}

	.fr-factor {
		width: 80px;
		height: 180px;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #0f1117;
		font-size: 0.78rem;
		color: #9ca3af;
		flex-shrink: 0;
	}

	.fr-factor.positive {
		color: #22c55e;
		background: #0f291e;
		border-color: #166534;
	}

	.tile-row-hint {
		margin-top: 14px;
		font-size: 0.72rem;
		color: #6b7280;
		padding: 10px 0;
		border-top: 1px solid #2a2d3a;
	}
</style>
