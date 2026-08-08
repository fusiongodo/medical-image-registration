<script lang="ts" module>
	export interface TreStats {
		mean: number | null;
		median: number | null;
		max: number | null;
		p95: number | null;
		per_point: number[];
		error?: string;
		field_set_id?: string | null;
		field_set_name?: string | null;
		field_estimator?: string | null;
	}
	export interface TreResult {
		n: number;
		field_set_id: string | null;
		none: TreStats;
		regwsi: TreStats;
		tps?: TreStats;
		wendland?: TreStats;
		ours?: TreStats;
	}
</script>

<script lang="ts">
	let {
		pairId = null,
		tre,
		treBusy,
		treErr,
		landmarkCount,
		mainSetName = null,
		mainSetId = null,
		emptyHint = 'Add correspondences to compute TRE',
		onClose = undefined
	}: {
		pairId?: number | null;
		tre: TreResult | null;
		treBusy: boolean;
		treErr: string | null;
		landmarkCount: number;
		mainSetName?: string | null;
		mainSetId?: string | null;
		emptyHint?: string;
		onClose?: (() => void) | undefined;
	} = $props();

	function fmt(v: number | null | undefined) {
		return v == null ? '—' : v.toFixed(2);
	}

	function fmtPt(arr: number[] | undefined, i: number) {
		const v = arr?.[i];
		return v == null || Number.isNaN(v) ? '—' : v.toFixed(1);
	}

	const tps = $derived(tre?.tps ?? tre?.ours ?? null);
	const wendland = $derived(tre?.wendland ?? null);
	const canOpenRegwsi = $derived(tre?.regwsi?.mean != null);
	const canOpenTps = $derived(tps?.mean != null);
	const canOpenWendland = $derived(wendland?.mean != null);

	const subtitle = $derived.by(() => {
		const parts: string[] = [];
		if (tps?.field_set_name) parts.push(`TPS: ${tps.field_set_name}`);
		else if (mainSetName) parts.push(`TPS: ${mainSetName}`);
		if (wendland?.field_set_name) parts.push(`Wendland: ${wendland.field_set_name}`);
		if (parts.length) return parts.join(' · ');
		if (mainSetId) return mainSetId;
		return '—';
	});

	function openOverlay(kind: 'regwsi' | 'fieldset', estimator?: 'tps' | 'wendland') {
		if (pairId == null) return;
		if (kind === 'regwsi') {
			window.open(`/eval/${pairId}/overlay/regwsi`, `eval-overlay-${pairId}-regwsi`);
			return;
		}
		const est = estimator ?? 'tps';
		window.open(
			`/eval/${pairId}/overlay/fieldset?estimator=${est}`,
			`eval-overlay-${pairId}-fieldset-${est}`
		);
	}

	function openBothOverlays() {
		if (pairId == null) return;
		if (canOpenRegwsi) openOverlay('regwsi');
		if (canOpenTps) openOverlay('fieldset', 'tps');
		else if (canOpenWendland) openOverlay('fieldset', 'wendland');
	}

	const canOpenBoth = $derived(canOpenRegwsi && (canOpenTps || canOpenWendland));
</script>

<aside class="tre-panel">
	<div class="head">
		<h2>
			TRE{#if pairId != null}
				<span class="pair"> · pair {pairId}</span>
			{/if}
		</h2>
		{#if onClose}
			<button type="button" class="close" onclick={onClose} title="Close">×</button>
		{/if}
	</div>
	<p class="tre-sub">L5 px · {subtitle}</p>
	{#if treBusy}
		<p class="muted">Computing…</p>
	{:else if landmarkCount === 0}
		<p class="muted">{emptyHint}</p>
	{:else if treErr}
		<p class="err">{treErr}</p>
	{:else if tre}
		<table>
			<thead>
				<tr>
					<th></th>
					<th>none</th>
					<th>regWSI</th>
					<th>TPS</th>
					<th>Wend.</th>
				</tr>
			</thead>
			<tbody>
				<tr>
					<td>mean L5 px</td>
					<td>{fmt(tre.none.mean)}</td>
					<td>{fmt(tre.regwsi.mean)}</td>
					<td>{fmt(tps?.mean)}</td>
					<td>{fmt(wendland?.mean)}</td>
				</tr>
				<tr>
					<td>median</td>
					<td>{fmt(tre.none.median)}</td>
					<td>{fmt(tre.regwsi.median)}</td>
					<td>{fmt(tps?.median)}</td>
					<td>{fmt(wendland?.median)}</td>
				</tr>
				<tr>
					<td>max</td>
					<td>{fmt(tre.none.max)}</td>
					<td>{fmt(tre.regwsi.max)}</td>
					<td>{fmt(tps?.max)}</td>
					<td>{fmt(wendland?.max)}</td>
				</tr>
				<tr>
					<td>p95</td>
					<td>{fmt(tre.none.p95)}</td>
					<td>{fmt(tre.regwsi.p95)}</td>
					<td>{fmt(tps?.p95)}</td>
					<td>{fmt(wendland?.p95)}</td>
				</tr>
			</tbody>
		</table>
		{#if tps?.error}
			<p class="err">TPS · {tps.error}</p>
		{/if}
		{#if wendland?.error}
			<p class="err">Wendland · {wendland.error}</p>
		{/if}
		<div class="launch">
			<button
				type="button"
				class="launch-btn primary"
				disabled={!canOpenBoth}
				onclick={openBothOverlays}
				title="Open regWSI and field-set overlays in two tabs"
			>Open both overlays</button>
			<button
				type="button"
				class="launch-btn"
				disabled={!canOpenRegwsi}
				onclick={() => openOverlay('regwsi')}
			>regWSI only</button>
			<button
				type="button"
				class="launch-btn"
				disabled={!canOpenTps}
				onclick={() => openOverlay('fieldset', 'tps')}
			>TPS only</button>
			<button
				type="button"
				class="launch-btn"
				disabled={!canOpenWendland}
				onclick={() => openOverlay('fieldset', 'wendland')}
			>Wendland only</button>
		</div>
		{#if tre.none.per_point?.length}
			<ul class="per">
				{#each tre.none.per_point as e, i}
					<li>
						#{i + 1}
						<span>n {fmtPt(tre.none.per_point, i)}</span>
						<span>r {fmtPt(tre.regwsi.per_point, i)}</span>
						<span>t {fmtPt(tps?.per_point, i)}</span>
						<span>w {fmtPt(wendland?.per_point, i)}</span>
					</li>
				{/each}
			</ul>
		{/if}
	{/if}
</aside>

<style>
	.tre-panel {
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		padding: 0.85rem;
		background: #181b23;
		font-size: 0.8rem;
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
		margin-bottom: 0.25rem;
	}
	.tre-panel h2 {
		margin: 0;
		font-size: 0.95rem;
	}
	.pair {
		font-weight: 500;
		color: #9ca3af;
	}
	.close {
		all: unset;
		cursor: pointer;
		width: 1.5rem;
		height: 1.5rem;
		display: grid;
		place-items: center;
		border-radius: 4px;
		color: #9ca3af;
		font-size: 1.1rem;
		line-height: 1;
	}
	.close:hover {
		background: #1e2130;
		color: #e8eaf0;
	}
	.tre-sub {
		margin: 0 0 0.75rem;
		color: #9ca3af;
		font-size: 0.75rem;
		line-height: 1.3;
	}
	.muted {
		color: #6b7280;
	}
	.tre-panel table {
		width: 100%;
		border-collapse: collapse;
		margin-top: 0.5rem;
	}
	.tre-panel th,
	.tre-panel td {
		text-align: right;
		padding: 0.2rem 0.25rem;
		border-bottom: 1px solid #2a2d3a;
	}
	.tre-panel th:first-child,
	.tre-panel td:first-child {
		text-align: left;
		color: #9ca3af;
	}
	.err {
		color: #f87171;
		font-size: 0.75rem;
		margin: 0.5rem 0 0;
	}
	.launch {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		margin-top: 0.75rem;
	}
	.launch-btn {
		padding: 0.4rem 0.55rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		background: #1e2130;
		color: #c4c9d4;
		font-size: 0.75rem;
		cursor: pointer;
		text-align: left;
		font-family: inherit;
	}
	.launch-btn:hover:not(:disabled) {
		border-color: #5b8def;
		color: #e8eaf0;
	}
	.launch-btn.primary {
		border-color: #3d6b3d;
		background: #152018;
		color: #e8eaf0;
		font-weight: 600;
	}
	.launch-btn.primary:hover:not(:disabled) {
		border-color: #4ade80;
	}
	.launch-btn:disabled {
		opacity: 0.4;
		cursor: default;
	}
	.per {
		list-style: none;
		margin: 0.75rem 0 0;
		padding: 0;
		max-height: 14rem;
		overflow: auto;
	}
	.per li {
		display: flex;
		gap: 0.45rem;
		justify-content: space-between;
		padding: 0.15rem 0;
		border-bottom: 1px solid #2a2d3a;
		font-variant-numeric: tabular-nums;
		color: #c4c9d4;
		font-size: 0.75rem;
	}
</style>
