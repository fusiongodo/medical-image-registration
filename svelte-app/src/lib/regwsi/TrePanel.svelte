<script lang="ts" module>
	export interface TreStats {
		mean: number | null;
		median: number | null;
		max: number | null;
		p95: number | null;
		per_point: number[];
		error?: string;
	}
	export interface TreResult {
		n: number;
		field_set_id: string | null;
		none: TreStats;
		regwsi: TreStats;
		ours: TreStats;
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
	<p class="tre-sub">
		L5 px · vs {mainSetName ?? '—'}
		{#if mainSetId}<span class="muted">({mainSetId})</span>{/if}
	</p>
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
					<th>ours</th>
				</tr>
			</thead>
			<tbody>
				<tr>
					<td>mean L5 px</td>
					<td>{fmt(tre.none.mean)}</td>
					<td>{fmt(tre.regwsi.mean)}</td>
					<td>{fmt(tre.ours.mean)}</td>
				</tr>
				<tr>
					<td>median</td>
					<td>{fmt(tre.none.median)}</td>
					<td>{fmt(tre.regwsi.median)}</td>
					<td>{fmt(tre.ours.median)}</td>
				</tr>
				<tr>
					<td>max</td>
					<td>{fmt(tre.none.max)}</td>
					<td>{fmt(tre.regwsi.max)}</td>
					<td>{fmt(tre.ours.max)}</td>
				</tr>
				<tr>
					<td>p95</td>
					<td>{fmt(tre.none.p95)}</td>
					<td>{fmt(tre.regwsi.p95)}</td>
					<td>{fmt(tre.ours.p95)}</td>
				</tr>
			</tbody>
		</table>
		{#if tre.ours.error}
			<p class="err">{tre.ours.error}</p>
		{/if}
		{#if tre.none.per_point?.length}
			<ul class="per">
				{#each tre.none.per_point as e, i}
					<li>
						#{i + 1}
						<span>n {e.toFixed(1)}</span>
						<span>r {(tre.regwsi.per_point[i] ?? NaN).toFixed(1)}</span>
						<span>o {(tre.ours.per_point[i] ?? NaN).toFixed(1)}</span>
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
	.per {
		list-style: none;
		margin: 0.75rem 0 0;
		padding: 0;
		max-height: 14rem;
		overflow: auto;
	}
	.per li {
		display: flex;
		gap: 0.5rem;
		justify-content: space-between;
		padding: 0.15rem 0;
		border-bottom: 1px solid #2a2d3a;
		font-variant-numeric: tabular-nums;
		color: #c4c9d4;
	}
</style>
