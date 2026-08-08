<script lang="ts">
	export interface TreStats {
		mean: number | null;
		median: number | null;
		max: number | null;
		p95: number | null;
		per_point?: number[];
		error?: string;
	}

	export interface MethodCell {
		key: string;
		lam: string;
		field_estimator: string;
		complete: boolean;
		tre: TreStats;
	}

	export interface BatchTreResult {
		pair_id: number;
		batch_id: string;
		n: number;
		none: TreStats;
		regwsi: TreStats;
		methods: MethodCell[];
	}

	let {
		pairId,
		tre,
		treBusy,
		treErr,
		landmarkCount,
		batchId = null,
		onClose = undefined
	}: {
		pairId: number | null;
		tre: BatchTreResult | null;
		treBusy: boolean;
		treErr: string | null;
		landmarkCount: number;
		batchId?: string | null;
		onClose?: (() => void) | undefined;
	} = $props();

	function fmt(v: number | null | undefined) {
		return v == null ? '—' : v.toFixed(2);
	}

	function openRegwsi() {
		if (pairId == null) return;
		window.open(`/eval/${pairId}/overlay/regwsi`, `eval-overlay-${pairId}-regwsi`);
	}

	function openMethod(m: MethodCell) {
		if (pairId == null || !batchId) return;
		const q = new URLSearchParams({
			estimator: m.field_estimator,
			lam: m.lam,
			batch: batchId
		});
		window.open(
			`/eval/${pairId}/overlay/fieldset?${q}`,
			`eval-overlay-${pairId}-${m.lam}-${m.field_estimator}`
		);
	}

	const methods = $derived(tre?.methods ?? []);
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
		L5 px{#if batchId}
			· batch {batchId}{/if}
	</p>

	{#if treBusy}
		<p class="muted">Computing…</p>
	{:else if landmarkCount === 0}
		<p class="muted">Add landmarks to compute TRE</p>
	{:else if treErr}
		<p class="err">{treErr}</p>
	{:else if tre}
		<div class="scroll">
			<table>
				<thead>
					<tr>
						<th></th>
						<th>none</th>
						<th>regWSI</th>
						{#each methods as m}
							<th title={m.key}>{m.lam.slice(0, 3)}/{m.field_estimator.slice(0, 3)}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					<tr>
						<td>mean</td>
						<td>{fmt(tre.none.mean)}</td>
						<td>{fmt(tre.regwsi.mean)}</td>
						{#each methods as m}
							<td class:missing={!m.complete}>{fmt(m.tre.mean)}</td>
						{/each}
					</tr>
					<tr>
						<td>median</td>
						<td>{fmt(tre.none.median)}</td>
						<td>{fmt(tre.regwsi.median)}</td>
						{#each methods as m}
							<td class:missing={!m.complete}>{fmt(m.tre.median)}</td>
						{/each}
					</tr>
					<tr>
						<td>max</td>
						<td>{fmt(tre.none.max)}</td>
						<td>{fmt(tre.regwsi.max)}</td>
						{#each methods as m}
							<td class:missing={!m.complete}>{fmt(m.tre.max)}</td>
						{/each}
					</tr>
					<tr>
						<td>p95</td>
						<td>{fmt(tre.none.p95)}</td>
						<td>{fmt(tre.regwsi.p95)}</td>
						{#each methods as m}
							<td class:missing={!m.complete}>{fmt(m.tre.p95)}</td>
						{/each}
					</tr>
				</tbody>
			</table>
		</div>
		<p class="n">n = {tre.n} landmarks</p>
		<div class="actions">
			<button type="button" class="btn" disabled={tre.regwsi.mean == null} onclick={openRegwsi}>
				Open regWSI overlay
			</button>
			{#each methods as m}
				<button
					type="button"
					class="btn"
					disabled={!m.complete || m.tre.mean == null || !batchId}
					onclick={() => openMethod(m)}
					title={m.key}
				>
					Open {m.lam}/{m.field_estimator}
				</button>
			{/each}
		</div>
	{/if}
</aside>

<style>
	.tre-panel {
		display: flex;
		flex-direction: column;
		gap: 0.65rem;
		min-height: 0;
		height: 100%;
		padding: 0.85rem 1rem;
		background: #12141a;
		border-left: 1px solid #2a2d3a;
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}
	h2 {
		margin: 0;
		font-size: 0.95rem;
		font-weight: 600;
		color: #e8eaf0;
	}
	.pair {
		color: #9ca3af;
		font-weight: 400;
	}
	.close {
		border: none;
		background: transparent;
		color: #9ca3af;
		font-size: 1.2rem;
		cursor: pointer;
		line-height: 1;
	}
	.tre-sub {
		margin: 0;
		font-size: 0.72rem;
		color: #6b7280;
	}
	.muted {
		margin: 0;
		color: #6b7280;
		font-size: 0.8rem;
	}
	.err {
		margin: 0;
		color: #f87171;
		font-size: 0.8rem;
	}
	.scroll {
		overflow: auto;
		flex: 1;
		min-height: 0;
	}
	table {
		border-collapse: collapse;
		font-size: 0.72rem;
		width: max-content;
		min-width: 100%;
	}
	th,
	td {
		padding: 0.28rem 0.45rem;
		border-bottom: 1px solid #1f2330;
		text-align: right;
		white-space: nowrap;
	}
	th:first-child,
	td:first-child {
		text-align: left;
		color: #9ca3af;
		position: sticky;
		left: 0;
		background: #12141a;
	}
	th {
		color: #9ca3af;
		font-weight: 500;
	}
	td.missing {
		color: #4b5563;
	}
	.n {
		margin: 0;
		font-size: 0.72rem;
		color: #6b7280;
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
	}
	.btn {
		padding: 0.3rem 0.55rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		background: #181b23;
		color: #e8eaf0;
		font-size: 0.7rem;
		cursor: pointer;
	}
	.btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.btn:not(:disabled):hover {
		border-color: #3b82f6;
	}
</style>
