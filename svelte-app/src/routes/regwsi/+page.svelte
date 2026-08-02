<script lang="ts">
	import TrePanel, { type TreResult } from '$lib/regwsi/TrePanel.svelte';

	let { data } = $props();

	let selectedPair = $state<number | null>(null);
	let tre = $state<TreResult | null>(null);
	let treBusy = $state(false);
	let treErr = $state<string | null>(null);
	let fetchGen = 0;

	const selected = $derived(
		selectedPair == null ? null : (data.pairs.find((p) => p.pairId === selectedPair) ?? null)
	);

	async function fetchTre(pairId: number) {
		const gen = ++fetchGen;
		treBusy = true;
		treErr = null;
		tre = null;
		try {
			const r = await fetch(`/api/regwsi/tre?pair=${pairId}`);
			if (!r.ok) throw new Error(await r.text());
			const json = await r.json();
			if (gen !== fetchGen) return;
			tre = json;
		} catch (e) {
			if (gen !== fetchGen) return;
			treErr = e instanceof Error ? e.message : 'tre failed';
			tre = null;
		} finally {
			if (gen === fetchGen) treBusy = false;
		}
	}

	function toggleTre(pairId: number) {
		if (selectedPair === pairId) {
			selectedPair = null;
			tre = null;
			treErr = null;
			treBusy = false;
			fetchGen += 1;
			return;
		}
		selectedPair = pairId;
		void fetchTre(pairId);
	}

	function closeTre() {
		selectedPair = null;
		tre = null;
		treErr = null;
		treBusy = false;
		fetchGen += 1;
	}
</script>

<div class="page">
	<div class="layout" class:with-panel={selectedPair != null}>
		<div class="list-col">
			<header>
				<h1>regWSI preregistration</h1>
				<p class="sub">
					DeeperHistReg affine + deformable field per pair. Open a registered pair to inspect the HE /
					warped-IHC overlay.
				</p>
			</header>
			<ul class="list">
				{#each data.pairs as p}
					<li class:ready={p.ready} class:selected={selectedPair === p.pairId}>
						<div class="row">
							<a class="main" href={`/regwsi/${p.pairId}`}>
								<span class="label">Pair {p.pairId}</span>
								<span class="meta">
									{#if p.landmarkCount > 0}
										<span class="landmarks">{p.landmarkCount} landmarks</span>
									{/if}
									<span class="badge">{p.ready ? 'registered' : 'not yet'}</span>
								</span>
							</a>
							{#if p.ready}
								<a class="action annotate" href={`/regwsi/${p.pairId}/annotate`}>Annotate</a>
								<button
									type="button"
									class="action tre-btn"
									class:active={selectedPair === p.pairId}
									onclick={() => toggleTre(p.pairId)}
								>
									TRE
								</button>
							{/if}
						</div>
					</li>
				{/each}
			</ul>
		</div>
		{#if selected && selectedPair != null}
			<div class="panel-col">
				<TrePanel
					pairId={selectedPair}
					{tre}
					{treBusy}
					{treErr}
					landmarkCount={selected.landmarkCount}
					mainSetName={selected.mainSetName}
					mainSetId={selected.mainSetId}
					onClose={closeTre}
				/>
			</div>
		{/if}
	</div>
</div>

<style>
	.page {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
	.layout {
		display: flex;
		align-items: flex-start;
		gap: 1rem;
		padding: 1.5rem 2rem 2.5rem;
		max-width: 40rem;
	}
	.layout.with-panel {
		max-width: 56rem;
	}
	.list-col {
		flex: 1;
		min-width: 0;
	}
	.panel-col {
		flex: 0 0 280px;
		position: sticky;
		top: 1.5rem;
	}
	h1 {
		margin: 0 0 0.35rem;
		font-size: 1.35rem;
		font-weight: 600;
	}
	.sub {
		margin: 0 0 1.25rem;
		color: #9ca3af;
		font-size: 0.9rem;
		line-height: 1.4;
	}
	.list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}
	.row {
		display: flex;
		align-items: stretch;
		gap: 0.35rem;
	}
	.list a.main {
		flex: 1;
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.55rem 0.75rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		text-decoration: none;
		color: #e8eaf0;
		background: #181b23;
	}
	.list a.main:hover {
		background: #1e2130;
	}
	.list li.ready a.main {
		border-color: #3d6b3d;
		background: #152018;
	}
	.list li.selected a.main {
		border-color: #5b8def;
	}
	.label {
		color: #f3f4f6;
		font-weight: 600;
		font-size: 0.95rem;
	}
	.action {
		display: flex;
		align-items: center;
		padding: 0.55rem 0.75rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		text-decoration: none;
		color: #c4c9d4;
		font-size: 0.8rem;
		background: #181b23;
		cursor: pointer;
		font-family: inherit;
	}
	.action:hover {
		background: #1e2130;
		color: #e8eaf0;
	}
	.tre-btn.active {
		border-color: #5b8def;
		color: #e8eaf0;
		background: #1e2740;
	}
	.meta {
		display: flex;
		align-items: center;
		gap: 0.65rem;
	}
	.landmarks {
		font-size: 0.8rem;
		color: #7dd3fc;
		font-variant-numeric: tabular-nums;
	}
	.badge {
		font-size: 0.75rem;
		color: #6b7280;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.list li.ready .badge {
		color: #4ade80;
	}
</style>
