<script lang="ts">
	let { data } = $props();
</script>

<div class="page">
	<div class="inner">
		<header>
			<h1>regWSI preregistration</h1>
			<p class="sub">
				DeeperHistReg affine + deformable field per pair. Open a registered pair to inspect the HE /
				warped-IHC overlay.
			</p>
		</header>
		<ul class="list">
			{#each data.pairs as p}
				<li class:ready={p.ready}>
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
							<a class="annotate" href={`/regwsi/${p.pairId}/annotate`}>Annotate</a>
						{/if}
					</div>
				</li>
			{/each}
		</ul>
	</div>
</div>

<style>
	.page {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
	.inner {
		padding: 1.5rem 2rem 2.5rem;
		max-width: 40rem;
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
	.label {
		color: #f3f4f6;
		font-weight: 600;
		font-size: 0.95rem;
	}
	.list a.annotate {
		display: flex;
		align-items: center;
		padding: 0.55rem 0.75rem;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		text-decoration: none;
		color: #c4c9d4;
		font-size: 0.8rem;
		background: #181b23;
	}
	.list a.annotate:hover {
		background: #1e2130;
		color: #e8eaf0;
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
