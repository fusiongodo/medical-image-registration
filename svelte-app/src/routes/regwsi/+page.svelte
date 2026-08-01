<script lang="ts">
	let { data } = $props();
</script>

<div class="page">
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
						<span class="badge">{p.ready ? 'registered' : 'not yet'}</span>
					</a>
					{#if p.ready}
						<a class="annotate" href={`/regwsi/${p.pairId}/annotate`}>Annotate</a>
					{/if}
				</div>
			</li>
		{/each}
	</ul>
</div>

<style>
	.page {
		padding: 1.5rem 2rem;
		max-width: 40rem;
	}
	h1 {
		margin: 0 0 0.35rem;
		font-size: 1.35rem;
		font-weight: 600;
	}
	.sub {
		margin: 0 0 1.25rem;
		color: #666;
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
		border: 1px solid #ddd;
		border-radius: 4px;
		text-decoration: none;
		color: inherit;
		background: #fafafa;
	}
	.list a.main:hover {
		background: #f0f0f0;
	}
	.list li.ready a.main {
		border-color: #9bc49b;
		background: #f4faf4;
	}
	.list a.annotate {
		display: flex;
		align-items: center;
		padding: 0.55rem 0.75rem;
		border: 1px solid #ccc;
		border-radius: 4px;
		text-decoration: none;
		color: #444;
		font-size: 0.8rem;
		background: #fff;
	}
	.list a.annotate:hover {
		background: #f0f0f0;
	}
	.badge {
		font-size: 0.75rem;
		color: #888;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.list li.ready .badge {
		color: #2a7a2a;
	}
</style>
