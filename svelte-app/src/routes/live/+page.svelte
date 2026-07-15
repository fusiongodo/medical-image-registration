<script lang="ts">
	let { data } = $props<{ data: { numPairs: number; maxDepth: number } }>();

	let pair = $state(0);
	let level = $state(2);
	let tiles = $state<string[]>([]);
	let meta = $state<{ grid: number; page: number | null; tile_w: number; tile_h: number } | null>(
		null
	);
	let loading = $state(false);
	let errorMsg = $state<string | null>(null);

	const pairs = $derived(Array.from({ length: data.numPairs }, (_, i) => i));
	const levels = $derived(Array.from({ length: data.maxDepth + 1 }, (_, i) => i));

	function tileUrl(loc: string, side: 'he' | 'ihc'): string {
		const [x, y] = loc.split('_');
		return `/api/live-crop/tile?pair=${pair}&level=${level}&x=${x}&y=${y}&side=${side}`;
	}

	$effect(() => {
		const p = pair;
		const l = level;
		loading = true;
		errorMsg = null;
		tiles = [];
		meta = null;

		fetch(`/api/live-crop/tiles?pair=${p}&level=${l}`)
			.then(async (r) => {
				if (!r.ok) throw new Error(await r.text());
				return r.json();
			})
			.then((d) => {
				if (p !== pair || l !== level) return;
				meta = { grid: d.grid, page: d.page, tile_w: d.tile_w, tile_h: d.tile_h };
				tiles = d.tiles ?? [];
			})
			.catch((e) => {
				if (p === pair && l === level) errorMsg = String(e?.message ?? e);
			})
			.finally(() => {
				if (p === pair && l === level) loading = false;
			});
	});
</script>

<div class="live">
	<header>
		<h1>Live TIFF crop</h1>
		<label>
			Pair
			<select bind:value={pair}>
				{#each pairs as p}
					<option value={p}>{p}</option>
				{/each}
			</select>
		</label>
		<label>
			Level
			<select bind:value={level}>
				{#each levels as l}
					<option value={l}>{l}</option>
				{/each}
			</select>
		</label>
		{#if meta}
			<span class="meta">
				grid {meta.grid}×{meta.grid} · page {meta.page} · {tiles.length} tissue tiles
			</span>
		{/if}
	</header>

	{#if errorMsg}
		<p class="err">{errorMsg}</p>
	{/if}
	{#if loading}
		<p class="status">Loading tile list…</p>
	{:else if !errorMsg && tiles.length === 0}
		<p class="status">No tissue tiles at this level.</p>
	{/if}

	<div class="rows">
		{#each tiles as loc (loc)}
			<div class="row">
				<span class="loc">{loc}</span>
				<figure>
					<img loading="lazy" src={tileUrl(loc, 'he')} alt={`HE ${loc}`} />
					<figcaption>HE</figcaption>
				</figure>
				<figure>
					<img loading="lazy" src={tileUrl(loc, 'ihc')} alt={`IHC ${loc}`} />
					<figcaption>IHC</figcaption>
				</figure>
			</div>
		{/each}
	</div>
</div>

<style>
	.live {
		display: flex;
		flex-direction: column;
		height: 100%;
		overflow: hidden;
	}

	header {
		display: flex;
		align-items: center;
		gap: 18px;
		padding: 14px 20px;
		border-bottom: 1px solid #2a2d3a;
		flex-shrink: 0;
	}

	h1 {
		font-size: 0.95rem;
		font-weight: 600;
	}

	label {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 0.8rem;
		color: #9ca3af;
	}

	select {
		background: #1e2130;
		color: #e8eaf0;
		border: 1px solid #2a2d3a;
		border-radius: 5px;
		padding: 4px 8px;
		font-size: 0.8rem;
	}

	.meta {
		margin-left: auto;
		font-size: 0.78rem;
		color: #6b7280;
	}

	.err {
		color: #ef4444;
		padding: 12px 20px;
		font-size: 0.82rem;
		white-space: pre-wrap;
	}

	.status {
		color: #6b7280;
		padding: 12px 20px;
		font-size: 0.82rem;
	}

	.rows {
		overflow-y: auto;
		padding: 12px 20px 40px;
		display: flex;
		flex-direction: column;
		gap: 14px;
	}

	.row {
		display: grid;
		grid-template-columns: 48px 1fr 1fr;
		align-items: center;
		gap: 14px;
	}

	.loc {
		font-size: 0.75rem;
		color: #6b7280;
		font-variant-numeric: tabular-nums;
	}

	figure {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	img {
		width: 100%;
		height: auto;
		display: block;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		background: #181b23;
		aspect-ratio: 512 / 344;
		object-fit: cover;
	}

	figcaption {
		font-size: 0.7rem;
		color: #6b7280;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}
</style>
