<script lang="ts">
	interface TileResult {
		tile_loc: string;
		psr: number;
		residual: number;
		kept: boolean;
		excluded?: boolean;
		masked?: boolean;
		annotated?: 'approve' | 'correct' | 'exclude' | null;
		dx: number;
		dy: number;
	}

	let {
		depth,
		tiles,
		tau,
		seed = [],
		selected = null,
		onhover,
		onselect
	}: {
		depth: number;
		tiles: TileResult[];
		tau: number;
		seed?: string[];
		selected?: string | null;
		onhover?: (tile_loc: string | null) => void;
		onselect?: (tile_loc: string | null) => void;
	} = $props();

	const SIZE = 460;
	let svgEl = $state<SVGSVGElement | null>(null);
	const grid = $derived(2 ** depth);
	const cell = $derived(SIZE / grid);
	const seedSet = $derived(new Set(seed));
	const tileMap = $derived(
		tiles.reduce((m, t) => {
			m.set(t.tile_loc, t);
			return m;
		}, new Map<string, TileResult>())
	);

	let lastHoverTile = $state<string | null>(null);

	function handleMove(e: MouseEvent) {
		if (!svgEl || !onhover) return;
		const rect = svgEl.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const y = e.clientY - rect.top;
		const xi = Math.floor(x / cell);
		const yi = Math.floor(y / cell);
		if (xi < 0 || xi >= grid || yi < 0 || yi >= grid) {
			onhover(null);
			return;
		}
		const tileLoc = `${xi}_${yi}`;
		const tile = tileMap.has(tileLoc) ? tileLoc : null;
		lastHoverTile = tile;
		onhover(tile);
	}

	function handleLeave() {
		onhover?.(null);
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			if (lastHoverTile) onselect?.(lastHoverTile);
		}
	}

	function handleClick(e: MouseEvent) {
		if (!svgEl || !onselect) return;
		const rect = svgEl.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const y = e.clientY - rect.top;
		const xi = Math.floor(x / cell);
		const yi = Math.floor(y / cell);
		if (xi < 0 || xi >= grid || yi < 0 || yi >= grid) {
			onselect(null);
			return;
		}
		const tileLoc = `${xi}_${yi}`;
		onselect(tileMap.has(tileLoc) ? tileLoc : null);
	}

	// The residual scale used for colour intensity: relative to tau.
	const cells = $derived.by(() => {
		return tiles.map((t) => {
			const [xi, yi] = t.tile_loc.split('_').map((n) => parseInt(n, 10));
			// intensity 0..1: how strongly the cell asserts its state
			const ratio = tau > 0 ? t.residual / tau : 0;
			const isMasked = t.masked === true;
			const isExcluded = t.annotated === 'exclude' || t.excluded;
			if (isMasked) {
				// masked out (propagated by index): neutral grey, no green/red, but
				// still hoverable/selectable so it can be unmasked.
				return {
					x: xi * cell,
					y: yi * cell,
					cx: xi * cell + cell / 2,
					cy: yi * cell + cell / 2,
					fill: '#94a3b8',
					opacity: 0.35,
					stroke: null,
					strokeWidth: 0,
					strokeOpacity: 0,
					label: 'M',
					fontSize: Math.min(cell * 0.55, 12),
					title: `${t.tile_loc}  masked out`,
					isSelected: t.tile_loc === selected
				};
			}
			// excluded tiles keep the residual-based green->red mapping (marked with an "E")
			const asKept = isExcluded ? ratio <= 1 : t.kept;
			let opacity: number;
			let fill: string;
			if (asKept) {
				// solid green for near-perfect agreement, fading toward tau
				fill = '#22c55e';
				opacity = 0.9 - 0.5 * Math.min(1, ratio);
			} else {
				// deeper red the further above tau
				fill = '#ef4444';
				opacity = 0.5 + 0.45 * Math.min(1, (ratio - 1) / 3);
			}
			const stateLabel = isExcluded ? 'excluded' : asKept ? 'kept' : 'rejected';
			const isSeed = seedSet.has(t.tile_loc);
			const stroke = isExcluded
				? null
				: t.annotated === 'correct' ? '#a5b4fc'
				: t.annotated === 'approve' ? '#eab308'
				: isSeed ? '#e8eaf0'
				: null;
			const strokeWidth = isExcluded ? 0 : t.annotated ? 2 : isSeed ? 1 : 0;
			const strokeOpacity = t.annotated ? 1 : 0.5;
			const isSelected = t.tile_loc === selected;
			return {
				x: xi * cell,
				y: yi * cell,
				cx: xi * cell + cell / 2,
				cy: yi * cell + cell / 2,
				fill,
				opacity,
				stroke,
				strokeWidth,
				strokeOpacity,
				label: isExcluded ? 'E' : null,
				fontSize: Math.min(cell * 0.55, 12),
				title: `${t.tile_loc}  psr=${t.psr.toFixed(1)}  res=${t.residual.toExponential(2)}  ${stateLabel}`,
				isSelected
			};
		});
	});
</script>

<svg
	width={SIZE}
	height={SIZE}
	class="heat"
	role="button"
	tabindex="0"
	aria-label="Coarse-to-fine residual heatmap. Click a cell to select it."
	bind:this={svgEl}
	onmousemove={handleMove}
	onmouseleave={handleLeave}
	onclick={handleClick}
	onkeydown={handleKeyDown}
>
	<rect x="0" y="0" width={SIZE} height={SIZE} class="bg" />
	{#each cells as c}
		<rect x={c.x} y={c.y} width={cell} height={cell} fill={c.fill} opacity={c.opacity}>
			<title>{c.title}</title>
		</rect>
		{#if c.stroke || c.isSelected}
			<rect
				x={c.x + 0.75}
				y={c.y + 0.75}
				width={cell - 1.5}
				height={cell - 1.5}
				fill="none"
				stroke={c.isSelected ? '#22d3ee' : c.stroke}
				stroke-width={c.isSelected ? 3 : c.strokeWidth}
				stroke-opacity={c.isSelected ? 1 : c.strokeOpacity}
			/>
		{/if}
		{#if c.label}
			<text x={c.cx} y={c.cy} class="cell-label" font-size={c.fontSize}>{c.label}</text>
		{/if}
	{/each}
</svg>

<style>
	.heat {
		display: block;
		border: 1px solid #2a2d3a;
		border-radius: 4px;
		cursor: crosshair;
		user-select: none;
	}
	.bg { fill: #0f1117; }
	.cell-label {
		fill: #0f1117;
		font-weight: 700;
		text-anchor: middle;
		dominant-baseline: central;
		pointer-events: none;
	}
</style>
