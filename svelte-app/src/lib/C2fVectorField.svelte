<script lang="ts">
	interface TileResult {
		tile_loc: string;
		dx: number;
		dy: number;
		ux: number;
		uy: number;
		prior_dx: number;
		prior_dy: number;
	}

	type VectorMode = 'refinement' | 'prior' | 'result';

	let {
		depth,
		tiles,
		mode,
		selected = null,
		hovered = null,
		onhover,
		onselect
	}: {
		depth: number;
		tiles: TileResult[];
		mode: VectorMode;
		selected?: string | null;
		hovered?: string | null;
		onhover?: (tile_loc: string | null) => void;
		onselect?: (tile_loc: string | null) => void;
	} = $props();

	const SIZE = 460;
	let svgEl = $state<SVGSVGElement | null>(null);
	const grid = $derived(2 ** depth);
	const cell = $derived(SIZE / grid);

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

	function vectorFor(t: TileResult): { dx: number; dy: number } {
		switch (mode) {
			case 'refinement':
				return { dx: t.ux, dy: t.uy };
			case 'prior':
				return { dx: t.prior_dx, dy: t.prior_dy };
			case 'result':
			default:
				return { dx: t.dx, dy: t.dy };
		}
	}

	const arrows = $derived.by(() => {
		// Arrows originate at the tile centre and are allowed to extend
		// beyond the tile so small displacements remain readable.
		const maxArrow = cell * 1.25;
		const minArrow = 3;
		const mags = tiles.map((t) => {
			const v = vectorFor(t);
			return Math.hypot(v.dx, v.dy);
		});
		const maxMag = Math.max(1e-6, ...mags);
		return tiles.map((t, i) => {
			const [xi, yi] = t.tile_loc.split('_').map((n) => parseInt(n, 10));
			const v = vectorFor(t);
			const mag = mags[i];
			const cx = (xi + 0.5) * cell;
			const cy = (yi + 0.5) * cell;
			let len = (mag / maxMag) * maxArrow;
			if (len > 0 && len < minArrow) len = minArrow;
			if (len > maxArrow) len = maxArrow;
			let x2 = cx;
			let y2 = cy;
			if (mag > 1e-6) {
				const scale = len / mag;
				x2 = cx + v.dx * scale;
				y2 = cy + v.dy * scale;
			}
			return {
				cx,
				cy,
				x1: cx,
				y1: cy,
				x2,
				y2,
				mag,
				tile_loc: t.tile_loc,
				title: `${t.tile_loc}  ${mode} vector  |Δ|=${mag.toFixed(1)}px`
			};
		});
	});

	const frameCells = $derived.by(() => {
		const out: { x: number; y: number; stroke: string }[] = [];
		if (selected) {
			const t = tileMap.get(selected);
			if (t) {
				const [xi, yi] = t.tile_loc.split('_').map((n) => parseInt(n, 10));
				out.push({ x: xi * cell, y: yi * cell, stroke: '#22d3ee' });
			}
		}
		if (hovered && hovered !== selected) {
			const t = tileMap.get(hovered);
			if (t) {
				const [xi, yi] = t.tile_loc.split('_').map((n) => parseInt(n, 10));
				out.push({ x: xi * cell, y: yi * cell, stroke: '#e8eaf0' });
			}
		}
		return out;
	});
</script>

<svg
	width={SIZE}
	height={SIZE}
	class="vector-field"
	role="button"
	tabindex="0"
	aria-label="Tile displacement vector field. Click a cell to select it."
	bind:this={svgEl}
	onmousemove={handleMove}
	onmouseleave={handleLeave}
	onclick={handleClick}
	onkeydown={handleKeyDown}
>
	<defs>
		<marker
			id="vf-arrow"
			markerWidth="4"
			markerHeight="4"
			refX="3.5"
			refY="2"
			orient="auto"
			markerUnits="userSpaceOnUse"
		>
			<path d="M0,0 L4,2 L0,4 Z" fill="#9ca3af" />
		</marker>
	</defs>
	<rect x="0" y="0" width={SIZE} height={SIZE} class="bg" />
	{#each arrows as a}
		<line
			x1={a.x1}
			y1={a.y1}
			x2={a.x2}
			y2={a.y2}
			stroke="#9ca3af"
			stroke-width="1"
			marker-end="url(#vf-arrow)"
		>
			<title>{a.title}</title>
		</line>
	{/each}
	{#each arrows as a}
		{#if a.mag > 1e-6}
			<circle cx={a.cx} cy={a.cy} r="1.2" fill="#9ca3af" />
		{/if}
	{/each}
	{#each frameCells as c}
		<rect
			x={c.x + 0.75}
			y={c.y + 0.75}
			width={cell - 1.5}
			height={cell - 1.5}
			fill="none"
			stroke={c.stroke}
			stroke-width="2"
		/>
	{/each}
</svg>

<style>
	.vector-field {
		display: block;
		flex-shrink: 0;
	}
	.bg {
		fill: #131520;
		stroke: #2a2d3a;
		stroke-width: 1;
	}
</style>
