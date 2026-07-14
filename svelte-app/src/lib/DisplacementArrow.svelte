<script lang="ts">
	let {
		dx,
		dy,
		maxLen = 28
	}: {
		dx: number;
		dy: number;
		maxLen?: number;
	} = $props();

	const id = `arrow-${Math.random().toString(36).slice(2)}`;
	const mag = $derived(Math.hypot(dx, dy));
	const scale = $derived(mag > 0 ? Math.min(maxLen, mag) / mag : 0);
	const ax = $derived(40 + dx * scale);
	const ay = $derived(76 + dy * scale);
</script>

<svg width="80" height="100" class="disp-svg">
	<defs>
		<marker id={id} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
			<path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
		</marker>
	</defs>
	<circle cx="40" cy="76" r="2.5" fill="#9ca3af" />
	{#if mag > 0.1}
		<line x1="40" y1="76" x2={ax} y2={ay}
			stroke="#9ca3af" stroke-width="1.5" marker-end="url(#{id})" />
	{/if}
	<text x="40" y="96" text-anchor="middle" class="disp-label">{mag.toFixed(1)}px</text>
</svg>

<style>
	.disp-svg { display: block; }
	.disp-label {
		font-size: 0.65rem;
		fill: #9ca3af;
		font-variant-numeric: tabular-nums;
	}
</style>
