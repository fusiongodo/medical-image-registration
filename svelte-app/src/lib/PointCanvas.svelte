<script lang="ts">
	import { normalizeImageData } from '$lib/imageUtils';

	interface Point { x: number; y: number; }

	let {
		src,
		active = false,
		points = [],
		keypoints = [],
		width = 269,
		height = 180,
		markerScale = 1,
		onpoint
	}: {
		src: string;
		active?: boolean;
		points?: Point[];
		keypoints?: number[][];
		width?: number;
		height?: number;
		markerScale?: number;
		onpoint?: (x: number, y: number) => void;
	} = $props();

	let canvas = $state<HTMLCanvasElement | null>(null);
	let naturalW = $state(0);
	let naturalH = $state(0);

	$effect(() => {
		if (!canvas) return;
		const observer = new IntersectionObserver(
			(entries) => {
				if (!entries[0].isIntersecting) return;
				observer.disconnect();
				const img = new Image();
				img.onload = () => {
					naturalW = img.naturalWidth;
					naturalH = img.naturalHeight;
					canvas!.width = naturalW;
					canvas!.height = naturalH;
					const ctx = canvas!.getContext('2d', { willReadFrequently: true })!;
					ctx.drawImage(img, 0, 0);
					const imageData = ctx.getImageData(0, 0, naturalW, naturalH);
					normalizeImageData(imageData.data);
					ctx.putImageData(imageData, 0, 0);
				};
				img.src = src;
			},
			{ rootMargin: '400px' }
		);
		observer.observe(canvas);
		return () => observer.disconnect();
	});

	function handleClick(e: MouseEvent) {
		if (!active || !onpoint || !canvas) return;
		const rect = canvas.getBoundingClientRect();
		const scaleX = naturalW / rect.width;
		const scaleY = naturalH / rect.height;
		const x = (e.clientX - rect.left) * scaleX;
		const y = (e.clientY - rect.top) * scaleY;
		onpoint(Math.round(x), Math.round(y));
	}

	const COLORS = ['#60a5fa', '#f97316'];
	const KP_R = $derived(naturalW * 0.008);
	const DOT_R = $derived(naturalW * 0.018 * markerScale);
	const DOT_STROKE = $derived(naturalW * 0.004 * markerScale);
	const DOT_FONT = $derived(naturalW * 0.035 * markerScale);
	const DOT_TEXT_DY = $derived(naturalW * 0.006 * markerScale);
</script>

<div class="wrap" class:active style="width:{width}px;height:{height}px">
	<canvas bind:this={canvas} onclick={handleClick} style="width:{width}px;height:{height}px"></canvas>
	{#if keypoints.length > 0 || points.length > 0}
		<svg class="overlay" viewBox="0 0 {naturalW} {naturalH}" preserveAspectRatio="none">
			{#each keypoints as kp}
				<circle cx={kp[0]} cy={kp[1]} r={KP_R} fill="#facc15" fill-opacity="0.7" stroke="none" />
			{/each}
			{#each points as pt, i}
				<circle cx={pt.x} cy={pt.y} r={DOT_R} fill={COLORS[i % 2]} stroke="#000" stroke-width={DOT_STROKE} opacity="0.85" />
				<text x={pt.x} y={pt.y + DOT_TEXT_DY} text-anchor="middle" dominant-baseline="middle" font-size={DOT_FONT} fill="#000" font-weight="bold">{i + 1}</text>
			{/each}
		</svg>
	{/if}
</div>

<style>
	.wrap {
		position: relative;
		display: block;
		height: 180px;
		width: 269px;
	}

	canvas {
		display: block;
		height: 180px;
		width: 269px;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #0f1117;
	}

	.wrap.active canvas {
		cursor: crosshair;
		border-color: #6366f1;
		box-shadow: 0 0 0 2px #6366f140;
	}

	.overlay {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
	}
</style>
