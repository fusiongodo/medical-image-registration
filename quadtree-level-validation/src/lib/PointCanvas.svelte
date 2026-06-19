<script lang="ts">
	import { normalizeImageData } from '$lib/imageUtils';

	interface Point { x: number; y: number; }

	let {
		src,
		active = false,
		points = [],
		onpoint
	}: {
		src: string;
		active?: boolean;
		points?: Point[];
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
					const ctx = canvas!.getContext('2d')!;
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
</script>

<div class="wrap" class:active>
	<canvas bind:this={canvas} onclick={handleClick}></canvas>
	{#if points.length > 0}
		<svg class="overlay" viewBox="0 0 {naturalW} {naturalH}" preserveAspectRatio="none">
			{#each points as pt, i}
				<circle cx={pt.x} cy={pt.y} r={naturalW * 0.018} fill={COLORS[i % 2]} stroke="#000" stroke-width={naturalW * 0.004} opacity="0.85" />
				<text x={pt.x} y={pt.y + naturalW * 0.006} text-anchor="middle" dominant-baseline="middle" font-size={naturalW * 0.035} fill="#000" font-weight="bold">{i + 1}</text>
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
