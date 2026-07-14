<script lang="ts">
	import { normalizeImageData } from '$lib/imageUtils';

	let {
		heSrc,
		ihcSrc,
		dx = 0,
		dy = 0,
		keypoints = [],
		emphasis = null
	}: {
		heSrc: string;
		ihcSrc: string;
		dx?: number;
		dy?: number;
		keypoints?: number[][];
		emphasis?: 'he' | 'ihc' | null;
	} = $props();

	let imgW = $state(0);
	let imgH = $state(0);

	const validKeypoints = $derived(
		keypoints.filter(([kx, ky]) => {
			const ix = kx - dx;
			const iy = ky - dy;
			return ix >= 0 && ix < imgW && iy >= 0 && iy < imgH;
		})
	);

	const KP_R = $derived(imgW * 0.008);

	let canvas = $state<HTMLCanvasElement | null>(null);
	let he = $state<HTMLImageElement | null>(null);
	let ihc = $state<HTMLImageElement | null>(null);
	let heNorm = $state<HTMLCanvasElement | null>(null);
	let ihcNorm = $state<HTMLCanvasElement | null>(null);

	function createNormalizedCanvas(img: HTMLImageElement): HTMLCanvasElement {
		const c = document.createElement('canvas');
		c.width = img.naturalWidth;
		c.height = img.naturalHeight;
		const ctx = c.getContext('2d', { willReadFrequently: true })!;
		ctx.drawImage(img, 0, 0);
		const imageData = ctx.getImageData(0, 0, c.width, c.height);
		normalizeImageData(imageData.data);
		ctx.putImageData(imageData, 0, 0);
		return c;
	}

	function draw() {
		if (!canvas || !heNorm || !ihcNorm) return;
		const w = heNorm.width;
		const h = heNorm.height;
		imgW = w;
		imgH = h;
		canvas.width = w;
		canvas.height = h;
		const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
		ctx.clearRect(0, 0, w, h);
		ctx.drawImage(heNorm, 0, 0);
		ctx.globalAlpha = emphasis === 'ihc' ? 0.85 : emphasis === 'he' ? 0.2 : 0.5;
		ctx.drawImage(ihcNorm, dx, dy);
		ctx.globalAlpha = 1;
	}

	// Lazy-load images; re-load whenever the source URLs change
	$effect(() => {
		if (!canvas) return;
		const currentHeSrc = heSrc;
		const currentIhcSrc = ihcSrc;
		let cancelled = false;

		const observer = new IntersectionObserver(
			(entries) => {
				if (!entries[0].isIntersecting) return;
				observer.disconnect();
				const imgHe = new Image();
				const imgIhc = new Image();
				let loaded = 0;
				function onLoad() {
					if (cancelled) return;
					loaded++;
					if (loaded < 2) return;
					he = imgHe;
					ihc = imgIhc;
				}
				imgHe.onload = onLoad;
				imgIhc.onload = onLoad;
				imgHe.src = currentHeSrc;
				imgIhc.src = currentIhcSrc;
			},
			{ rootMargin: '400px' }
		);
		observer.observe(canvas);
		return () => { cancelled = true; observer.disconnect(); };
	});

	// Build brightness-normalized canvases when the source images load
	$effect(() => {
		if (he && ihc) {
			heNorm = createNormalizedCanvas(he);
			ihcNorm = createNormalizedCanvas(ihc);
		}
	});

	// Redraw whenever normalized images or displacement/emphasis changes
	$effect(() => {
		if (heNorm && ihcNorm) draw();
	});

	$effect(() => {
		dx; dy; emphasis;
		if (heNorm && ihcNorm) draw();
	});
</script>

<div class="wrap">
	<canvas bind:this={canvas}></canvas>
	{#if validKeypoints.length > 0}
		<svg class="overlay" viewBox="0 0 {imgW} {imgH}" preserveAspectRatio="none">
			{#each validKeypoints as kp}
				<circle cx={kp[0]} cy={kp[1]} r={KP_R} fill="#facc15" fill-opacity="0.7" stroke="none" />
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

	.overlay {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
	}
</style>
