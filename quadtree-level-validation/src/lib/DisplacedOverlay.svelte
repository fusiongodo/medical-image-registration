<script lang="ts">
	let {
		heSrc,
		ihcSrc,
		dx = 0,
		dy = 0
	}: { heSrc: string; ihcSrc: string; dx?: number; dy?: number } = $props();

	let canvas = $state<HTMLCanvasElement | null>(null);
	let he = $state<HTMLImageElement | null>(null);
	let ihc = $state<HTMLImageElement | null>(null);

	function stretchContrast(d: Uint8ClampedArray) {
		const n = d.length / 4;
		const luma = new Float32Array(n);
		for (let i = 0; i < n; i++) luma[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;
		const sorted = Float32Array.from(luma).sort();
		const lo = sorted[Math.floor(n * 0.02)];
		const hi = sorted[Math.floor(n * 0.98)];
		const range = hi - lo || 1;
		for (let i = 0; i < d.length; i += 4) {
			for (let c = 0; c < 3; c++) {
				d[i + c] = Math.min(255, Math.max(0, ((d[i + c] - lo) / range) * 255));
			}
		}
	}

	function draw() {
		if (!canvas || !he || !ihc) return;
		const w = he.naturalWidth;
		const h = he.naturalHeight;
		canvas.width = w;
		canvas.height = h;
		const ctx = canvas.getContext('2d')!;
		ctx.drawImage(he, 0, 0);
		ctx.globalAlpha = 0.5;
		ctx.drawImage(ihc, dx, dy);
		ctx.globalAlpha = 1;
		const imageData = ctx.getImageData(0, 0, w, h);
		stretchContrast(imageData.data);
		ctx.putImageData(imageData, 0, 0);
	}

	// Lazy-load images once
	$effect(() => {
		if (!canvas) return;
		const observer = new IntersectionObserver(
			(entries) => {
				if (!entries[0].isIntersecting) return;
				observer.disconnect();
				const imgHe = new Image();
				const imgIhc = new Image();
				let loaded = 0;
				function onLoad() {
					loaded++;
					if (loaded < 2) return;
					he = imgHe;
					ihc = imgIhc;
				}
				imgHe.onload = onLoad;
				imgIhc.onload = onLoad;
				imgHe.src = heSrc;
				imgIhc.src = ihcSrc;
			},
			{ rootMargin: '400px' }
		);
		observer.observe(canvas);
		return () => observer.disconnect();
	});

	// Redraw whenever images or displacement changes
	$effect(() => {
		if (he && ihc) draw();
	});

	$effect(() => {
		dx; dy;
		if (he && ihc) draw();
	});
</script>

<canvas bind:this={canvas}></canvas>

<style>
	canvas {
		display: block;
		height: 180px;
		width: 269px;
		border-radius: 4px;
		border: 1px solid #2a2d3a;
		background: #0f1117;
	}
</style>
