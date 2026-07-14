/**
 * Builds a summed-area table (SAT) for a Float32Array image.
 * sat[(y+1)*(w+1)+(x+1)] = sum of all values in the rectangle [0,0]..[x,y]
 */
function buildSAT(gray: Float32Array, w: number, h: number): Float64Array {
	const sat = new Float64Array((w + 1) * (h + 1));
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			sat[(y + 1) * (w + 1) + (x + 1)] =
				gray[y * w + x] +
				sat[y * (w + 1) + (x + 1)] +
				sat[(y + 1) * (w + 1) + x] -
				sat[y * (w + 1) + x];
		}
	}
	return sat;
}

function rectSum(sat: Float64Array, w: number, x1: number, y1: number, x2: number, y2: number): number {
	return (
		sat[(y2 + 1) * (w + 1) + (x2 + 1)] -
		sat[y1 * (w + 1) + (x2 + 1)] -
		sat[(y2 + 1) * (w + 1) + x1] +
		sat[y1 * (w + 1) + x1]
	);
}

/**
 * Computes mean Local Normalised Cross-Correlation between two grayscale images.
 * squared=false → LNCC ∈ [-1, 1]:  num / sqrt(den1 * den2)
 * squared=true  → LNCC² ∈ [0, 1]:  num² / (den1 * den2)
 */
export function computeLNCC(
	gray1: Float32Array,
	gray2: Float32Array,
	w: number,
	h: number,
	patchSize: number,
	squared = false
): number {
	const r = Math.floor(patchSize / 2);
	const area = patchSize * patchSize;

	const g1sq = new Float32Array(w * h);
	const g2sq = new Float32Array(w * h);
	const g12  = new Float32Array(w * h);
	for (let i = 0; i < w * h; i++) {
		g1sq[i] = gray1[i] * gray1[i];
		g2sq[i] = gray2[i] * gray2[i];
		g12[i]  = gray1[i] * gray2[i];
	}

	const sat1   = buildSAT(gray1, w, h);
	const sat2   = buildSAT(gray2, w, h);
	const sat1sq = buildSAT(g1sq, w, h);
	const sat2sq = buildSAT(g2sq, w, h);
	const sat12  = buildSAT(g12, w, h);

	let sum = 0;
	let count = 0;

	for (let y = r; y < h - r; y++) {
		for (let x = r; x < w - r; x++) {
			const x1 = x - r, y1 = y - r, x2 = x + r, y2 = y + r;
			const s1   = rectSum(sat1,   w, x1, y1, x2, y2);
			const s2   = rectSum(sat2,   w, x1, y1, x2, y2);
			const s1sq = rectSum(sat1sq, w, x1, y1, x2, y2);
			const s2sq = rectSum(sat2sq, w, x1, y1, x2, y2);
			const s12  = rectSum(sat12,  w, x1, y1, x2, y2);

			const mu1 = s1 / area;
			const mu2 = s2 / area;
			const num  = s12  - area * mu1 * mu2;
			const den1 = Math.max(0, s1sq - area * mu1 * mu1);
			const den2 = Math.max(0, s2sq - area * mu2 * mu2);

			if (squared) {
				const den = den1 * den2;
				if (den > 1e-6) { sum += (num * num) / den; count++; }
			} else {
				const den = Math.sqrt(den1 * den2);
				if (den > 1e-6) { sum += num / den; count++; }
			}
		}
	}

	return count > 0 ? sum / count : 0;
}

/**
 * Patch-based SSIM averaged over the tile.
 * Reuses the same five SATs as computeLNCC (mean, mean², cross-product).
 * C1, C2 tuned to the normalized intensity range [0, 255] with std ≈ 64:
 *   C1 = (0.01 * 255)² ≈ 6.5   C2 = (0.03 * 255)² ≈ 58.5
 * Returns mean SSIM ∈ [-1, 1].
 */
export function computeSSIM(
	gray1: Float32Array,
	gray2: Float32Array,
	w: number,
	h: number,
	patchSize: number
): number {
	const C1 = 6.5025;
	const C2 = 58.5225;
	const r = Math.floor(patchSize / 2);
	const area = patchSize * patchSize;

	const g1sq = new Float32Array(w * h);
	const g2sq = new Float32Array(w * h);
	const g12  = new Float32Array(w * h);
	for (let i = 0; i < w * h; i++) {
		g1sq[i] = gray1[i] * gray1[i];
		g2sq[i] = gray2[i] * gray2[i];
		g12[i]  = gray1[i] * gray2[i];
	}

	const sat1   = buildSAT(gray1, w, h);
	const sat2   = buildSAT(gray2, w, h);
	const sat1sq = buildSAT(g1sq, w, h);
	const sat2sq = buildSAT(g2sq, w, h);
	const sat12  = buildSAT(g12, w, h);

	let sum = 0;
	let count = 0;

	for (let y = r; y < h - r; y++) {
		for (let x = r; x < w - r; x++) {
			const x1 = x - r, y1 = y - r, x2 = x + r, y2 = y + r;
			const mu1  = rectSum(sat1,   w, x1, y1, x2, y2) / area;
			const mu2  = rectSum(sat2,   w, x1, y1, x2, y2) / area;
			const sig1 = Math.max(0, rectSum(sat1sq, w, x1, y1, x2, y2) / area - mu1 * mu1);
			const sig2 = Math.max(0, rectSum(sat2sq, w, x1, y1, x2, y2) / area - mu2 * mu2);
			const cov  = rectSum(sat12,  w, x1, y1, x2, y2) / area - mu1 * mu2;

			const num = (2 * mu1 * mu2 + C1) * (2 * cov  + C2);
			const den = (mu1*mu1 + mu2*mu2 + C1) * (sig1 + sig2 + C2);
			sum += num / den;
			count++;
		}
	}

	return count > 0 ? sum / count : 0;
}

/**
 * NGF score ∈ [0, 1]: mean over all interior pixels of
 *   sim(p) = (g_a · g_b)² / (|g_a|² · |g_b|² + ε)
 * where g_a, g_b are Sobel gradient vectors of gray1, gray2 at pixel p.
 */
export function computeNGF(
	gray1: Float32Array,
	gray2: Float32Array,
	w: number,
	h: number,
	eps = 1e-3
): number {
	let sum = 0;
	let count = 0;
	for (let y = 1; y < h - 1; y++) {
		for (let x = 1; x < w - 1; x++) {
			const tl1 = gray1[(y-1)*w+(x-1)], tc1 = gray1[(y-1)*w+x], tr1 = gray1[(y-1)*w+(x+1)];
			const ml1 = gray1[y*w+(x-1)],                               mr1 = gray1[y*w+(x+1)];
			const bl1 = gray1[(y+1)*w+(x-1)], bc1 = gray1[(y+1)*w+x], br1 = gray1[(y+1)*w+(x+1)];
			const ax = -tl1 - 2*ml1 - bl1 + tr1 + 2*mr1 + br1;
			const ay = -tl1 - 2*tc1 - tr1 + bl1 + 2*bc1 + br1;

			const tl2 = gray2[(y-1)*w+(x-1)], tc2 = gray2[(y-1)*w+x], tr2 = gray2[(y-1)*w+(x+1)];
			const ml2 = gray2[y*w+(x-1)],                               mr2 = gray2[y*w+(x+1)];
			const bl2 = gray2[(y+1)*w+(x-1)], bc2 = gray2[(y+1)*w+x], br2 = gray2[(y+1)*w+(x+1)];
			const bx = -tl2 - 2*ml2 - bl2 + tr2 + 2*mr2 + br2;
			const by = -tl2 - 2*tc2 - tr2 + bl2 + 2*bc2 + br2;

			const dot = ax * bx + ay * by;
			const den = (ax*ax + ay*ay) * (bx*bx + by*by) + eps;
			sum += (dot * dot) / den;
			count++;
		}
	}
	return count > 0 ? sum / count : 0;
}

/**
 * Loads an image URL and returns a z-score-normalised grayscale Float32Array.
 * Browser-only (uses HTMLImageElement + Canvas).
 */
export function loadGray(src: string): Promise<{ gray: Float32Array; w: number; h: number }> {
	return new Promise((resolve) => {
		const img = new Image();
		img.onload = () => {
			const w = img.naturalWidth, h = img.naturalHeight;
			const canvas = document.createElement('canvas');
			canvas.width = w; canvas.height = h;
			const ctx = canvas.getContext('2d')!;
			ctx.drawImage(img, 0, 0);
			const d = ctx.getImageData(0, 0, w, h).data;
			normalizeImageData(d);
			const gray = new Float32Array(w * h);
			for (let i = 0; i < w * h; i++) gray[i] = (d[i*4] + d[i*4+1] + d[i*4+2]) / 3;
			resolve({ gray, w, h });
		};
		img.src = src;
	});
}

function shiftGrayInner(gray: Float32Array, w: number, h: number, dx: number, dy: number): Float32Array {
	const out = new Float32Array(w * h);
	const rdx = Math.round(dx), rdy = Math.round(dy);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			const srcX = x - rdx, srcY = y - rdy;
			if (srcX >= 0 && srcX < w && srcY >= 0 && srcY < h)
				out[y * w + x] = gray[srcY * w + srcX];
		}
	}
	return out;
}

export function shiftGray(gray: Float32Array, w: number, h: number, dx: number, dy: number): Float32Array {
	return shiftGrayInner(gray, w, h, dx, dy);
}


export const NORM_MEAN = 128;
export const NORM_STD = 64;

export function normalizeImageData(d: Uint8ClampedArray): void {
	const n = d.length / 4;
	let sum = 0;
	for (let i = 0; i < n; i++) sum += (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;
	const mean = sum / n;

	let variance = 0;
	for (let i = 0; i < n; i++) {
		const diff = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3 - mean;
		variance += diff * diff;
	}
	const std = Math.sqrt(variance / n) || 1;

	for (let i = 0; i < d.length; i += 4) {
		for (let c = 0; c < 3; c++) {
			d[i + c] = Math.min(255, Math.max(0, ((d[i + c] - mean) / std) * NORM_STD + NORM_MEAN));
		}
	}
}

export function applySobel(d: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
	const gray = new Float32Array(w * h);
	for (let i = 0; i < w * h; i++) gray[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;

	const mag = new Float32Array(w * h);
	let maxG = 0;
	for (let y = 1; y < h - 1; y++) {
		for (let x = 1; x < w - 1; x++) {
			const tl = gray[(y-1)*w+(x-1)], tc = gray[(y-1)*w+x], tr = gray[(y-1)*w+(x+1)];
			const ml = gray[y*w+(x-1)],                             mr = gray[y*w+(x+1)];
			const bl = gray[(y+1)*w+(x-1)], bc = gray[(y+1)*w+x], br = gray[(y+1)*w+(x+1)];
			const gx = -tl - 2*ml - bl + tr + 2*mr + br;
			const gy = -tl - 2*tc - tr + bl + 2*bc + br;
			const g = Math.sqrt(gx*gx + gy*gy);
			mag[y*w+x] = g;
			if (g > maxG) maxG = g;
		}
	}

	const out = new Uint8ClampedArray(d.length);
	const scale = maxG > 0 ? 255 / maxG : 1;
	for (let i = 0; i < w * h; i++) {
		const v = mag[i] * scale;
		out[i*4] = out[i*4+1] = out[i*4+2] = v;
		out[i*4+3] = 255;
	}
	return out;
}
