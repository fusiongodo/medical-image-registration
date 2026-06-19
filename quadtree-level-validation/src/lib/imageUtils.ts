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
