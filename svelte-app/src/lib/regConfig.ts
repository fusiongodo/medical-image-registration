export type Lam = 'fft' | 'superpoint_glue';
export type FieldEstimator = 'tps' | 'wendland' | 'bspline';

export interface RegConfig {
	lam: Lam;
	fieldEstimator: FieldEstimator;
}

export const DEFAULT_REG_CONFIG: RegConfig = {
	lam: 'fft',
	fieldEstimator: 'tps'
};

export const LAM_OPTIONS: { id: Lam; label: string; hint: string }[] = [
	{ id: 'fft', label: 'FFT phase correlation', hint: 'Per-tile residual via Hann-windowed phase correlation + PSR.' },
	{
		id: 'superpoint_glue',
		label: 'SuperPoint + LightGlue',
		hint: 'Matcher not implemented yet — field sets can still be created on this branch.'
	}
];

export const ESTIMATOR_OPTIONS: { id: FieldEstimator; label: string; hint: string }[] = [
	{ id: 'tps', label: 'Thin-plate spline', hint: 'Global τ-gated TPS (current default).' },
	{ id: 'wendland', label: 'Wendland RBF', hint: 'Compactly supported C² RBF — local influence.' },
	{
		id: 'bspline',
		label: 'B-spline FFD',
		hint: 'Cubic free-form deformation on a uniform control grid.'
	}
];

function parseEstimator(v: unknown): FieldEstimator {
	if (v === 'wendland' || v === 'bspline' || v === 'tps') return v;
	return 'tps';
}

export function storageKey(pairId: number): string {
	return `mvrRegConfig:${pairId}`;
}

export function loadRegConfig(pairId: number): RegConfig {
	try {
		const raw = localStorage.getItem(storageKey(pairId));
		if (!raw) return { ...DEFAULT_REG_CONFIG };
		const parsed = JSON.parse(raw) as Partial<RegConfig>;
		return {
			lam: parsed.lam === 'superpoint_glue' ? 'superpoint_glue' : 'fft',
			fieldEstimator: parseEstimator(parsed.fieldEstimator)
		};
	} catch {
		return { ...DEFAULT_REG_CONFIG };
	}
}

export function saveRegConfig(pairId: number, config: RegConfig): void {
	try {
		localStorage.setItem(storageKey(pairId), JSON.stringify(config));
	} catch {
		/* ignore */
	}
}

export function branchQuery(config: RegConfig): string {
	return `lam=${encodeURIComponent(config.lam)}&field_estimator=${encodeURIComponent(config.fieldEstimator)}`;
}
